"""Network mocker — stub API responses, errors, and latency in tests.

No browser needed: feed requests through intercept() and get back
canned responses. Record every interception for assertions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MockPriority(Enum):
    """Rule evaluation order — HIGH rules match first."""

    HIGH = 0
    NORMAL = 10
    LOW = 20


@dataclass
class NetworkRequest:
    """An outgoing request to match against rules."""

    method: str = "GET"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    body: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "url": self.url,
            "headers": self.headers,
        }


@dataclass
class MockResponse:
    """Canned response returned for a matched request."""

    status: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str = ""  # set to simulate a network failure

    @property
    def ok(self) -> bool:
        return not self.error and 200 <= self.status < 300

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "body": self.body,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class MockRule:
    """Match requests by URL pattern + method, reply with response."""

    name: str
    url_pattern: str  # substring, or "re:<regex>"
    response: MockResponse = field(default_factory=MockResponse)
    method: str = ""  # empty = any method
    priority: MockPriority = MockPriority.NORMAL
    times: int | None = None  # max interceptions; None = unlimited
    hits: int = 0

    @property
    def exhausted(self) -> bool:
        return self.times is not None and self.hits >= self.times

    def matches(self, request: NetworkRequest) -> bool:
        if self.exhausted:
            return False
        if self.method and self.method.upper() != request.method.upper():
            return False
        if self.url_pattern.startswith("re:"):
            return re.search(self.url_pattern[3:], request.url) is not None
        return self.url_pattern in request.url


@dataclass
class MockStats:
    """Activity summary."""

    interceptions: int = 0
    passthrough: int = 0
    rule_hits: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interceptions": self.interceptions,
            "passthrough": self.passthrough,
            "rule_hits": self.rule_hits,
        }


class NetworkMocker:
    """Holds mock rules and intercepts requests.

    Usage::

        mocker = NetworkMocker()
        mocker.add_rule(MockRule("health", "/api/health",
                                 MockResponse(status=200, body={"ok": True})))
        resp = await mocker.intercept(NetworkRequest("GET", "/api/health"))
        assert resp.ok
    """

    def __init__(self):
        self._rules: list[MockRule] = []
        self._activity: list[dict[str, Any]] = []
        self._stats = MockStats()

    def add_rule(self, rule: MockRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: (r.priority.value, r.name))

    def remove_rule(self, name: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < before

    async def intercept(self, request: NetworkRequest) -> MockResponse | None:
        """First matching rule wins. None = let the request through."""
        import asyncio

        for rule in self._rules:
            if not rule.matches(request):
                continue
            rule.hits += 1
            self._stats.interceptions += 1
            self._stats.rule_hits[rule.name] = self._stats.rule_hits.get(rule.name, 0) + 1
            resp = rule.response
            if resp.latency_ms > 0:
                await asyncio.sleep(resp.latency_ms / 1000)
            self._activity.append(
                {
                    "request": request.to_dict(),
                    "rule": rule.name,
                    "response": resp.to_dict(),
                }
            )
            return resp
        self._stats.passthrough += 1
        self._activity.append({"request": request.to_dict(), "rule": None})
        return None

    @property
    def activity(self) -> list[dict[str, Any]]:
        return list(self._activity)

    def stats(self) -> MockStats:
        return self._stats

    def clear(self) -> None:
        self._rules.clear()
        self._activity.clear()
        self._stats = MockStats()
