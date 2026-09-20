"""Network tests — matching, priority, errors, latency, activity."""

import asyncio
import time

from arken.network import (
    MockPriority,
    MockResponse,
    MockRule,
    NetworkMocker,
    NetworkRequest,
)


def run(coro):
    return asyncio.run(coro)


class TestMatch:
    def test_match_returns_response(self):
        m = NetworkMocker()
        m.add_rule(MockRule("h", "/api/health", MockResponse(200, {"ok": True})))
        r = run(m.intercept(NetworkRequest("GET", "http://x/api/health")))
        assert r is not None and r.ok and r.body == {"ok": True}

    def test_no_match_returns_none(self):
        m = NetworkMocker()
        assert run(m.intercept(NetworkRequest("GET", "http://x/other"))) is None

    def test_method_filter(self):
        m = NetworkMocker()
        m.add_rule(MockRule("p", "/api", MockResponse(200), method="POST"))
        assert run(m.intercept(NetworkRequest("GET", "http://x/api"))) is None
        assert run(m.intercept(NetworkRequest("POST", "http://x/api"))).ok

    def test_regex_pattern(self):
        m = NetworkMocker()
        m.add_rule(MockRule("u", r"re:/users/\d+", MockResponse(200)))
        assert run(m.intercept(NetworkRequest("GET", "http://x/users/42"))).ok
        assert run(m.intercept(NetworkRequest("GET", "http://x/users"))) is None

    def test_priority_wins(self):
        m = NetworkMocker()
        m.add_rule(MockRule("low", "/api", MockResponse(200, "low"), priority=MockPriority.LOW))
        m.add_rule(MockRule("high", "/api", MockResponse(200, "high"), priority=MockPriority.HIGH))
        r = run(m.intercept(NetworkRequest("GET", "http://x/api")))
        assert r.body == "high"

    def test_times_exhausts(self):
        m = NetworkMocker()
        m.add_rule(MockRule("once", "/api", MockResponse(200), times=1))
        run(m.intercept(NetworkRequest("GET", "http://x/api")))
        assert run(m.intercept(NetworkRequest("GET", "http://x/api"))) is None


class TestBehavior:
    def test_error_response_not_ok(self):
        m = NetworkMocker()
        m.add_rule(MockRule("down", "/api", MockResponse(error="conn refused")))
        r = run(m.intercept(NetworkRequest("GET", "http://x/api")))
        assert not r.ok and r.error == "conn refused"

    def test_latency_waits(self):
        m = NetworkMocker()
        m.add_rule(MockRule("slow", "/api", MockResponse(200, latency_ms=50)))
        start = time.monotonic()
        run(m.intercept(NetworkRequest("GET", "http://x/api")))
        assert (time.monotonic() - start) >= 0.04

    def test_stats_and_activity(self):
        m = NetworkMocker()
        m.add_rule(MockRule("h", "/api/health", MockResponse(200)))
        run(m.intercept(NetworkRequest("GET", "http://x/api/health")))
        run(m.intercept(NetworkRequest("GET", "http://x/miss")))
        s = m.stats().to_dict()
        assert s == {
            "interceptions": 1,
            "passthrough": 1,
            "rule_hits": {"h": 1},
        }
        assert m.activity[0]["rule"] == "h"
        assert m.activity[1]["rule"] is None

    def test_remove_and_clear(self):
        m = NetworkMocker()
        m.add_rule(MockRule("h", "/api", MockResponse(200)))
        assert m.remove_rule("h") is True
        assert m.remove_rule("h") is False
        m.add_rule(MockRule("h", "/api", MockResponse(200)))
        m.clear()
        assert m.stats().interceptions == 0
