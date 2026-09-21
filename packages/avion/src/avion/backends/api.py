"""API-only backend — test REST/GraphQL endpoints, no browser.

Stdlib only (urllib), so journey health checks run anywhere.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ApiResponse:
    """A completed HTTP call."""

    status: int
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and 200 <= self.status < 300

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "body": self.body, "error": self.error}


class ApiBackend:
    """Minimal async HTTP client for endpoint journeys.

    Usage::

        backend = ApiBackend(base_url="http://localhost:8000")
        await backend.start()
        resp = await backend.get("/api/health")
        assert resp.ok
        await backend.stop()
    """

    name = "api"

    def __init__(
        self, base_url: str = "", timeout: float = 10.0, headers: dict[str, str] | None = None
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._headers = dict(headers or {})
        self._started = False

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    def _url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        return f"{self._base_url}{path}"

    async def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> ApiResponse:
        import time

        start = time.monotonic()
        data = None
        merged = {**self._headers, **(headers or {})}
        if body is not None and not isinstance(body, (bytes, str)):
            data = json.dumps(body).encode()
            merged.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            data = body.encode()
        req = urllib.request.Request(
            self._url(path), data=data, headers=merged, method=method.upper()
        )
        try:
            import asyncio

            resp = await asyncio.to_thread(urllib.request.urlopen, req, None, self._timeout)
            raw = resp.read()
            status = resp.status
            resp_headers = dict(resp.headers.items())
        except Exception as e:
            code = getattr(e, "code", None)
            if code is None:  # connection-level failure
                return ApiResponse(
                    status=0, error=str(e), duration_ms=(time.monotonic() - start) * 1000
                )
            raw = e.read() if hasattr(e, "read") else b""
            status = code
            resp_headers = (
                dict(getattr(e, "headers", {}).items())
                if hasattr(getattr(e, "headers", None), "items")
                else {}
            )
        duration = (time.monotonic() - start) * 1000
        try:
            parsed = json.loads(raw.decode() or "null")
        except Exception:
            parsed = raw.decode(errors="replace")
        return ApiResponse(status=status, body=parsed, headers=resp_headers, duration_ms=duration)

    async def get(self, path: str, headers: dict[str, str] | None = None) -> ApiResponse:
        return await self.request("GET", path, headers=headers)

    async def post(
        self, path: str, body: Any = None, headers: dict[str, str] | None = None
    ) -> ApiResponse:
        return await self.request("POST", path, body=body, headers=headers)

    async def put(
        self, path: str, body: Any = None, headers: dict[str, str] | None = None
    ) -> ApiResponse:
        return await self.request("PUT", path, body=body, headers=headers)

    async def delete(self, path: str, headers: dict[str, str] | None = None) -> ApiResponse:
        return await self.request("DELETE", path, headers=headers)
