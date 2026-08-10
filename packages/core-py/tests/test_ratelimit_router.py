"""Tests for the ratelimit API router (routers/ratelimit.py).

Covers: _RateLimiter (is_allowed, get_wait_time), RatelimitRouter endpoints.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.ratelimit import RatelimitRouter, _RateLimiter  # noqa: E402


def _app(rr: RatelimitRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(rr.router)
    return app


class TestRateLimiter:
    def test_is_allowed_within_limit(self):
        rl = _RateLimiter(requests_per_minute=5)
        assert rl.is_allowed("key1") is True

    def test_is_allowed_at_limit(self):
        rl = _RateLimiter(requests_per_minute=3)
        assert rl.is_allowed("k") is True
        assert rl.is_allowed("k") is True
        assert rl.is_allowed("k") is True
        assert rl.is_allowed("k") is False

    def test_wait_time_when_not_limited(self):
        rl = _RateLimiter(requests_per_minute=10)
        assert rl.get_wait_time("k") == 0.0

    def test_wait_time_when_limited(self):
        rl = _RateLimiter(requests_per_minute=2)
        rl.is_allowed("k")
        rl.is_allowed("k")
        wt = rl.get_wait_time("k")
        assert wt > 0.0

    def test_separate_keys(self):
        rl = _RateLimiter(requests_per_minute=1)
        assert rl.is_allowed("a") is True
        assert rl.is_allowed("b") is True
        assert rl.is_allowed("a") is False
        assert rl.is_allowed("b") is False

    def test_window_expiry(self):
        rl = _RateLimiter(requests_per_minute=1)
        rl._history["k"] = [time.time() - 61]
        assert rl.is_allowed("k") is True


class TestRatelimitEndpoints:
    def test_status(self):
        rr = RatelimitRouter()
        client = TestClient(_app(rr))
        resp = client.get("/rate-limit/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "requests_per_minute" in data
        assert data["enabled"] is True

    def test_check_allowed(self):
        rr = RatelimitRouter()
        client = TestClient(_app(rr))
        resp = client.get("/rate-limit/check")
        assert resp.status_code == 200
        assert resp.json()["data"]["allowed"] is True
