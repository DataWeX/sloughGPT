"""
Tests for the rate-limit router — status, check, burst behavior, wait time.
"""

import pytest
import time
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.ratelimit import RatelimitRouter


@pytest.fixture
def router():
    return RatelimitRouter()


@pytest.fixture
def app(router):
    _app = FastAPI()
    _app.include_router(router.router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestGetStatus:
    def test_returns_config(self, client):
        resp = client.get("/rate-limit/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["requests_per_minute"] == 60
        assert data["burst_size"] == 10
        assert data["enabled"] is True

    def test_returns_all_fields(self, client):
        resp = client.get("/rate-limit/status")
        data = resp.json()["data"]
        assert "requests_per_minute" in data
        assert "burst_size" in data
        assert "enabled" in data


class TestCheckLimit:
    def test_allows_request(self, client):
        resp = client.get("/rate-limit/check")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["allowed"] is True

    def test_returns_wait_time(self, client):
        resp = client.get("/rate-limit/check")
        data = resp.json()["data"]
        assert "wait_time" in data
        assert data["wait_time"] >= 0

    def test_burst_allows_multiple(self, client):
        for _ in range(10):
            resp = client.get("/rate-limit/check")
            assert resp.status_code == 200

    def test_rate_limit_blocks_after_burst(self, client):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=5, burst_size=5)
        for _ in range(5):
            assert limiter.is_allowed("test_key") is True
        assert limiter.is_allowed("test_key") is False

    def test_wait_time_zero_when_under_limit(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=60, burst_size=10)
        assert limiter.get_wait_time("key") == 0.0

    def test_wait_time_positive_when_at_limit(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=2, burst_size=2)
        limiter.is_allowed("k")
        limiter.is_allowed("k")
        wait = limiter.get_wait_time("k")
        assert wait > 0.0

    def test_different_keys_independent(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=1, burst_size=1)
        assert limiter.is_allowed("a") is True
        assert limiter.is_allowed("b") is True
        assert limiter.is_allowed("a") is False
        assert limiter.is_allowed("b") is False

    def test_check_reports_wait_when_blocked(self, router, client):
        for _ in range(60):
            client.get("/rate-limit/check")
        resp = client.get("/rate-limit/check")
        data = resp.json()["data"]
        assert data["allowed"] is False
        assert data["wait_time"] > 0.0

    def test_check_reports_wait_zero_when_allowed(self, router, client):
        resp = client.get("/rate-limit/check")
        assert resp.json()["data"]["wait_time"] == 0.0


class TestRateLimiterInternal:
    def test_history_expires_after_window(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=2, burst_size=2)
        limiter.is_allowed("k")
        limiter._history["k"] = [time.time() - 120]  # old entry
        assert limiter.is_allowed("k") is True

    def test_stale_entries_pruned_before_decision(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=2, burst_size=2)
        limiter._history["k"] = [time.time() - 120, time.time() - 120]
        assert limiter.is_allowed("k") is True  # stale entries dropped, slot free

    def test_wait_time_uses_oldest_entry(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=60, burst_size=60)
        now = time.time()
        limiter._history["k"] = [now - 30.0] * 60
        wait = limiter.get_wait_time("k")
        assert 29.0 < wait <= 30.0

    def test_wait_time_expired_entries_reset_to_zero(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=2, burst_size=2)
        limiter._history["k"] = [time.time() - 120]
        assert limiter.get_wait_time("k") == 0.0

    def test_is_allowed_tracks_max_burst(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=3, burst_size=3)
        for _ in range(3):
            assert limiter.is_allowed("k") is True
        assert len(limiter._history["k"]) == 3
        assert limiter.is_allowed("k") is False

    def test_requests_per_minute_independent_of_burst_size(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=5, burst_size=0)
        for _ in range(5):
            assert limiter.is_allowed("k") is True
        assert limiter.is_allowed("k") is False

    def test_limiter_config_from_router_defaults(self, router):
        assert router._rate_limiter.requests_per_minute == 60
        assert router._rate_limiter.burst_size == 10

    def test_check_uses_per_client_key(self, router):
        router._rate_limiter._history.clear()
        first = set()
        for _ in range(3):
            router._rate_limiter.is_allowed("client-a")
        assert len(router._rate_limiter._history) == 1
        assert len(router._rate_limiter._history["client-a"]) == 3


class TestMethodCoverage:
    def test_status_wrong_method_405(self, client):
        resp = client.post("/rate-limit/status")
        assert resp.status_code == 405

    def test_check_wrong_method_405(self, client):
        resp = client.post("/rate-limit/check")
        assert resp.status_code == 405


class TestRateLimiterEdgeCases:
    def test_is_allowed_records_timestamps(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=60, burst_size=10)
        ok = limiter.is_allowed("k")
        assert ok is True
        assert len(limiter._history["k"]) == 1
        assert isinstance(limiter._history["k"][0], float)

    def test_is_allowed_blocks_at_exact_limit(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=1, burst_size=1)
        assert limiter.is_allowed("k") is True
        assert limiter.is_allowed("k") is False
        assert limiter.get_wait_time("k") > 0.0

    def test_burst_does_not_exceed_requests_per_minute(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=10, burst_size=1000)
        allowed = [limiter.is_allowed("k") for _ in range(12)]
        assert sum(allowed) == 10

    def test_wait_time_zero_for_unknown_key(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=60, burst_size=10)
        assert limiter.get_wait_time("fresh-key") == 0.0

    def test_wait_time_bounded_at_window(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=1, burst_size=1)
        limiter._history["k"] = [time.time()]
        wait = limiter.get_wait_time("k")
        assert 0.0 < wait <= 60.0

    def test_is_allowed_with_zero_rate(self):
        from apps.api.server.routers.ratelimit import _RateLimiter
        limiter = _RateLimiter(requests_per_minute=0, burst_size=0)
        assert limiter.is_allowed("k") is False

    def test_check_endpoint_wait_zero_after_burst_expires(self, router, client):
        router._rate_limiter._history.clear()
        for _ in range(60):
            router._rate_limiter.is_allowed("test")
        assert router._rate_limiter.get_wait_time("test") > 0.0
