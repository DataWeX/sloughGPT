"""Tests for domains.infrastructure.rate_limiter — RateLimiter."""

import time
from domains.infrastructure.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_first_request_allowed(self):
        rl = RateLimiter(max_requests=3, window_seconds=10)
        allowed, remaining = rl.check("ip1")
        assert allowed is True
        assert remaining == 2

    def test_exhaust_window(self):
        rl = RateLimiter(max_requests=2, window_seconds=10)
        rl.check("ip1")
        rl.check("ip1")
        allowed, remaining = rl.check("ip1")
        assert allowed is False
        assert remaining == 0

    def test_reset(self):
        rl = RateLimiter(max_requests=1, window_seconds=10)
        rl.check("ip1")
        rl.reset("ip1")
        allowed, _ = rl.check("ip1")
        assert allowed is True

    def test_separate_keys(self):
        rl = RateLimiter(max_requests=1, window_seconds=10)
        rl.check("ip1")
        allowed, _ = rl.check("ip2")
        assert allowed is True

    def test_window_expires(self):
        rl = RateLimiter(max_requests=1, window_seconds=0.1)
        rl.check("ip1")
        allowed, _ = rl.check("ip1")
        assert allowed is False
        time.sleep(0.15)
        allowed, _ = rl.check("ip1")
        assert allowed is True
