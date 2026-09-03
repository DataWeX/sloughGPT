"""Tests for RateLimiter — sliding window rate limiter."""
from __future__ import annotations

import time

from domains.infrastructure.rate_limiter import (
    RATE_LIMIT_HEADER_LIMIT,
    RATE_LIMIT_HEADER_REMAINING,
    RATE_LIMIT_HEADER_RESET,
    RateLimiter,
    get_rate_limiter,
    reset_rate_limiter,
)


class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        allowed, remaining = rl.check("client1")
        assert allowed is True
        assert remaining == 4

    def test_blocks_after_limit(self):
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.check("client1")
        allowed, remaining = rl.check("client1")
        assert allowed is False
        assert remaining == 0

    def test_separate_keys(self):
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.check("a")
        rl.check("a")
        allowed_a, _ = rl.check("a")
        allowed_b, _ = rl.check("b")
        assert allowed_a is False
        assert allowed_b is True

    def test_reset(self):
        rl = RateLimiter(max_requests=1, window_seconds=60)
        rl.check("client1")
        rl.reset("client1")
        allowed, _ = rl.check("client1")
        assert allowed is True

    def test_window_expiry(self):
        rl = RateLimiter(max_requests=2, window_seconds=0)
        rl.check("client1")
        time.sleep(0.01)
        allowed, remaining = rl.check("client1")
        assert allowed is True
        assert remaining == 1


class TestHeaderConstants:
    def test_constants_exist(self):
        assert RATE_LIMIT_HEADER_REMAINING == "X-RateLimit-Remaining"
        assert RATE_LIMIT_HEADER_LIMIT == "X-RateLimit-Limit"
        assert RATE_LIMIT_HEADER_RESET == "X-RateLimit-Reset"


class TestGetRateLimiter:
    def test_singleton(self):
        reset_rate_limiter()
        a = get_rate_limiter()
        b = get_rate_limiter()
        assert a is b

    def test_custom_params(self):
        reset_rate_limiter()
        rl = get_rate_limiter(max_requests=10, window_seconds=30)
        assert rl.max_requests == 10
        assert rl.window_seconds == 30
