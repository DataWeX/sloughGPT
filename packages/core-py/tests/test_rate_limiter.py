"""
Tests for the rate limiter middleware.
"""

import time
import pytest
from domains.infrastructure.rate_limiter import RateLimiter, get_rate_limiter


class TestRateLimiter:
    """Tests for the sliding-window rate limiter."""

    def test_allows_within_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            allowed, remaining = limiter.check("test")
            assert allowed is True
        assert remaining == 0

    def test_blocks_when_exceeded(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.check("test")
        allowed, remaining = limiter.check("test")
        assert allowed is False
        assert remaining == 0

    def test_different_keys_independent(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.check("alice")
        limiter.check("alice")
        limiter.check("bob")
        allowed_alice, _ = limiter.check("alice")
        assert allowed_alice is False
        allowed_bob, _ = limiter.check("bob")
        assert allowed_bob is True

    def test_window_expires(self):
        limiter = RateLimiter(max_requests=2, window_seconds=0.05)
        limiter.check("test")
        limiter.check("test")
        allowed, _ = limiter.check("test")
        assert allowed is False
        time.sleep(0.06)
        allowed, _ = limiter.check("test")
        assert allowed is True

    def test_reset_clears_key(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.check("test")
        limiter.check("test")
        limiter.reset("test")
        allowed, remaining = limiter.check("test")
        assert allowed is True
        assert remaining == 1

    def test_remaining_decrements(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for i in range(4, 0, -1):
            _, remaining = limiter.check("test")
            assert remaining == i

    def test_singleton(self):
        limiter1 = get_rate_limiter(10, 10)
        limiter2 = get_rate_limiter(20, 20)
        assert limiter1 is limiter2

    def test_zero_max_requests_blocks_all(self):
        limiter = RateLimiter(max_requests=0, window_seconds=60)
        allowed, remaining = limiter.check("test")
        assert allowed is False
        assert remaining == 0

    def test_single_request_limit(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        allowed, remaining = limiter.check("test")
        assert allowed is True
        assert remaining == 0
        allowed, _ = limiter.check("test")
        assert allowed is False

    def test_reset_nonexistent_key(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        limiter.reset("nonexistent")
        allowed, remaining = limiter.check("nonexistent")
        assert allowed is True
        assert remaining == 4

    def test_many_keys(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        for i in range(100):
            allowed, _ = limiter.check(f"key_{i}")
            assert allowed is True
        for i in range(100):
            allowed, _ = limiter.check(f"key_{i}")
            assert allowed is False

    def test_remaining_never_negative(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(10):
            _, remaining = limiter.check("test")
            assert remaining >= 0

    def test_concurrent_same_key(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        results = [limiter.check("test") for _ in range(7)]
        allowed_count = sum(1 for allowed, _ in results if allowed)
        assert allowed_count == 5
