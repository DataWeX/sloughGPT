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
        # Last one should have 0 remaining
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
        limiter.check("bob")  # bob should still be allowed
        allowed_alice, _ = limiter.check("alice")
        assert allowed_alice is False
        allowed_bob, _ = limiter.check("bob")
        assert allowed_bob is True

    def test_window_expires(self):
        limiter = RateLimiter(max_requests=2, window_seconds=0.05)
        limiter.check("test")
        limiter.check("test")
        # Should be blocked
        allowed, _ = limiter.check("test")
        assert allowed is False
        # Wait for window to expire
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
