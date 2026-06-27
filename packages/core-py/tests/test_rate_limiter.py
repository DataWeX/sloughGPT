"""
Tests for Rate Limiter (rate_limiter.py).
"""

import time
import pytest
from domains.infrastructure.rate_limiter import (
    TokenBucket, RateLimiter, AsyncRateLimiter,
    DEFAULT_LIMITS, get_rate_limiter, set_rate_limiter,
)


class TestTokenBucket:
    def test_init_full(self):
        b = TokenBucket(rate=1.0, burst=5)
        assert b.tokens == 5.0
        assert b.fill_pct == 100.0

    def test_try_consume_success(self):
        b = TokenBucket(rate=10.0, burst=5)
        assert b.try_consume(1.0) is True
        assert b.tokens == 4.0

    def test_try_consume_failure(self):
        b = TokenBucket(rate=1.0, burst=2)
        b.tokens = 0.5
        assert b.try_consume(1.0) is False

    def test_refill_over_time(self):
        b = TokenBucket(rate=2.0, burst=10)
        b.tokens = 0
        b.last_refill = time.monotonic() - 1.0  # 1 second ago
        b.refill()
        assert b.tokens == pytest.approx(2.0, rel=0.1)

    def test_refill_caps_at_burst(self):
        b = TokenBucket(rate=1.0, burst=5)
        b.tokens = 4.0
        b.last_refill = time.monotonic() - 10.0
        b.refill()
        assert b.tokens == 5.0

    def test_wait_time_zero_when_available(self):
        b = TokenBucket(rate=1.0, burst=5)
        assert b.wait_time(1.0) == 0.0

    def test_wait_time_positive(self):
        b = TokenBucket(rate=2.0, burst=5)
        b.tokens = 0
        assert b.wait_time(1.0) == pytest.approx(0.5, rel=0.1)

    def test_wait_time_infinite_for_zero_rate(self):
        b = TokenBucket(rate=0.0, burst=1)
        assert b.wait_time(1.0) == float("inf")

    def test_fill_pct(self):
        b = TokenBucket(rate=1.0, burst=10)
        b.tokens = 3.0
        assert b.fill_pct == 30.0


class TestRateLimiter:
    def test_add_limit(self):
        rl = RateLimiter()
        rl.add_limit("endpoint:test", rate=5.0, burst=10)
        assert rl.bucket_count == 1

    def test_check_no_limit_allowed(self):
        rl = RateLimiter()
        assert rl.check("nonexistent") is True

    def test_check_within_limit(self):
        rl = RateLimiter()
        rl.add_limit("endpoint:test", rate=10.0, burst=5)
        # First 5 should be allowed
        for _ in range(5):
            assert rl.check("endpoint:test") is True
        # 6th should be denied
        assert rl.check("endpoint:test") is False

    def test_check_multiple_keys(self):
        rl = RateLimiter()
        rl.add_limit("key:a", rate=10.0, burst=2)
        rl.add_limit("key:b", rate=10.0, burst=2)
        assert rl.check("key:a") is True
        assert rl.check("key:a") is True
        assert rl.check("key:a") is False
        assert rl.check("key:b") is True  # separate bucket

    def test_remove_limit(self):
        rl = RateLimiter()
        rl.add_limit("test", rate=1.0, burst=1)
        rl.remove_limit("test")
        assert rl.bucket_count == 0

    def test_clear(self):
        rl = RateLimiter()
        rl.add_limit("a", rate=1.0, burst=1)
        rl.add_limit("b", rate=1.0, burst=1)
        rl.clear()
        assert rl.bucket_count == 0

    def test_wait_seconds(self):
        rl = RateLimiter()
        rl.add_limit("test", rate=2.0, burst=2)
        rl.check("test")
        rl.check("test")
        # Bucket empty, 1 token needs 0.5s
        assert rl.wait_seconds("test", 1.0) > 0

    def test_wait_seconds_no_limit(self):
        rl = RateLimiter()
        assert rl.wait_seconds("nonexistent") == 0.0

    def test_stats(self):
        rl = RateLimiter()
        rl.add_limit("test", rate=5.0, burst=10)
        stats = rl.stats()
        assert len(stats) == 1
        assert stats[0]["key"] == "test"
        assert stats[0]["rate"] == 5.0
        assert stats[0]["burst"] == 10

    def test_bucket_refills_over_time(self):
        rl = RateLimiter()
        rl.add_limit("refill_test", rate=10.0, burst=3)
        assert rl.check("refill_test") is True
        assert rl.check("refill_test") is True
        assert rl.check("refill_test") is True
        assert rl.check("refill_test") is False

        # Wait for refill
        bucket = rl.get_bucket("refill_test")
        assert bucket is not None
        bucket.last_refill = time.monotonic() - 0.5  # 0.5s ago
        assert rl.check("refill_test") is True  # 5 tokens added in 0.5s at 10/s
        bucket.last_refill = time.monotonic() - 0.3  # another 0.3s gives 3 more
        assert rl.check("refill_test") is True
        assert rl.check("refill_test") is True
        assert rl.check("refill_test") is True


class TestDefaultLimits:
    def test_defaults_loaded(self):
        rl = get_rate_limiter()
        for key in DEFAULT_LIMITS:
            assert rl.get_bucket(key) is not None, f"Missing default limit: {key}"

    def test_set_singleton(self):
        rl = RateLimiter()
        set_rate_limiter(rl)
        assert get_rate_limiter() is rl


class TestAsyncRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_immediate(self):
        arl = AsyncRateLimiter()
        arl.add_limit("test", rate=10.0, burst=5)
        ok = await arl.acquire("test", timeout=1.0)
        assert ok is True

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        arl = AsyncRateLimiter()
        arl.add_limit("test", rate=0.0, burst=1)
        await arl.acquire("test")  # use the only token
        ok = await arl.acquire("test", timeout=0.1)
        assert ok is False

    @pytest.mark.asyncio
    async def test_acquire_no_limit(self):
        arl = AsyncRateLimiter()
        ok = await arl.acquire("nonexistent")
        assert ok is True

    @pytest.mark.asyncio
    async def test_acquire_wait_then_succeed(self):
        """Bucket refills during wait."""
        arl = AsyncRateLimiter()
        arl.add_limit("test", rate=10.0, burst=1)
        assert await arl.acquire("test") is True
        # Second call should wait (bucket empty) but succeed within timeout
        # since rate=10/s means 1 token in 0.1s
        ok = await arl.acquire("test", timeout=0.5)
        assert ok is True
