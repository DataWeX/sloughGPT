"""Tests for domains.infrastructure.rate_limiter — RateLimiter."""

import threading
import time
from domains.infrastructure.rate_limiter import RateLimiter, get_rate_limiter, reset_rate_limiter, RATE_LIMIT_HEADER_REMAINING, RATE_LIMIT_HEADER_LIMIT, RATE_LIMIT_HEADER_RESET


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

    def test_concurrent_check_thread_safe(self):
        rl = RateLimiter(max_requests=10, window_seconds=60)
        results = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            allowed, remaining = rl.check("shared")
            results.append((allowed, remaining))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        allowed_count = sum(1 for a, _ in results if a)
        denied_count = sum(1 for a, _ in results if not a)
        assert allowed_count + denied_count == 10
        assert allowed_count == 10

    def test_reset_nonexistent_key_no_error(self):
        rl = RateLimiter()
        rl.reset("ghost")

    def test_separate_keys_independent_windows(self):
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.check("a")
        rl.check("a")
        allowed_a, _ = rl.check("a")
        assert allowed_a is False
        allowed_b, remaining_b = rl.check("b")
        assert allowed_b is True
        assert remaining_b == 1

    def test_default_params(self):
        rl = RateLimiter()
        assert rl.max_requests == 60
        assert rl.window_seconds == 60

    def test_custom_params(self):
        rl = RateLimiter(max_requests=100, window_seconds=30)
        assert rl.max_requests == 100
        assert rl.window_seconds == 30

    def test_remaining_decreases(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        _, r1 = rl.check("k")
        _, r2 = rl.check("k")
        _, r3 = rl.check("k")
        assert r1 > r2 > r3

    def test_remaining_always_zero_when_exhausted(self):
        rl = RateLimiter(max_requests=1, window_seconds=60)
        rl.check("k")
        _, remaining = rl.check("k")
        assert remaining == 0

    def test_many_keys(self):
        rl = RateLimiter(max_requests=1, window_seconds=60)
        for i in range(100):
            allowed, _ = rl.check(f"key_{i}")
            assert allowed is True

    def test_check_returns_tuple(self):
        rl = RateLimiter()
        result = rl.check("k")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_remaining_type(self):
        rl = RateLimiter()
        _, remaining = rl.check("k")
        assert isinstance(remaining, int)

    def test_allowed_type(self):
        rl = RateLimiter()
        allowed, _ = rl.check("k")
        assert isinstance(allowed, bool)

    def test_reset_allows_full_window_again(self):
        rl = RateLimiter(max_requests=3, window_seconds=60)
        rl.check("k")
        rl.check("k")
        rl.check("k")
        rl.reset("k")
        allowed, remaining = rl.check("k")
        assert allowed is True
        assert remaining == 2

    def test_concurrent_many_threads(self):
        rl = RateLimiter(max_requests=50, window_seconds=60)
        results = []
        barrier = threading.Barrier(50)

        def worker():
            barrier.wait()
            allowed, _ = rl.check("shared")
            results.append(allowed)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 50

    def test_window_seconds_boundary(self):
        rl = RateLimiter(max_requests=1, window_seconds=0.05)
        rl.check("k")
        time.sleep(0.1)
        allowed, _ = rl.check("k")
        assert allowed is True


# ── Singleton ─────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_rate_limiter_returns_singleton(self):
        import domains.infrastructure.rate_limiter as mod
        original = mod._limiter
        mod._limiter = None
        try:
            a = get_rate_limiter()
            b = get_rate_limiter()
            assert a is b
        finally:
            mod._limiter = original

    def test_get_rate_limiter_preserves_existing(self):
        import domains.infrastructure.rate_limiter as mod
        original = mod._limiter
        custom = RateLimiter(max_requests=99)
        mod._limiter = custom
        try:
            result = get_rate_limiter()
            assert result is custom
        finally:
            mod._limiter = original

    def test_reset_creates_new_instance(self):
        original = get_rate_limiter()
        reset_rate_limiter()
        new = get_rate_limiter()
        assert new is not original
        assert isinstance(new, RateLimiter)

    def test_singleton_default_params(self):
        import domains.infrastructure.rate_limiter as mod
        original = mod._limiter
        mod._limiter = None
        try:
            rl = get_rate_limiter()
            assert rl.max_requests == 60
            assert rl.window_seconds == 60
        finally:
            mod._limiter = original

    def test_singleton_custom_params(self):
        import domains.infrastructure.rate_limiter as mod
        original = mod._limiter
        mod._limiter = None
        try:
            rl = get_rate_limiter(max_requests=100, window_seconds=30)
            assert rl.max_requests == 100
            assert rl.window_seconds == 30
        finally:
            mod._limiter = original

    def test_reset_then_singleton_new(self):
        import domains.infrastructure.rate_limiter as mod
        original = mod._limiter
        mod._limiter = None
        try:
            a = get_rate_limiter()
            reset_rate_limiter()
            b = get_rate_limiter()
            assert a is not b
            assert isinstance(b, RateLimiter)
        finally:
            mod._limiter = original

    def test_reset_clears_existing(self):
        import domains.infrastructure.rate_limiter as mod
        original = mod._limiter
        custom = RateLimiter(max_requests=1)
        mod._limiter = custom
        try:
            reset_rate_limiter()
            assert mod._limiter is None
        finally:
            mod._limiter = original


# ── Header constants ──────────────────────────────────────────────────

class TestHeaderConstants:
    def test_header_remaining(self):
        assert RATE_LIMIT_HEADER_REMAINING == "X-RateLimit-Remaining"

    def test_header_limit(self):
        assert RATE_LIMIT_HEADER_LIMIT == "X-RateLimit-Limit"

    def test_header_reset(self):
        assert RATE_LIMIT_HEADER_RESET == "X-RateLimit-Reset"
