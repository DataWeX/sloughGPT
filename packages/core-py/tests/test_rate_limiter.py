"""Tests for domains.infrastructure.rate_limiter — RateLimiter."""

import threading
import time
from domains.infrastructure.rate_limiter import RateLimiter, get_rate_limiter


class TestSingleton:
    def test_get_rate_limiter_returns_singleton(self):
        """get_rate_limiter() returns the same instance on repeated calls."""
        # Reset global state
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
        """get_rate_limiter() doesn't overwrite an existing limiter."""
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
        from domains.infrastructure.rate_limiter import reset_rate_limiter
        original = get_rate_limiter()
        reset_rate_limiter()
        new = get_rate_limiter()
        assert new is not original
        assert isinstance(new, RateLimiter)


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
        """Multiple threads checking the same key — no races, exact count."""
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
        # With max_requests=10, exactly 10 threads hit simultaneously —
        # the lock serialises them so all 10 should be allowed.
        assert allowed_count == 10

    def test_reset_nonexistent_key_no_error(self):
        """Resetting a key that was never used doesn't raise."""
        rl = RateLimiter()
        rl.reset("ghost")  # should not raise

    def test_separate_keys_independent_windows(self):
        """Different keys maintain independent rate windows."""
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.check("a")
        rl.check("a")
        # 'a' exhausted
        allowed_a, _ = rl.check("a")
        assert allowed_a is False
        # 'b' still has budget
        allowed_b, remaining_b = rl.check("b")
        assert allowed_b is True
        assert remaining_b == 1
