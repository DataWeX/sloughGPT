"""Tests for SessionKVManager."""

import threading
import time

from domain.inference._internal.session_kv_manager import SessionKVManager


class TestSessionKVManager:
    def test_get_set_remove(self):
        mgr = SessionKVManager(kv_max_sessions=10)
        assert mgr.get_session("s1") is None
        mgr.set_session("s1", "state1")
        assert mgr.get_session("s1") == "state1"
        assert mgr.remove_session("s1") is True
        assert mgr.get_session("s1") is None
        assert mgr.remove_session("s1") is False

    def test_clear_all(self):
        mgr = SessionKVManager()
        mgr.set_session("a", "1")
        mgr.set_session("b", "2")
        assert mgr.clear_all() == 2
        assert mgr.get_session("a") is None

    def test_lru_eviction(self):
        mgr = SessionKVManager(kv_max_sessions=3)
        mgr.set_session("s1", "v1")
        mgr.set_session("s2", "v2")
        mgr.set_session("s3", "v3")
        mgr.set_session("s4", "v4")
        # s1 should be evicted (LRU)
        assert mgr.get_session("s1") is None
        assert mgr.get_session("s2") == "v2"

    def test_lru_eviction_excludes_current(self):
        mgr = SessionKVManager(kv_max_sessions=2)
        mgr.set_session("s1", "v1")
        mgr.set_session("s2", "v2")
        mgr.set_session("s2", "v2_updated")
        assert mgr.get_session("s1") == "v1"
        assert mgr.get_session("s2") == "v2_updated"

    def test_stats(self):
        mgr = SessionKVManager(kv_max_sessions=10, kv_ttl=600)
        mgr.set_session("a", "1")
        stats = mgr.get_stats()
        assert stats["active_sessions"] == 1
        assert stats["max_sessions"] == 10
        assert stats["ttl_seconds"] == 600

    def test_thread_safety(self):
        mgr = SessionKVManager(kv_max_sessions=100)
        barrier = threading.Barrier(10)

        def worker(i):
            barrier.wait()
            for j in range(50):
                mgr.set_session(f"s{i}_{j}", f"v{i}_{j}")
                mgr.get_session(f"s{i}_{j}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stats = mgr.get_stats()
        assert stats["active_sessions"] == 100

    def test_evict_stale_sessions(self):
        mgr = SessionKVManager(kv_ttl=0.01)
        mgr.set_session("s1", "v1")
        time.sleep(0.02)
        evicted = mgr.evict_stale_sessions()
        assert evicted == 1
        assert mgr.get_session("s1") is None

    def test_get_stats_with_complex_state(self):
        class MockKVState:
            def __init__(self):
                import numpy as np

                self.k = np.zeros((1, 4, 8))
                self.v = np.zeros((1, 4, 8))

        mgr = SessionKVManager()
        mgr.set_session("s1", MockKVState())
        stats = mgr.get_stats()
        assert "k_shape" in stats["session_sizes"]["s1"]

    def test_get_or_create_uses_factory_on_miss(self):
        calls = []

        def factory():
            calls.append(1)
            return "kv-state"

        mgr = SessionKVManager(state_factory=factory)
        state = mgr.get_or_create("s1")
        assert state == "kv-state"
        assert calls == [1]
        assert mgr.get_or_create("s1") == "kv-state"
        assert calls == [1]  # hit reuses the cached state, factory not called again

    def test_get_or_create_no_factory_returns_none(self):
        mgr = SessionKVManager()
        assert mgr.get_or_create("s1") is None
        assert mgr.get_session("s1") is None

    def test_get_or_create_evicts_lru_when_over_cap(self):
        mgr = SessionKVManager(kv_max_sessions=2, state_factory=lambda: "v")
        mgr.set_session("s1", "v1")
        mgr.get_or_create("s2")
        mgr.get_or_create("s3")  # s1 is now oldest -> evicted
        assert mgr.get_session("s1") is None
        assert mgr.get_session("s2") == "v"
        assert mgr.get_session("s3") == "v"

    def test_cached_token_count(self):
        class State:
            def __init__(self, kv_len):
                self.kv_len = kv_len

        mgr = SessionKVManager()
        mgr.set_session("a", State([5, 3]))
        mgr.set_session("b", State(2))
        mgr.set_session("c", State(None))  # no kv_len is counted as 0
        assert mgr.cached_token_count() == 10

    def test_session_summary_shape(self):
        mgr = SessionKVManager(kv_max_sessions=8, kv_ttl=300, state_factory=lambda: "v")
        assert mgr.session_summary() == {
            "active_sessions": 0,
            "max_sessions": 8,
            "ttl_seconds": 300,
            "cached_tokens": 0,
            "oldest_session_age": 0.0,
        }
        mgr.set_session("a", "v1")
        summary = mgr.session_summary()
        assert summary["active_sessions"] == 1
        assert summary["cached_tokens"] == 0
        assert set(summary) == {
            "active_sessions",
            "max_sessions",
            "ttl_seconds",
            "cached_tokens",
            "oldest_session_age",
        }
