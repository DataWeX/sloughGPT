"""Tests for SessionKVManager — per-session KV cache management."""
from __future__ import annotations

from domains.inference.session_kv_manager import SessionKVManager


class TestSessionKVManager:
    def test_set_and_get(self):
        mgr = SessionKVManager(kv_max_sessions=10)
        mgr.set_session("s1", "state1")
        assert mgr.get_session("s1") == "state1"

    def test_get_missing(self):
        mgr = SessionKVManager()
        assert mgr.get_session("nonexistent") is None

    def test_remove_session(self):
        mgr = SessionKVManager()
        mgr.set_session("s1", "state1")
        assert mgr.remove_session("s1") is True
        assert mgr.get_session("s1") is None

    def test_remove_nonexistent(self):
        mgr = SessionKVManager()
        assert mgr.remove_session("nope") is False

    def test_clear_all(self):
        mgr = SessionKVManager()
        mgr.set_session("s1", "a")
        mgr.set_session("s2", "b")
        count = mgr.clear_all()
        assert count == 2
        assert mgr.get_session("s1") is None

    def test_lru_eviction(self):
        mgr = SessionKVManager(kv_max_sessions=3)
        mgr.set_session("s1", "a")
        mgr.set_session("s2", "b")
        mgr.set_session("s3", "c")
        # Adding s4 should evict s1 (LRU)
        mgr.set_session("s4", "d")
        assert mgr.get_session("s1") is None
        assert mgr.get_session("s2") is not None

    def test_get_updates_access_time(self):
        mgr = SessionKVManager(kv_max_sessions=3)
        mgr.set_session("s1", "a")
        mgr.set_session("s2", "b")
        mgr.set_session("s3", "c")
        # Access s1 to make it recently used
        mgr.get_session("s1")
        # Adding s4 should evict s2 (oldest unused)
        mgr.set_session("s4", "d")
        assert mgr.get_session("s1") is not None
        assert mgr.get_session("s2") is None

    def test_get_stats(self):
        mgr = SessionKVManager(kv_max_sessions=10, kv_ttl=600)
        mgr.set_session("s1", "state")
        stats = mgr.get_stats()
        assert stats["active_sessions"] == 1
        assert stats["max_sessions"] == 10
        assert stats["ttl_seconds"] == 600
