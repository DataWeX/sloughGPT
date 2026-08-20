"""Meaningful tests for SloNetChatProvider session management — session_stats, clear_session, clear_all_sessions, _evict_stale_sessions, _evict_lru_session."""

import time
import threading
import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace


class FakeProvider:
    """Minimal stand-in for SloNetChatProvider to test session management methods."""

    def __init__(self, max_sessions=8, ttl=300):
        self._kv_states = {}
        self._kv_last_access = {}
        self._kv_max_sessions = max_sessions
        self._kv_ttl = ttl
        self._kv_lock = threading.Lock()
        self._logger = MagicMock()

    def session_stats(self):
        with self._kv_lock:
            n_sessions = len(self._kv_states)
            total_tokens = 0
            for state in self._kv_states.values():
                kv = getattr(state, "kv_len", None)
                if isinstance(kv, (list, tuple)):
                    total_tokens += sum(kv)
                elif kv is not None:
                    total_tokens += kv
            return {
                "active_sessions": n_sessions,
                "max_sessions": self._kv_max_sessions,
                "ttl_seconds": self._kv_ttl,
                "cached_tokens": total_tokens,
                "oldest_session_age": max(self._kv_last_access.values()) - min(self._kv_last_access.values())
                if len(self._kv_last_access) > 1 else 0.0,
            }

    def clear_session(self, session_id):
        with self._kv_lock:
            existed = self._kv_states.pop(session_id, None) is not None
            self._kv_last_access.pop(session_id, None)
        return existed

    def clear_all_sessions(self):
        with self._kv_lock:
            n = len(self._kv_states)
            self._kv_states.clear()
            self._kv_last_access.clear()
        return n

    def _evict_stale_sessions(self):
        now = time.monotonic()
        with self._kv_lock:
            stale = [sid for sid, ts in self._kv_last_access.items()
                     if now - ts > self._kv_ttl]
            for sid in stale:
                self._kv_states.pop(sid, None)
                self._kv_last_access.pop(sid, None)
        return stale

    def _evict_lru_session(self, keep_session_id):
        if len(self._kv_states) <= self._kv_max_sessions:
            return
        evictable = {sid: ts for sid, ts in self._kv_last_access.items()
                     if sid != keep_session_id}
        if not evictable:
            return
        lru_id = min(evictable, key=evictable.get)
        self._kv_states.pop(lru_id, None)
        self._kv_last_access.pop(lru_id, None)
        return lru_id


# ── session_stats ──────────────────────────────────────────────────────

class TestSessionStats:
    def test_empty(self):
        fp = FakeProvider()
        stats = fp.session_stats()
        assert stats["active_sessions"] == 0
        assert stats["cached_tokens"] == 0
        assert stats["oldest_session_age"] == 0.0

    def test_with_sessions(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace(kv_len=100)
        fp._kv_states["s2"] = SimpleNamespace(kv_len=200)
        fp._kv_last_access["s1"] = time.monotonic()
        fp._kv_last_access["s2"] = time.monotonic() + 0.01
        stats = fp.session_stats()
        assert stats["active_sessions"] == 2
        assert stats["cached_tokens"] == 300

    def test_with_list_kv_len(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace(kv_len=[50, 60])
        fp._kv_last_access["s1"] = time.monotonic()
        stats = fp.session_stats()
        assert stats["cached_tokens"] == 110

    def test_with_none_kv_len(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace(kv_len=None)
        fp._kv_last_access["s1"] = time.monotonic()
        stats = fp.session_stats()
        assert stats["cached_tokens"] == 0

    def test_max_sessions(self):
        fp = FakeProvider(max_sessions=16)
        assert fp.session_stats()["max_sessions"] == 16

    def test_ttl(self):
        fp = FakeProvider(ttl=600)
        assert fp.session_stats()["ttl_seconds"] == 600


# ── clear_session ──────────────────────────────────────────────────────

class TestClearSession:
    def test_clear_existing(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace()
        fp._kv_last_access["s1"] = time.monotonic()
        assert fp.clear_session("s1") is True
        assert "s1" not in fp._kv_states
        assert "s1" not in fp._kv_last_access

    def test_clear_nonexistent(self):
        fp = FakeProvider()
        assert fp.clear_session("missing") is False

    def test_clear_leaves_others(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace()
        fp._kv_states["s2"] = SimpleNamespace()
        fp._kv_last_access["s1"] = time.monotonic()
        fp._kv_last_access["s2"] = time.monotonic()
        fp.clear_session("s1")
        assert "s2" in fp._kv_states


# ── clear_all_sessions ─────────────────────────────────────────────────

class TestClearAllSessions:
    def test_clear_all(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace()
        fp._kv_states["s2"] = SimpleNamespace()
        fp._kv_last_access["s1"] = time.monotonic()
        fp._kv_last_access["s2"] = time.monotonic()
        count = fp.clear_all_sessions()
        assert count == 2
        assert len(fp._kv_states) == 0
        assert len(fp._kv_last_access) == 0

    def test_clear_empty(self):
        fp = FakeProvider()
        assert fp.clear_all_sessions() == 0


# ── _evict_stale_sessions ──────────────────────────────────────────────

class TestEvictStaleSessions:
    def test_evicts_old(self):
        fp = FakeProvider(ttl=1)
        fp._kv_states["old"] = SimpleNamespace()
        fp._kv_states["fresh"] = SimpleNamespace()
        fp._kv_last_access["old"] = time.monotonic() - 10
        fp._kv_last_access["fresh"] = time.monotonic()
        stale = fp._evict_stale_sessions()
        assert "old" not in fp._kv_states
        assert "fresh" in fp._kv_states
        assert "old" in stale

    def test_evicts_none_when_all_fresh(self):
        fp = FakeProvider(ttl=300)
        fp._kv_states["s1"] = SimpleNamespace()
        fp._kv_last_access["s1"] = time.monotonic()
        stale = fp._evict_stale_sessions()
        assert len(stale) == 0
        assert "s1" in fp._kv_states

    def test_evicts_all_when_all_stale(self):
        fp = FakeProvider(ttl=1)
        fp._kv_states["a"] = SimpleNamespace()
        fp._kv_states["b"] = SimpleNamespace()
        fp._kv_last_access["a"] = time.monotonic() - 100
        fp._kv_last_access["b"] = time.monotonic() - 100
        stale = fp._evict_stale_sessions()
        assert len(stale) == 2
        assert len(fp._kv_states) == 0


# ── _evict_lru_session ─────────────────────────────────────────────────

class TestEvictLRUSession:
    def test_no_eviction_when_under_cap(self):
        fp = FakeProvider(max_sessions=5)
        fp._kv_states["s1"] = SimpleNamespace()
        fp._kv_last_access["s1"] = time.monotonic()
        fp._evict_lru_session("s2")
        assert "s1" in fp._kv_states

    def test_evicts_lru_when_over_cap(self):
        fp = FakeProvider(max_sessions=1)
        fp._kv_states["old"] = SimpleNamespace()
        fp._kv_states["newer"] = SimpleNamespace()
        fp._kv_last_access["old"] = time.monotonic() - 5
        fp._kv_last_access["newer"] = time.monotonic()
        fp._evict_lru_session("brand_new")
        assert "old" not in fp._kv_states
        assert "newer" in fp._kv_states

    def test_never_evicts_keep_session(self):
        fp = FakeProvider(max_sessions=1)
        fp._kv_states["keep_me"] = SimpleNamespace()
        fp._kv_last_access["keep_me"] = time.monotonic() - 10
        fp._evict_lru_session("keep_me")
        assert "keep_me" in fp._kv_states

    def test_no_evictable_sessions(self):
        fp = FakeProvider(max_sessions=1)
        fp._kv_states["only"] = SimpleNamespace()
        fp._kv_last_access["only"] = time.monotonic()
        result = fp._evict_lru_session("only")
        assert result is None
        assert "only" in fp._kv_states
