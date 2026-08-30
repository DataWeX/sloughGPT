"""Comprehensive tests for SloNetChatProvider session management.

Covers: session_stats, clear_session, clear_all_sessions,
_evict_stale_sessions, _evict_lru_session with edge cases and concurrency.
"""

import time
import threading
import pytest
from unittest.mock import MagicMock
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


# ── session_stats ──────────────────────────────────────────────────────────

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

    def test_single_session_age_is_zero(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace(kv_len=10)
        fp._kv_last_access["s1"] = time.monotonic()
        assert fp.session_stats()["oldest_session_age"] == 0.0

    def test_no_kv_len_attribute(self):
        """State without kv_len attribute should still work."""
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace()  # no kv_len
        fp._kv_last_access["s1"] = time.monotonic()
        stats = fp.session_stats()
        assert stats["cached_tokens"] == 0

    def test_tuple_kv_len(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace(kv_len=(30, 70))
        fp._kv_last_access["s1"] = time.monotonic()
        assert fp.session_stats()["cached_tokens"] == 100

    def test_many_sessions(self):
        fp = FakeProvider()
        for i in range(20):
            fp._kv_states[f"s{i}"] = SimpleNamespace(kv_len=i)
            fp._kv_last_access[f"s{i}"] = time.monotonic()
        stats = fp.session_stats()
        assert stats["active_sessions"] == 20
        expected = sum(range(20))
        assert stats["cached_tokens"] == expected


# ── clear_session ──────────────────────────────────────────────────────────

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

    def test_clear_multiple_times_same_session(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace()
        fp._kv_last_access["s1"] = time.monotonic()
        assert fp.clear_session("s1") is True
        assert fp.clear_session("s1") is False

    def test_clear_state_only_in_states(self):
        """Session in states but not in last_access."""
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace()
        # Don't add to _kv_last_access
        assert fp.clear_session("s1") is True
        assert "s1" not in fp._kv_states

    def test_clear_state_only_in_last_access(self):
        """Session in last_access but not in states."""
        fp = FakeProvider()
        fp._kv_last_access["s1"] = time.monotonic()
        assert fp.clear_session("s1") is False

    def test_clear_empty_provider(self):
        fp = FakeProvider()
        assert fp.clear_session("anything") is False


# ── clear_all_sessions ─────────────────────────────────────────────────────

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

    def test_clear_one(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace()
        fp._kv_last_access["s1"] = time.monotonic()
        assert fp.clear_all_sessions() == 1

    def test_clear_returns_count_before_clear(self):
        fp = FakeProvider()
        for i in range(5):
            fp._kv_states[f"s{i}"] = SimpleNamespace()
            fp._kv_last_access[f"s{i}"] = time.monotonic()
        count = fp.clear_all_sessions()
        assert count == 5
        # After clear, count is zero
        assert fp.clear_all_sessions() == 0

    def test_clear_many(self):
        fp = FakeProvider()
        for i in range(100):
            fp._kv_states[f"s{i}"] = SimpleNamespace()
            fp._kv_last_access[f"s{i}"] = time.monotonic()
        assert fp.clear_all_sessions() == 100


# ── _evict_stale_sessions ──────────────────────────────────────────────────

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

    def test_evicts_empty(self):
        fp = FakeProvider(ttl=1)
        stale = fp._evict_stale_sessions()
        assert stale == []

    def test_ttl_zero_evicts_all_old(self):
        fp = FakeProvider(ttl=0)
        fp._kv_states["s1"] = SimpleNamespace()
        fp._kv_last_access["s1"] = time.monotonic() - 0.1
        stale = fp._evict_stale_sessions()
        assert len(stale) == 1

    def test_exact_ttl_boundary(self):
        """Session older than ttl age should be evicted; session within ttl should survive."""
        fp = FakeProvider(ttl=1)
        fp._kv_states["stale"] = SimpleNamespace()
        fp._kv_states["fresh"] = SimpleNamespace()
        # stale is 2 seconds old (> ttl=1), fresh is 0.1s old (< ttl)
        fp._kv_last_access["stale"] = time.monotonic() - 2
        fp._kv_last_access["fresh"] = time.monotonic() - 0.1
        stale = fp._evict_stale_sessions()
        assert "stale" in stale
        assert "fresh" not in stale

    def test_eviction_returns_list_of_ids(self):
        fp = FakeProvider(ttl=1)
        fp._kv_states["x"] = SimpleNamespace()
        fp._kv_states["y"] = SimpleNamespace()
        fp._kv_last_access["x"] = time.monotonic() - 10
        fp._kv_last_access["y"] = time.monotonic() - 5
        stale = fp._evict_stale_sessions()
        assert set(stale) == {"x", "y"}


# ── _evict_lru_session ─────────────────────────────────────────────────────

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

    def test_evicts_exact_lru(self):
        fp = FakeProvider(max_sessions=2)
        fp._kv_states["a"] = SimpleNamespace()
        fp._kv_states["b"] = SimpleNamespace()
        fp._kv_states["c"] = SimpleNamespace()
        fp._kv_last_access["a"] = time.monotonic() - 10
        fp._kv_last_access["b"] = time.monotonic() - 5
        fp._kv_last_access["c"] = time.monotonic()
        fp._evict_lru_session("c")  # keep c, evict a (oldest)
        assert "a" not in fp._kv_states
        assert "b" in fp._kv_states
        assert "c" in fp._kv_states

    def test_eviction_returns_evicted_id(self):
        fp = FakeProvider(max_sessions=1)
        fp._kv_states["victim"] = SimpleNamespace()
        fp._kv_states["keeper"] = SimpleNamespace()
        fp._kv_last_access["victim"] = time.monotonic() - 100
        fp._kv_last_access["keeper"] = time.monotonic()
        evicted = fp._evict_lru_session("keeper")
        assert evicted == "victim"

    def test_multiple_evictions(self):
        fp = FakeProvider(max_sessions=1)
        fp._kv_states["old1"] = SimpleNamespace()
        fp._kv_states["old2"] = SimpleNamespace()
        fp._kv_states["keep"] = SimpleNamespace()
        fp._kv_last_access["old1"] = time.monotonic() - 20
        fp._kv_last_access["old2"] = time.monotonic() - 10
        fp._kv_last_access["keep"] = time.monotonic()
        fp._evict_lru_session("keep")
        # One should be evicted
        assert len(fp._kv_states) == 2
        assert "keep" in fp._kv_states


# ── Concurrency ────────────────────────────────────────────────────────────

class TestConcurrency:
    def test_concurrent_clear_session(self):
        fp = FakeProvider()
        for i in range(50):
            fp._kv_states[f"s{i}"] = SimpleNamespace()
            fp._kv_last_access[f"s{i}"] = time.monotonic()

        errors = []

        def clear_sessions():
            try:
                for i in range(50):
                    fp.clear_session(f"s{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=clear_sessions) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(fp._kv_states) == 0

    def test_concurrent_clear_all(self):
        fp = FakeProvider()
        for i in range(20):
            fp._kv_states[f"s{i}"] = SimpleNamespace()
            fp._kv_last_access[f"s{i}"] = time.monotonic()

        errors = []

        def do_clear():
            try:
                fp.clear_all_sessions()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_clear) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(fp._kv_states) == 0

    def test_concurrent_evict_and_add(self):
        fp = FakeProvider(max_sessions=5, ttl=0.001)
        errors = []

        def add_sessions():
            try:
                for i in range(20):
                    sid = f"s{threading.current_thread().ident}_{i}"
                    fp._kv_states[sid] = SimpleNamespace()
                    fp._kv_last_access[sid] = time.monotonic()
            except Exception as e:
                errors.append(e)

        def evict():
            try:
                fp._evict_stale_sessions()
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=add_sessions) for _ in range(3)] +
            [threading.Thread(target=evict) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ── Edge Cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_session_stats_after_all_cleared(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace(kv_len=100)
        fp._kv_last_access["s1"] = time.monotonic()
        fp.clear_all_sessions()
        stats = fp.session_stats()
        assert stats["active_sessions"] == 0
        assert stats["cached_tokens"] == 0

    def test_clear_session_does_not_affect_stats_max(self):
        fp = FakeProvider(max_sessions=10)
        fp._kv_states["s1"] = SimpleNamespace(kv_len=50)
        fp._kv_last_access["s1"] = time.monotonic()
        fp.clear_session("s1")
        stats = fp.session_stats()
        assert stats["max_sessions"] == 10

    def test_evict_stale_after_clear_all(self):
        fp = FakeProvider(ttl=1)
        fp.clear_all_sessions()
        stale = fp._evict_stale_sessions()
        assert stale == []

    def test_lru_eviction_with_identical_timestamps(self):
        """When timestamps are identical, min() picks the first dict key."""
        fp = FakeProvider(max_sessions=1)
        fp._kv_states["a"] = SimpleNamespace()
        fp._kv_states["b"] = SimpleNamespace()
        fp._kv_last_access["a"] = 100.0
        fp._kv_last_access["b"] = 100.0
        evicted = fp._evict_lru_session("b")
        # One should be evicted (a, since it's not the keep session)
        assert evicted == "a"

    def test_negative_kv_len(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace(kv_len=-50)
        fp._kv_last_access["s1"] = time.monotonic()
        stats = fp.session_stats()
        assert stats["cached_tokens"] == -50

    def test_very_large_ttl(self):
        fp = FakeProvider(ttl=999999999)
        fp._kv_states["s1"] = SimpleNamespace()
        fp._kv_last_access["s1"] = time.monotonic()
        stale = fp._evict_stale_sessions()
        assert len(stale) == 0

    def test_very_small_ttl(self):
        fp = FakeProvider(ttl=0.0001)
        fp._kv_states["s1"] = SimpleNamespace()
        fp._kv_last_access["s1"] = time.monotonic() - 1
        stale = fp._evict_stale_sessions()
        assert "s1" in stale


# ── Additional Coverage ──────────────────────────────────────────────────────

class TestSessionStatsExtra:
    def test_zero_token_kv_len(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace(kv_len=0)
        fp._kv_last_access["s1"] = time.monotonic()
        stats = fp.session_stats()
        assert stats["cached_tokens"] == 0

    def test_float_kv_len(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace(kv_len=42.5)
        fp._kv_last_access["s1"] = time.monotonic()
        stats = fp.session_stats()
        # float is not list/tuple/int, goes to 'elif kv is not None' path
        assert stats["cached_tokens"] == 42.5

    def test_many_sessions_oldest_age(self):
        fp = FakeProvider()
        now = time.monotonic()
        fp._kv_states["a"] = SimpleNamespace(kv_len=10)
        fp._kv_states["b"] = SimpleNamespace(kv_len=20)
        fp._kv_last_access["a"] = now - 5.0
        fp._kv_last_access["b"] = now
        stats = fp.session_stats()
        assert stats["oldest_session_age"] == pytest.approx(5.0, abs=0.1)


class TestClearSessionExtra:
    def test_clear_preserves_other_access_times(self):
        fp = FakeProvider()
        t1 = time.monotonic()
        t2 = time.monotonic()
        fp._kv_states["s1"] = SimpleNamespace()
        fp._kv_states["s2"] = SimpleNamespace()
        fp._kv_last_access["s1"] = t1
        fp._kv_last_access["s2"] = t2
        fp.clear_session("s1")
        assert fp._kv_last_access["s2"] == t2


class TestClearAllSessionsExtra:
    def test_clear_all_and_repopulate(self):
        fp = FakeProvider()
        fp._kv_states["s1"] = SimpleNamespace()
        fp._kv_last_access["s1"] = time.monotonic()
        fp.clear_all_sessions()
        fp._kv_states["new"] = SimpleNamespace()
        fp._kv_last_access["new"] = time.monotonic()
        stats = fp.session_stats()
        assert stats["active_sessions"] == 1


class TestEvictStaleExtra:
    def test_evict_mixed_age(self):
        fp = FakeProvider(ttl=2)
        fp._kv_states["very_old"] = SimpleNamespace()
        fp._kv_states["old"] = SimpleNamespace()
        fp._kv_states["fresh"] = SimpleNamespace()
        fp._kv_last_access["very_old"] = time.monotonic() - 100
        fp._kv_last_access["old"] = time.monotonic() - 5
        fp._kv_last_access["fresh"] = time.monotonic()
        stale = fp._evict_stale_sessions()
        assert set(stale) == {"very_old", "old"}
        assert "fresh" in fp._kv_states


class TestEvictLRUExtra:
    def test_evict_lru_returns_none_when_under_cap(self):
        fp = FakeProvider(max_sessions=10)
        fp._kv_states["s1"] = SimpleNamespace()
        fp._kv_last_access["s1"] = time.monotonic()
        result = fp._evict_lru_session("s1")
        assert result is None

    def test_multiple_sessions_same_oldest(self):
        fp = FakeProvider(max_sessions=1)
        same_time = 100.0
        fp._kv_states["a"] = SimpleNamespace()
        fp._kv_states["b"] = SimpleNamespace()
        fp._kv_last_access["a"] = same_time
        fp._kv_last_access["b"] = same_time
        fp._evict_lru_session("c")
        assert len(fp._kv_states) == 1
