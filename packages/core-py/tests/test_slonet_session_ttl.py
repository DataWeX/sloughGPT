"""
Tests for cross-turn KV session TTL eviction in SloNetChatProvider.

Verifies that stale sessions are evicted after the TTL expires,
active sessions survive eviction, and the resolve helper creates
new states correctly.
"""
import threading
import time
import numpy as np
import pytest

from domains.training.slonet import (
    SloTransformer, SloTransformerBlock, SloLinear, SloEmbedding,
    SloLayerNorm, NumpyKVState,
)
from domains.inference.slonet_provider import SloNetChatProvider


@pytest.fixture
def tiny_model():
    E, V, BS = 64, 256, 128
    net = SloTransformer(
        vocab_size=V, n_embed=E, n_layer=1, n_head=4, block_size=BS,
        n_kv_head=2,
    )
    net.max_seq_len = BS
    return net


class TestSessionTTL:
    """Tests for _evict_stale_sessions and _resolve_session_kv."""

    def _make_provider_stub(self, model):
        """Create a minimal provider-like object with the TTL machinery.

        We don't need a full SloNetChatProvider (requires tokenizer, etc.),
        just the _kv_states / _kv_last_access / _evict_stale_sessions /
        _resolve_session_kv attributes and methods.
        """
        class _Stub:
            pass

        stub = _Stub()
        stub._model = model
        stub._get_model = lambda: stub._model
        stub._kv_states = {}
        stub._kv_last_access = {}
        stub._kv_ttl = 60.0  # 60s default
        stub._kv_max_sessions = 64
        stub._kv_lock = threading.Lock()

        from types import MethodType
        stub._evict_stale_sessions = MethodType(
            SloNetChatProvider._evict_stale_sessions, stub)
        stub._evict_lru_session = MethodType(
            SloNetChatProvider._evict_lru_session, stub)
        stub._resolve_session_kv = MethodType(
            SloNetChatProvider._resolve_session_kv, stub)
        return stub

    def test_resolve_creates_new_state(self, tiny_model):
        stub = self._make_provider_stub(tiny_model)

        state = stub._resolve_session_kv("session-abc")
        assert isinstance(state, NumpyKVState)
        assert "session-abc" in stub._kv_states

    def test_resolve_returns_existing_state(self, tiny_model):
        """_resolve_session_kv returns the same state for repeated calls."""
        stub = self._make_provider_stub(tiny_model)

        s1 = stub._resolve_session_kv("session-xyz")
        s2 = stub._resolve_session_kv("session-xyz")
        assert s1 is s2

    def test_resolve_none_session_returns_none(self, tiny_model):
        """_resolve_session_kv returns None when session_id is None."""
        stub = self._make_provider_stub(tiny_model)

        result = stub._resolve_session_kv(None)
        assert result is None
        assert len(stub._kv_states) == 0

    def test_resolve_updates_last_access(self, tiny_model):
        """_resolve_session_kv updates the monotonic timestamp on each access."""
        stub = self._make_provider_stub(tiny_model)

        stub._resolve_session_kv("s1")
        t1 = stub._kv_last_access["s1"]
        time.sleep(0.01)
        stub._resolve_session_kv("s1")
        t2 = stub._kv_last_access["s1"]
        assert t2 > t1

    def test_evict_removes_stale_sessions(self, tiny_model):
        """Sessions idle longer than TTL are evicted."""
        stub = self._make_provider_stub(tiny_model)
        stub._kv_ttl = 0.05  # 50ms TTL

        stub._resolve_session_kv("stale-1")
        stub._resolve_session_kv("stale-2")
        assert len(stub._kv_states) == 2

        time.sleep(0.1)  # > 50ms TTL
        stub._evict_stale_sessions()
        assert len(stub._kv_states) == 0
        assert "stale-1" not in stub._kv_last_access
        assert "stale-2" not in stub._kv_last_access

    def test_evict_keeps_fresh_sessions(self, tiny_model):
        """Sessions accessed within TTL are not evicted."""
        stub = self._make_provider_stub(tiny_model)
        stub._kv_ttl = 10.0  # 10s TTL — plenty of time

        stub._resolve_session_kv("fresh-1")
        stub._resolve_session_kv("fresh-2")
        stub._evict_stale_sessions()
        assert len(stub._kv_states) == 2

    def test_evict_mixed_stale_and_fresh(self, tiny_model):
        """Only stale sessions are evicted when mixed with fresh ones."""
        stub = self._make_provider_stub(tiny_model)
        stub._kv_ttl = 0.05

        stub._resolve_session_kv("old")
        time.sleep(0.06)  # make "old" stale
        stub._resolve_session_kv("new")  # "new" is fresh

        stub._evict_stale_sessions()
        assert "old" not in stub._kv_states
        assert "new" in stub._kv_states
        assert len(stub._kv_states) == 1

    def test_eviction_on_resolve(self, tiny_model):
        """_resolve_session_kv triggers eviction before resolving."""
        stub = self._make_provider_stub(tiny_model)
        stub._kv_ttl = 0.05

        stub._resolve_session_kv("stale")
        time.sleep(0.06)

        # This resolve call should evict "stale" first
        state = stub._resolve_session_kv("fresh")
        assert "stale" not in stub._kv_states
        assert "fresh" in stub._kv_states
        assert isinstance(state, NumpyKVState)

    def test_multiple_sessions_independent(self, tiny_model):
        """Each session gets its own independent KV state."""
        stub = self._make_provider_stub(tiny_model)

        s1 = stub._resolve_session_kv("a")
        s2 = stub._resolve_session_kv("b")
        s3 = stub._resolve_session_kv("c")
        assert s1 is not s2
        assert s2 is not s3
        assert len(stub._kv_states) == 3

    def test_resolve_after_eviction_creates_fresh(self, tiny_model):
        """After a session is evicted, resolving it creates a brand-new state."""
        stub = self._make_provider_stub(tiny_model)
        stub._kv_ttl = 0.05

        s1 = stub._resolve_session_kv("revive")
        # Generate something to dirty the state
        ids = np.array([[10, 20, 30]])
        list(stub._model.generate_numpy_stream(
            ids, max_new_tokens=3, temperature=0.0, kv_state=s1))
        assert s1.prev_ids is not None

        time.sleep(0.06)
        s2 = stub._resolve_session_kv("revive")
        assert s2 is not s1  # new object
        assert s2.prev_ids is None  # fresh state, clean

    def test_ttl_configurable(self, tiny_model):
        """TTL can be set to different values."""
        stub = self._make_provider_stub(tiny_model)

        stub._kv_ttl = 0.01
        stub._resolve_session_kv("fast")
        time.sleep(0.02)
        stub._evict_stale_sessions()
        assert len(stub._kv_states) == 0

        stub._kv_ttl = 9999
        stub._resolve_session_kv("slow")
        stub._evict_stale_sessions()
        assert len(stub._kv_states) == 1


class TestSessionStats:
    """Tests for session_stats() observability report."""

    def _make_provider_stub(self, model):
        class _Stub:
            pass

        stub = _Stub()
        stub._model = model
        stub._get_model = lambda: stub._model
        stub._kv_states = {}
        stub._kv_last_access = {}
        stub._kv_ttl = 3600.0
        stub._kv_max_sessions = 64
        stub._kv_lock = threading.Lock()

        from types import MethodType
        stub._evict_stale_sessions = MethodType(
            SloNetChatProvider._evict_stale_sessions, stub)
        stub._evict_lru_session = MethodType(
            SloNetChatProvider._evict_lru_session, stub)
        stub._resolve_session_kv = MethodType(
            SloNetChatProvider._resolve_session_kv, stub)
        stub.session_stats = MethodType(SloNetChatProvider.session_stats, stub)
        return stub

    def test_empty_stats(self, tiny_model):
        """Empty provider reports zero sessions and zero cached tokens."""
        stub = self._make_provider_stub(tiny_model)
        stats = stub.session_stats()
        assert stats["active_sessions"] == 0
        assert stats["cached_tokens"] == 0
        assert stats["ttl_seconds"] == 3600.0

    def test_stats_after_resolve(self, tiny_model):
        """Resolving sessions updates the active count."""
        stub = self._make_provider_stub(tiny_model)
        stub._resolve_session_kv("a")
        stub._resolve_session_kv("b")
        stats = stub.session_stats()
        assert stats["active_sessions"] == 2

    def test_stats_cached_tokens_grows(self, tiny_model):
        """Generation through a session increments cached tokens."""
        stub = self._make_provider_stub(tiny_model)
        state = stub._resolve_session_kv("s")
        ids = np.array([[10, 20, 30]])
        list(stub._model.generate_numpy_stream(
            ids, max_new_tokens=5, temperature=0.0, kv_state=state))
        stats = stub.session_stats()
        assert stats["cached_tokens"] > 0

    def test_stats_eviction_reflects(self, tiny_model):
        """Eviction reduces the reported session count."""
        stub = self._make_provider_stub(tiny_model)
        stub._kv_ttl = 0.05
        stub._resolve_session_kv("gone")
        time.sleep(0.06)
        stub._evict_stale_sessions()
        stats = stub.session_stats()
        assert stats["active_sessions"] == 0


class TestSessionClear:
    """Tests for clear_session and clear_all_sessions."""

    def _make_provider_stub(self, model):
        class _Stub:
            pass

        stub = _Stub()
        stub._model = model
        stub._get_model = lambda: stub._model
        stub._kv_states = {}
        stub._kv_last_access = {}
        stub._kv_ttl = 3600.0
        stub._kv_max_sessions = 64
        stub._kv_lock = threading.Lock()

        from types import MethodType
        stub._evict_stale_sessions = MethodType(
            SloNetChatProvider._evict_stale_sessions, stub)
        stub._evict_lru_session = MethodType(
            SloNetChatProvider._evict_lru_session, stub)
        stub._resolve_session_kv = MethodType(
            SloNetChatProvider._resolve_session_kv, stub)
        stub.clear_session = MethodType(SloNetChatProvider.clear_session, stub)
        stub.clear_all_sessions = MethodType(
            SloNetChatProvider.clear_all_sessions, stub)
        return stub

    def test_clear_removes_existing_session(self, tiny_model):
        stub = self._make_provider_stub(tiny_model)
        stub._resolve_session_kv("s1")
        assert stub.clear_session("s1") is True
        assert "s1" not in stub._kv_states
        assert "s1" not in stub._kv_last_access

    def test_clear_missing_session_returns_false(self, tiny_model):
        stub = self._make_provider_stub(tiny_model)
        assert stub.clear_session("never-existed") is False

    def test_clear_does_not_affect_other_sessions(self, tiny_model):
        stub = self._make_provider_stub(tiny_model)
        stub._resolve_session_kv("keep")
        stub._resolve_session_kv("drop")
        stub.clear_session("drop")
        assert "drop" not in stub._kv_states
        assert "keep" in stub._kv_states

    def test_clear_all_empties_store(self, tiny_model):
        stub = self._make_provider_stub(tiny_model)
        stub._resolve_session_kv("a")
        stub._resolve_session_kv("b")
        assert stub.clear_all_sessions() == 2
        assert len(stub._kv_states) == 0
        assert len(stub._kv_last_access) == 0

    def test_clear_all_empty_store_returns_zero(self, tiny_model):
        stub = self._make_provider_stub(tiny_model)
        assert stub.clear_all_sessions() == 0

    def test_clear_stats_reflect_removal(self, tiny_model):
        stub = self._make_provider_stub(tiny_model)
        from types import MethodType
        stub.session_stats = MethodType(SloNetChatProvider.session_stats, stub)
        stub._resolve_session_kv("a")
        stub._resolve_session_kv("b")
        stub.clear_session("a")
        assert stub.session_stats()["active_sessions"] == 1


class TestKvSessionCap:
    """LRU cap: the session KV map never exceeds max_sessions entries."""

    def _make_provider_stub(self, model, max_sessions=2):
        class _Stub:
            pass

        stub = _Stub()
        stub._model = model
        stub._get_model = lambda: stub._model
        stub._kv_states = {}
        stub._kv_last_access = {}
        stub._kv_ttl = 3600.0
        stub._kv_max_sessions = max_sessions
        stub._kv_lock = threading.Lock()

        from types import MethodType
        stub._evict_stale_sessions = MethodType(
            SloNetChatProvider._evict_stale_sessions, stub)
        stub._resolve_session_kv = MethodType(
            SloNetChatProvider._resolve_session_kv, stub)
        stub._evict_lru_session = MethodType(
            SloNetChatProvider._evict_lru_session, stub)
        stub.session_stats = MethodType(SloNetChatProvider.session_stats, stub)
        return stub

    def test_no_eviction_below_cap(self, tiny_model):
        stub = self._make_provider_stub(tiny_model, max_sessions=3)
        stub._resolve_session_kv("a")
        stub._resolve_session_kv("b")
        assert len(stub._kv_states) == 2
        assert "a" in stub._kv_states and "b" in stub._kv_states

    def test_exact_cap_keeps_all(self, tiny_model):
        stub = self._make_provider_stub(tiny_model, max_sessions=2)
        stub._resolve_session_kv("a")
        stub._resolve_session_kv("b")
        assert len(stub._kv_states) == 2
        assert "a" in stub._kv_states and "b" in stub._kv_states

    def test_over_cap_evicts_lru(self, tiny_model):
        """Adding a third session evicts the least-recently-used one."""
        stub = self._make_provider_stub(tiny_model, max_sessions=2)
        stub._resolve_session_kv("a")
        stub._resolve_session_kv("b")
        stub._resolve_session_kv("c")
        assert len(stub._kv_states) == 2
        assert "a" not in stub._kv_states
        assert "b" in stub._kv_states and "c" in stub._kv_states

    def test_recently_touched_survives_eviction(self, tiny_model):
        """Touching a session refreshes its recency, protecting it from LRU."""
        stub = self._make_provider_stub(tiny_model, max_sessions=2)
        stub._resolve_session_kv("a")
        stub._resolve_session_kv("b")
        stub._resolve_session_kv("a")  # touch a → a is now recent
        stub._resolve_session_kv("c")  # evicts b (LRU), keeps a
        assert "a" in stub._kv_states and "c" in stub._kv_states
        assert "b" not in stub._kv_states

    def test_stats_expose_max_sessions(self, tiny_model):
        stub = self._make_provider_stub(tiny_model, max_sessions=7)
        assert stub.session_stats()["max_sessions"] == 7

    def test_cap_bounds_active_sessions(self, tiny_model):
        """active_sessions never exceeds the cap across many resolutions."""
        stub = self._make_provider_stub(tiny_model, max_sessions=3)
        for i in range(10):
            stub._resolve_session_kv(f"s{i}")
        assert len(stub._kv_states) <= 3
        assert stub.session_stats()["active_sessions"] <= 3


class TestKvConcurrency:
    """Thread-safety of the session KV map under concurrent resolution."""

    def _make_provider_stub(self, model):
        class _Stub:
            pass

        stub = _Stub()
        stub._model = model
        stub._get_model = lambda: stub._model
        stub._kv_states = {}
        stub._kv_last_access = {}
        stub._kv_ttl = 3600.0
        stub._kv_max_sessions = 64
        stub._kv_lock = threading.Lock()

        from types import MethodType
        stub._evict_stale_sessions = MethodType(
            SloNetChatProvider._evict_stale_sessions, stub)
        stub._evict_lru_session = MethodType(
            SloNetChatProvider._evict_lru_session, stub)
        stub._resolve_session_kv = MethodType(
            SloNetChatProvider._resolve_session_kv, stub)
        stub.clear_session = MethodType(SloNetChatProvider.clear_session, stub)
        stub.session_stats = MethodType(SloNetChatProvider.session_stats, stub)
        return stub

    def test_concurrent_resolve_single_state(self, tiny_model):
        """Concurrent resolution of the same session yields one shared state."""
        stub = self._make_provider_stub(tiny_model)
        n_threads = 16
        barrier = threading.Barrier(n_threads)
        results = {}
        lock = threading.Lock()

        def resolve():
            barrier.wait()
            state = stub._resolve_session_kv("shared")
            with lock:
                results[threading.get_ident()] = id(state)

        threads = [threading.Thread(target=resolve) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        ids = set(results.values())
        assert len(ids) == 1
        assert len(stub._kv_states) == 1
        assert stub.session_stats()["active_sessions"] == 1

    def test_concurrent_resolve_distinct_sessions(self, tiny_model):
        """Concurrent resolution of distinct sessions never cross-contaminates."""
        stub = self._make_provider_stub(tiny_model)
        n_threads = 8
        barrier = threading.Barrier(n_threads)
        ids = set()

        def resolve(i):
            barrier.wait()
            state = stub._resolve_session_kv(f"sess-{i}")
            with stub._kv_lock:
                ids.add(id(state))

        threads = [threading.Thread(target=resolve, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == n_threads
        assert len(stub._kv_states) == n_threads
        assert stub.session_stats()["active_sessions"] == n_threads

    def test_concurrent_clear_during_resolve(self, tiny_model):
        """clear_session racing resolve never leaves a stale entry behind."""
        stub = self._make_provider_stub(tiny_model)
        stop = threading.Event()
        results = []
        lock = threading.Lock()

        def resolver():
            while not stop.is_set():
                state = stub._resolve_session_kv("raced")
                with lock:
                    results.append(id(state))

        def clearer():
            while not stop.is_set():
                stub.clear_session("raced")

        t_resolve = threading.Thread(target=resolver)
        t_clear = threading.Thread(target=clearer)
        t_resolve.start()
        t_clear.start()
        time.sleep(0.05)
        stop.set()
        t_resolve.join()
        t_clear.join()

        state = stub._resolve_session_kv("raced")
        assert state is not None
        assert len(stub._kv_states) == 1
        assert stub.session_stats()["active_sessions"] == 1
