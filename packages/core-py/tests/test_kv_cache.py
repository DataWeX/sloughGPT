"""Direct contract tests for the unified KVCache (session + paged).

Session caching funnels through KVCache now (SessionKVCache facade,
SessionKVManager, model_server), and paged block access is the forward
path. Concat is deliberately absent here — that lives in NativeKVCache.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from domain.infrastructure._internal.kv_cache.kv_cache import KVCache


def _kv(seq_len: int, n_heads: int = 8, head_dim: int = 16, batch: int = 1) -> np.ndarray:
    return np.random.randn(batch, seq_len, n_heads, head_dim).astype(np.float32)


# ── Paged mode ───────────────────────────────────────────────────────────────


class TestPagedMode:
    def test_paged_enabled_with_heads(self):
        cache = KVCache(2, n_heads=8, head_dim=16, block_size=4)
        assert cache.is_paged is True
        assert cache.block_size == 4

    def test_update_requires_paged(self):
        cache = KVCache(1)
        assert cache.is_paged is False
        with pytest.raises(ValueError):
            cache.update(0, _kv(2, n_heads=0), _kv(2, n_heads=0))

    def test_get_blocks_after_single_update(self):
        cache = KVCache(1, n_heads=8, head_dim=16, block_size=4)
        cache.update(0, _kv(5), _kv(5))
        k_blocks, v_blocks, n_valid = cache.get_blocks(0)
        assert n_valid == 5
        assert [b.shape[1] for b in k_blocks] == [4, 1]
        assert [b.shape[1] for b in v_blocks] == [4, 1]

    def test_blocks_split_on_block_size_boundaries(self):
        cache = KVCache(1, n_heads=8, head_dim=16, block_size=4)
        cache.update(0, _kv(9), _kv(9))
        k_blocks, _, n_valid = cache.get_blocks(0)
        assert n_valid == 9
        assert [b.shape[1] for b in k_blocks] == [4, 4, 1]

    def test_non_paged_returns_empty_blocks(self):
        cache = KVCache(1)
        k_blocks, v_blocks, n_valid = cache.get_blocks(0)
        assert k_blocks == []
        assert v_blocks == []
        assert n_valid == 0

    def test_seq_len_tracks_paged_len(self):
        cache = KVCache(1, n_heads=8, head_dim=16, block_size=4)
        assert cache.seq_len == 0
        cache.update(0, _kv(3), _kv(3))
        cache.update(0, _kv(2), _kv(2))
        assert cache.seq_len == 5

    def test_paged_reset_clears_blocks(self):
        cache = KVCache(1, n_heads=8, head_dim=16, block_size=4)
        cache.update(0, _kv(5), _kv(5))
        cache.reset()
        k_blocks, _, n_valid = cache.get_blocks(0)
        assert k_blocks == []
        assert n_valid == 0


# ── Session mode ─────────────────────────────────────────────────────────────


class TestSessionMode:
    def test_session_get_miss(self):
        cache = KVCache(1)
        assert cache.session_get("s1", [1, 2]) == (None, 0)

    def test_session_store_and_prefix_match(self):
        cache = KVCache(1)
        cache.session_store("s1", [1, 2, 3], "pkv")
        pkv, prefix = cache.session_get("s1", [1, 2, 3, 4])
        assert pkv == "pkv"
        assert prefix == 3

    def test_session_prefix_mismatch_returns_miss(self):
        cache = KVCache(1)
        cache.session_store("s1", [1, 2, 3], "pkv")
        pkv, prefix = cache.session_get("s1", [9, 8, 7])
        assert pkv is None
        assert prefix == 0

    def test_session_store_overwrite(self):
        cache = KVCache(1)
        cache.session_store("s1", [1, 2, 3], "old")
        cache.session_store("s1", [1, 2, 3, 4], "new")
        pkv, prefix = cache.session_get("s1", [1, 2, 3, 4, 5])
        assert pkv == "new"
        assert prefix == 4

    def test_session_clear(self):
        cache = KVCache(1)
        cache.session_store("s1", [1], "pkv")
        cache.session_clear("s1")
        assert cache.session_get("s1", [1]) == (None, 0)

    def test_session_clear_all(self):
        cache = KVCache(1)
        cache.session_store("s1", [1], "a")
        cache.session_store("s2", [2], "b")
        cache.session_clear_all()
        assert cache.session_get("s1", [1]) == (None, 0)
        assert cache.session_get("s2", [2]) == (None, 0)

    def test_session_lru_evicts_oldest(self):
        cache = KVCache(1, max_sessions=2, ttl=600.0)
        cache.session_store("s1", [1], "a")
        cache.session_store("s2", [2], "b")
        cache.session_store("s3", [3], "c")
        assert cache.session_get("s1", [1]) == (None, 0)
        assert cache.session_get("s2", [2]) == ("b", 1)

    def test_session_ttl_expiry(self):
        cache = KVCache(1, ttl=0.01)
        cache.session_store("s1", [1], "a")
        time.sleep(0.05)
        cache._session_evict_expired()
        assert cache.session_get("s1", [1]) == (None, 0)

    def test_session_ttl_preserves_fresh(self):
        cache = KVCache(1, ttl=0.01)
        cache.session_store("s1", [1], "a")
        time.sleep(0.05)
        cache.session_store("s2", [2], "b")
        cache._session_evict_expired()
        assert cache.session_get("s1", [1]) == (None, 0)
        assert cache.session_get("s2", [2]) == ("b", 1)

    def test_max_sessions_zero_raises(self):
        cache = KVCache(1, max_sessions=0)
        with pytest.raises(ValueError):
            cache.session_store("s1", [1], "a")
