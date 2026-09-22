"""Direct contract tests for the unified KVCacheBase (canonical KV cache).

Everything session-related funnels through KVCacheBase now (SessionKVCache
facade + model_server + benchmark), so the canonical class needs its own
lock on concat, paged and session modes.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from domain.infrastructure._internal.kv_cache.base import KVCacheBase


def _kv(seq_len: int, n_heads: int = 8, head_dim: int = 16, batch: int = 1) -> np.ndarray:
    return np.random.randn(batch, seq_len, n_heads, head_dim).astype(np.float32)


# ── Concat mode ──────────────────────────────────────────────────────────────


class TestConcatMode:
    def test_init(self):
        cache = KVCacheBase(4)
        assert cache.n_layers == 4
        assert cache.seq_len == 0
        assert cache.is_paged is False
        assert cache.block_size == 64

    def test_get_empty_is_none(self):
        cache = KVCacheBase(2)
        assert cache.get(0) is None

    def test_update_copies_first_tensor(self):
        cache = KVCacheBase(1)
        k, v = _kv(3), _kv(3)
        k_out, v_out = cache.update(0, k, v)
        assert k_out is k
        assert v_out is v
        assert cache.seq_len == 3

    def test_update_concatenates_along_seq(self):
        cache = KVCacheBase(1)
        cache.update(0, _kv(3), _kv(3))
        k2, v2 = _kv(2), _kv(2)
        k_out, v_out = cache.update(0, k2, v2)
        assert k_out.shape[1] == 5
        assert v_out.shape[1] == 5
        np.testing.assert_array_equal(k_out[:, 3:5], k2)

    def test_get_returns_concatenated(self):
        cache = KVCacheBase(2)
        cache.update(0, _kv(2), _kv(2))
        k, v = cache.get(0)
        assert k.shape[1] == 2
        assert cache.get(1) is None

    def test_reset_clears(self):
        cache = KVCacheBase(1)
        cache.update(0, _kv(4), _kv(4))
        assert cache.seq_len == 4
        cache.reset()
        assert cache.seq_len == 0
        assert cache.get(0) is None

    def test_layers_independent(self):
        cache = KVCacheBase(3)
        cache.update(0, _kv(5), _kv(5))
        assert cache.seq_len == 5
        k1, _ = cache.update(1, _kv(2), _kv(2))
        assert k1.shape[1] == 2
        # seq_len reflects layer 0 only
        assert cache.seq_len == 5
        assert cache.get(2) is None


# ── Paged mode ───────────────────────────────────────────────────────────────


class TestPagedMode:
    def test_paged_enabled_with_heads(self):
        cache = KVCacheBase(2, n_heads=8, head_dim=16, block_size=4)
        assert cache.is_paged is True
        assert cache.block_size == 4

    def test_get_blocks_after_single_update(self):
        cache = KVCacheBase(1, n_heads=8, head_dim=16, block_size=4)
        cache.update(0, _kv(5), _kv(5))
        k_blocks, v_blocks, n_valid = cache.get_blocks(0)
        assert n_valid == 5
        assert [b.shape[1] for b in k_blocks] == [4, 1]
        assert [b.shape[1] for b in v_blocks] == [4, 1]

    def test_blocks_split_on_block_size_boundaries(self):
        cache = KVCacheBase(1, n_heads=8, head_dim=16, block_size=4)
        cache.update(0, _kv(9), _kv(9))
        k_blocks, _, n_valid = cache.get_blocks(0)
        assert n_valid == 9
        assert [b.shape[1] for b in k_blocks] == [4, 4, 1]

    def test_get_blocks_non_paged_falls_back_to_concat(self):
        cache = KVCacheBase(1)
        cache.update(0, _kv(2), _kv(2))
        k_blocks, v_blocks, n_valid = cache.get_blocks(0)
        assert n_valid == 2
        assert len(k_blocks) == 1
        assert k_blocks[0].shape[1] == 2

    def test_paged_reset_clears_blocks(self):
        cache = KVCacheBase(1, n_heads=8, head_dim=16, block_size=4)
        cache.update(0, _kv(5), _kv(5))
        cache.reset()
        k_blocks, _, n_valid = cache.get_blocks(0)
        assert k_blocks == []
        assert n_valid == 0


# ── Session mode ─────────────────────────────────────────────────────────────


class TestSessionMode:
    def test_session_get_miss(self):
        cache = KVCacheBase(1)
        assert cache.session_get("s1", [1, 2]) == (None, 0)

    def test_session_store_and_prefix_match(self):
        cache = KVCacheBase(1)
        cache.session_store("s1", [1, 2, 3], "pkv")
        pkv, prefix = cache.session_get("s1", [1, 2, 3, 4])
        assert pkv == "pkv"
        assert prefix == 3

    def test_session_prefix_mismatch_returns_miss(self):
        cache = KVCacheBase(1)
        cache.session_store("s1", [1, 2, 3], "pkv")
        pkv, prefix = cache.session_get("s1", [9, 8, 7])
        assert pkv is None
        assert prefix == 0

    def test_session_store_overwrite(self):
        cache = KVCacheBase(1)
        cache.session_store("s1", [1, 2, 3], "old")
        cache.session_store("s1", [1, 2, 3, 4], "new")
        pkv, prefix = cache.session_get("s1", [1, 2, 3, 4, 5])
        assert pkv == "new"
        assert prefix == 4

    def test_session_clear(self):
        cache = KVCacheBase(1)
        cache.session_store("s1", [1], "pkv")
        cache.session_clear("s1")
        assert cache.session_get("s1", [1]) == (None, 0)

    def test_session_clear_all(self):
        cache = KVCacheBase(1)
        cache.session_store("s1", [1], "a")
        cache.session_store("s2", [2], "b")
        cache.session_clear_all()
        assert cache.session_get("s1", [1]) == (None, 0)
        assert cache.session_get("s2", [2]) == (None, 0)

    def test_session_lru_evicts_oldest(self):
        cache = KVCacheBase(1, max_sessions=2, ttl=600.0)
        cache.session_store("s1", [1], "a")
        cache.session_store("s2", [2], "b")
        cache.session_store("s3", [3], "c")
        assert cache.session_get("s1", [1]) == (None, 0)
        assert cache.session_get("s2", [2]) == ("b", 1)

    def test_session_ttl_expiry(self):
        cache = KVCacheBase(1, ttl=0.01)
        cache.session_store("s1", [1], "a")
        time.sleep(0.05)
        cache._session_evict_expired()
        assert cache.session_get("s1", [1]) == (None, 0)

    def test_session_ttl_preserves_fresh(self):
        cache = KVCacheBase(1, ttl=0.01)
        cache.session_store("s1", [1], "a")
        time.sleep(0.05)
        cache.session_store("s2", [2], "b")
        cache._session_evict_expired()
        assert cache.session_get("s1", [1]) == (None, 0)
        assert cache.session_get("s2", [2]) == ("b", 1)

    def test_max_sessions_zero_raises(self):
        cache = KVCacheBase(1, max_sessions=0)
        with pytest.raises(ValueError):
            cache.session_store("s1", [1], "a")
