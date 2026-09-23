"""Tests for domain.infrastructure._internal.kv_cache.base — KVCacheBase.

Covers the three modes: concat (preallocated buffers, no per-token concat),
paged (block refs, no alloc), and session (cross-turn prefix LRU).
"""

import numpy as np

from domain.infrastructure._internal.kv_cache.base import KVCacheBase

N_LAYERS = 4
N_HEADS = 2
HEAD_DIM = 8


def _kv(seq_len):
    k = np.random.randn(1, seq_len, N_HEADS, HEAD_DIM).astype(np.float32)
    v = np.random.randn(1, seq_len, N_HEADS, HEAD_DIM).astype(np.float32)
    return k, v


class TestConcatMode:
    def test_init_empty(self):
        cache = KVCacheBase(N_LAYERS)
        assert cache.seq_len == 0
        for layer in range(N_LAYERS):
            assert cache.get(layer) is None

    def test_single_update_stores_tensor(self):
        cache = KVCacheBase(N_LAYERS)
        k, v = _kv(5)
        k_out, v_out = cache.update(0, k, v)
        assert k_out.shape[1] == 5
        assert v_out.shape[1] == 5
        assert cache.seq_len == 5
        np.testing.assert_array_equal(k_out, k)
        np.testing.assert_array_equal(v_out, v)

    def test_incremental_updates_accumulate(self):
        cache = KVCacheBase(N_LAYERS)
        k1, v1 = _kv(3)
        k2, v2 = _kv(2)
        cache.update(0, k1, v1)
        _, v_out = cache.update(0, k2, v2)
        assert cache.seq_len == 5
        k_out, _ = cache.get(0)
        # first chunk preserved, second appended
        assert k_out.shape[1] == 5
        np.testing.assert_array_equal(k_out[0, 3:5], k2[0])
        np.testing.assert_array_equal(v_out[0, :3], v1[0])

    def test_growth_beyond_capacity(self):
        cache = KVCacheBase(1)
        n_total = 0
        for n in range(0, 200, 7):
            k, v = _kv(7)
            cache.update(0, k, v)
            n_total += 7
        assert cache.seq_len == n_total
        k_out, _ = cache.get(0)
        assert k_out.shape[1] == n_total

    def test_per_layer_independence(self):
        cache = KVCacheBase(N_LAYERS)
        k0, v0 = _kv(4)
        k1, v1 = _kv(2)
        cache.update(0, k0, v0)
        cache.update(2, k1, v1)
        assert cache.get(1) is None
        assert cache.get(2)[0].shape[1] == 2
        assert cache.get(0)[0].shape[1] == 4

    def test_reset(self):
        cache = KVCacheBase(N_LAYERS)
        for layer in range(N_LAYERS):
            k, v = _kv(2)
            cache.update(layer, k, v)
        cache.reset()
        assert cache.seq_len == 0
        for layer in range(N_LAYERS):
            assert cache.get(layer) is None

    def test_updated_returns_fill_length_view(self):
        cache = KVCacheBase(1)
        k, v = _kv(8)
        cache.update(0, k, v)
        k2, v2 = _kv(4)
        k_out, _ = cache.update(0, k2, v2)
        assert k_out.shape[1] == 12
        k_get, _ = cache.get(0)
        np.testing.assert_array_equal(k_out, k_get)


class TestPagedMode:
    def test_paged_blocks_accumulate_without_concat(self):
        cache = KVCacheBase(1, n_heads=N_HEADS, head_dim=HEAD_DIM, block_size=4)
        for _ in range(3):
            k, v = _kv(4)
            cache.update(0, k, v)
        k_blocks, v_blocks, n_valid = cache.get_blocks(0)
        assert n_valid == 12
        assert len(k_blocks) == 3
        assert isinstance(k_blocks, list)

    def test_paged_non_paged_fallback(self):
        cache = KVCacheBase(1)
        k, v = _kv(3)
        cache.update(0, k, v)
        k_blocks, v_blocks, n_valid = cache.get_blocks(0)
        assert n_valid == 3
        assert len(k_blocks) == 1
        assert k_blocks[0].shape[1] == 3


class TestSessionMode:
    def test_session_store_get_prefix(self):
        cache = KVCacheBase(0, max_sessions=5, ttl=600.0)
        pkv = (np.zeros((1, 3, 1, 1)),)
        cache.session_store("s1", [1, 2, 3], pkv)
        cached, prefix_len = cache.session_get("s1", [1, 2, 3, 4])
        assert cached is pkv
        assert prefix_len == 3

    def test_session_partial_prefix(self):
        cache = KVCacheBase(0, max_sessions=5, ttl=600.0)
        cache.session_store("s1", [1, 2, 3], ("pkv",))
        cached, prefix_len = cache.session_get("s1", [1, 2, 9])
        assert prefix_len == 2

    def test_session_no_match(self):
        cache = KVCacheBase(0, max_sessions=5, ttl=600.0)
        cache.session_store("s1", [1, 2, 3], ("pkv",))
        cached, prefix_len = cache.session_get("s1", [9, 9, 9])
        assert cached is None
        assert prefix_len == 0

    def test_session_clear(self):
        cache = KVCacheBase(0, max_sessions=5, ttl=600.0)
        cache.session_store("s1", [1], ("pkv",))
        cache.session_clear("s1")
        assert cache.session_get("s1", [1])[0] is None
        cache.session_store("s2", [1], ("pkv",))
        cache.session_clear_all()
        assert cache.session_get("s2", [1])[0] is None

    def test_session_lru_eviction(self):
        cache = KVCacheBase(0, max_sessions=2, ttl=600.0)
        cache.session_store("a", [1], ("pkv",))
        cache.session_store("b", [1], ("pkv",))
        cache.session_store("c", [1], ("pkv",))
        assert cache.session_get("a", [1])[0] is None
        assert cache.session_get("b", [1])[0] is not None
        assert cache.session_get("c", [1])[0] is not None

    def test_session_ttl_eviction(self):
        cache = KVCacheBase(0, max_sessions=5, ttl=-1.0)
        cache.session_store("s1", [1], ("pkv",))
        # eviction is lazy: the next store drops the already-expired s1
        cache.session_store("s2", [1], ("pkv",))
        assert cache.session_get("s1", [1])[0] is None
