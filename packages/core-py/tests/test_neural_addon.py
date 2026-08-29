"""Tests for domains.shell.addons.neural — NeuralOp, NeuralState, NeuralProcessType, NeuralMemoryType, CacheStrategy, NeuralKVCache, KVCacheEntry."""

import numpy as np
from domains.shell.addons.neural import (
    NeuralOp, NeuralState, NeuralProcessType, NeuralMemoryType,
    CacheStrategy, NeuralKVCache, KVCacheEntry,
)


class TestNeuralOp:
    def test_all_members(self):
        assert len(NeuralOp) == 10
    def test_values(self):
        assert NeuralOp.NONE.value == 0
        assert NeuralOp.EMBEDDING.value == 1
        assert NeuralOp.ATTENTION.value == 2


class TestNeuralState:
    def test_all_members(self):
        assert len(NeuralState) == 8
    def test_values(self):
        assert NeuralState.IDLE.value == 0
        assert NeuralState.COMPUTING.value == 2


class TestNeuralProcessType:
    def test_all_members(self):
        assert len(NeuralProcessType) == 4
    def test_values(self):
        assert NeuralProcessType.INFERENCE.value == 0
        assert NeuralProcessType.TRAINING.value == 1


class TestNeuralMemoryType:
    def test_all_members(self):
        assert len(NeuralMemoryType) >= 7
    def test_values(self):
        assert NeuralMemoryType.KV_CACHE.value == 0
        assert NeuralMemoryType.EMBEDDING.value == 1


class TestCacheStrategy:
    def test_all_members(self):
        assert len(CacheStrategy) == 4
    def test_values(self):
        assert CacheStrategy.LRU.value == 0
        assert CacheStrategy.FIFO.value == 2


class TestKVCacheEntry:
    def test_fields(self):
        entry = KVCacheEntry(layer_idx=0, seq_len=10, last_access=1.0)
        assert entry.layer_idx == 0
        assert entry.seq_len == 10


class TestNeuralKVCache:
    def test_init(self):
        cache = NeuralKVCache(num_layers=4, head_dim=32, max_positions=64)
        assert cache.get_position() == 0
        assert cache.total_tokens_cached == 0

    def test_initialize(self):
        cache = NeuralKVCache(num_layers=4, head_dim=32, max_positions=64)
        cache.initialize(num_heads=2)
        assert cache.memory_bytes() > 0

    def test_update_and_get(self):
        cache = NeuralKVCache(num_layers=4, head_dim=32, max_positions=64)
        cache.initialize(num_heads=2)
        k = np.random.randn(2, 32).astype(np.float32)
        v = np.random.randn(2, 32).astype(np.float32)
        cache.update(0, k, v)
        cache.advance()
        k2, v2 = cache.get(0)
        assert k2 is not None
        assert k2.ndim == 3
        assert k2.shape[0] == 2
        assert k2.shape[2] == 32

    def test_update_out_of_range(self):
        cache = NeuralKVCache(num_layers=2, head_dim=32, max_positions=64)
        cache.initialize(num_heads=2)
        k = np.random.randn(2, 32).astype(np.float32)
        v = np.random.randn(2, 32).astype(np.float32)
        try:
            cache.update(10, k, v)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_memory_bytes(self):
        cache = NeuralKVCache(num_layers=4, head_dim=32, max_positions=64)
        cache.initialize(num_heads=2)
        assert cache.memory_bytes() > 0

    def test_evictions(self):
        cache = NeuralKVCache(num_layers=4, head_dim=32, max_positions=64)
        cache.initialize(num_heads=2)
        assert cache.evictions == 0
