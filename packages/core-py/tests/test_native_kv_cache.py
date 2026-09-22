"""Contract tests for the native KV stack: C wrapper + numpy fallback.

In this environment ``load_lib()`` cannot load a library (the C payload is
a macOS dylib living in the mirror tree), so the C-backed path is exercised
only when a loadable library is configured via ``MAN_TRANSFORMER_LIB``.
The numpy fallback is tested directly and as the active backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from domain.infrastructure._internal.kv_cache import NativeKVState
from domain.infrastructure._internal.kv_cache.native import _NumpyState


def _kv(seq_len: int, n_kv_heads: int = 4, head_dim: int = 8) -> np.ndarray:
    return np.random.randn(1, seq_len, n_kv_heads, head_dim).astype(np.float32)


@pytest.fixture()
def native_cache() -> NativeKVState:
    return NativeKVState(n_layers=2, n_kv_heads=4, head_dim=8, seq_capacity=32)


# ── numpy backend (internal to native) ──────────────────────────────────────────────────────────────


class TestNumpyBackend:
    def test_init_empty(self):
        cache = _NumpyState(2)
        assert cache.seq_len == 0
        assert cache.get(0) is None

    def test_update_returns_full_tensor(self):
        cache = _NumpyState(1)
        k, v = cache.update(0, _kv(3), _kv(3))
        assert k.shape == (1, 3, 4, 8)
        assert v.shape == (1, 3, 4, 8)

    def test_update_concatenates_along_seq(self):
        cache = _NumpyState(1)
        cache.update(0, _kv(2), _kv(2))
        k, v = cache.update(0, _kv(4), _kv(4))
        assert k.shape == (1, 6, 4, 8)
        assert cache.seq_len == 6
        assert np.allclose(k[:, :2], cache.get(0)[0][:, :2])

    def test_layers_independent(self):
        cache = _NumpyState(2)
        cache.update(0, _kv(3), _kv(3))
        assert cache.get(0) is not None
        assert cache.get(1) is None

    def test_get_returns_none_before_first_update(self):
        cache = _NumpyState(1)
        assert cache.get(0) is None

    def test_reset_clears(self):
        cache = _NumpyState(1)
        cache.update(0, _kv(3), _kv(3))
        cache.reset()
        assert cache.seq_len == 0
        assert cache.get(0) is None


# ── NativeKVState ─────────────────────────────────────────────────────────────


class TestNativeKVState:
    def test_falls_back_to_numpy_when_lib_unavailable(self, native_cache):
        assert native_cache.is_c is False
        assert native_cache.n_layers == 2

    def test_numpy_backend_full_cycle(self, native_cache):
        k, v = native_cache.update(0, _kv(5), _kv(5))
        assert k.shape == (1, 5, 4, 8)
        native_cache.update(1, _kv(2), _kv(2))
        k1, _ = native_cache.update(0, _kv(3), _kv(3))
        assert k1.shape == (1, 8, 4, 8)
        assert native_cache.seq_len == 8
        assert native_cache.get(1)[0].shape == (1, 2, 4, 8)
        assert native_cache.get(0) is not None

    def test_reset(self, native_cache):
        native_cache.update(0, _kv(4), _kv(4))
        native_cache.reset()
        assert native_cache.seq_len == 0
        assert native_cache.get(0) is None

    def test_free_and_close_are_noops_on_fallback(self, native_cache):
        native_cache.update(0, _kv(2), _kv(2))
        native_cache.free()
        assert native_cache.is_c is False

    def test_close_alias(self, native_cache):
        assert NativeKVState.close is NativeKVState.free

    def test_update_deviates_to_numpy_on_odd_shape(self, native_cache):
        batch2 = np.random.randn(2, 3, 4, 8).astype(np.float32)
        k, v = native_cache.update(0, batch2, batch2)
        assert k.shape == (2, 3, 4, 8)

    def test_fallback_ignores_seq_capacity_cap(self):
        cache = NativeKVState(n_layers=1, n_kv_heads=2, head_dim=4, seq_capacity=8)
        for _ in range(10):
            cache.update(0, _kv(1, 2, 4), _kv(1, 2, 4))
        assert cache.seq_len == 10
