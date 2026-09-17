"""Native C KV cache — ctypes wrapper for libtransformer_forward."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np


class NativeKVCache:
    """KV cache backed by the C transformer library.

    Delegates to transformer_kv_cache_init/reset/free via ctypes.
    Falls back to NumpyKVCache if the C library is unavailable.
    """

    __slots__ = ("_lib", "_cache", "_n_layers", "_n_kv_heads", "_head_dim", "_fallback")

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, seq_capacity: int = 2048):
        self._n_layers = n_layers
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._fallback: Any = None
        self._lib = None
        self._cache = None

        try:
            from domain.inference._internal.native.bindings import get_lib

            lib = get_lib()
            self._lib = lib
            self._cache = lib._KVCache()
            from ctypes import c_int

            cfg = lib._Config()
            cfg.n_layers = c_int(n_layers)
            cfg.hidden_dim = c_int(n_kv_heads * head_dim)
            cfg.n_heads = c_int(n_kv_heads)
            cfg.n_kv_heads = c_int(n_kv_heads)
            cfg.head_dim = c_int(head_dim)
            lib.transformer_kv_cache_init(self._cache, ctypes.byref(cfg), c_int(seq_capacity))
        except Exception:
            from domain.infrastructure._internal.kv_cache.numpy import NumpyKVCache

            self._fallback = NumpyKVCache(n_layers)

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def seq_len(self) -> int:
        if self._fallback is not None:
            return self._fallback.seq_len
        if self._cache is None:
            return 0
        return self._cache.seq_len

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._fallback is not None:
            return self._fallback.update(layer_idx, k, v)
        # Native path: copy into C buffer (placeholder — full impl needs buffer management)
        return self._fallback.update(layer_idx, k, v) if self._fallback else (k, v)

    def get(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        if self._fallback is not None:
            return self._fallback.get(layer_idx)
        return None

    def reset(self) -> None:
        if self._fallback is not None:
            self._fallback.reset()
            return
        if self._lib and self._cache:
            self._lib.transformer_kv_cache_reset(self._cache)
