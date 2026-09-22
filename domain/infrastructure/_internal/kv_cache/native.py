"""Native KV state — C backend (ctypes) with an internal numpy fallback.

``NativeKVState`` is the concatenating per-layer KV state used for greedy
decoding. Two backends, one API:

- C path: mirrors the layout of ``libtransformer_forward`` — one flat float
  buffer per K/V, ``n_kv_heads * head_dim`` floats per token step, layers
  laid out contiguously. Driven through ``transformer_kv_cache_init`` /
  ``reset`` / ``free``.
- numpy path: classic per-layer ``np.concatenate`` (the "numpy" half of the
  native C + numpy stack).

When the C library is absent or unloadable (it ships as a macOS dylib next
to ``domain.inference._internal.native.bindings``), the numpy backend serves
the same API. This is *native*; the modular default lives in :mod:`.kv_state`.
"""

from __future__ import annotations

import ctypes

import numpy as np


class _NumpyBackend:
    """Internal numpy concatenating backend (batch-1) for NativeKVState."""

    __slots__ = ("_n_layers", "_k", "_v")

    def __init__(self, n_layers: int):
        self._n_layers = n_layers
        self._k: list[np.ndarray | None] = [None] * n_layers
        self._v: list[np.ndarray | None] = [None] * n_layers

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def seq_len(self) -> int:
        for kv in self._k:
            if kv is not None:
                return kv.shape[1]
        return 0

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Append new K, V for a layer; return the full concatenated (K, V)."""
        prev = self._k[layer_idx]
        if prev is None:
            self._k[layer_idx] = k
            self._v[layer_idx] = v
        else:
            self._k[layer_idx] = np.concatenate([prev, k], axis=1)
            self._v[layer_idx] = np.concatenate([self._v[layer_idx], v], axis=1)
        return self._k[layer_idx], self._v[layer_idx]

    def get(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        if self._k[layer_idx] is None:
            return None
        return self._k[layer_idx], self._v[layer_idx]

    def reset(self) -> None:
        self._k = [None] * self._n_layers
        self._v = [None] * self._n_layers


class NativeKVState:
    """Concatenating KV cache — C (ctypes) with numpy fallback.

    Parameters match the transformer forward path: ``n_kv_heads`` and
    ``head_dim`` define the per-token stride of the C buffers; the layout
    matches ``TransformerKVCache`` (``seq_capacity`` steps per layer).
    """

    __slots__ = (
        "_n_layers",
        "_n_kv_heads",
        "_head_dim",
        "_seq_capacity",
        "_lib",
        "_cache",
        "_lens",
        "_fallback",
    )

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, seq_capacity: int = 2048):
        self._n_layers = n_layers
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._seq_capacity = seq_capacity
        self._lens = [0] * n_layers
        self._lib = None
        self._cache = None
        self._fallback = _NumpyBackend(n_layers)

        try:
            from domain.inference._internal.native.bindings import load_lib

            lib = load_lib()
            cache = lib._KVCache()
            cfg = lib._Config()
            cfg.n_layers = n_layers
            cfg.hidden_dim = n_kv_heads * head_dim
            cfg.n_heads = n_kv_heads
            cfg.n_kv_heads = n_kv_heads
            cfg.head_dim = head_dim
            cfg.block_size = seq_capacity
            lib.transformer_kv_cache_init(ctypes.byref(cache), ctypes.byref(cfg), seq_capacity)
            if bool(cache.k) and bool(cache.v):
                self._lib = lib
                self._cache = cache
                self._cache.seq_len = 0
        except Exception:
            pass

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def is_c(self) -> bool:
        return self._cache is not None

    @property
    def seq_len(self) -> int:
        if self._cache is not None:
            return self._cache.seq_len
        return self._fallback.seq_len if self._fallback is not None else 0

    @property
    def stride(self) -> int:
        return self._n_kv_heads * self._head_dim

    def _layer_ptr(self, layer_idx: int, which: str) -> ctypes.POINTER(ctypes.c_float):
        ptr = self._cache.k if which == "k" else self._cache.v
        offset = layer_idx * self.stride * self._seq_capacity
        return ctypes.cast(ptr + offset, ctypes.POINTER(ctypes.c_float))

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Append new K, V for a layer; return the full concatenated (K, V)."""
        if self._cache is None:
            return self._fallback.update(layer_idx, k, v)

        if k.shape != (1, k.shape[1], self._n_kv_heads, self._head_dim):
            # Not a single-seq contiguous step — deviate to the numpy backend.
            return self._fallback.update(layer_idx, k, v)

        n_new = int(k.shape[1])
        base = self._lens[layer_idx]
        if base < self._seq_capacity:
            n_copy = min(n_new, self._seq_capacity - base)
            for which, arr in (("k", k), ("v", v)):
                dst = self._layer_ptr(layer_idx, which) + base * self.stride
                ctypes.memmove(
                    dst,
                    np.ascontiguousarray(arr, dtype=np.float32).ctypes.data,
                    n_copy * self.stride * 4,
                )
        self._lens[layer_idx] = min(base + n_new, self._seq_capacity)
        self._cache.seq_len = max(self._lens)

        full, cf = self.get(layer_idx)
        return full, cf

    def get(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the concatenated (K, V) for a layer, or None."""
        if self._cache is None:
            return self._fallback.get(layer_idx)
        n = self._lens[layer_idx]
        if n == 0:
            return None
        out = []
        for which in ("k", "v"):
            view = np.ctypeslib.as_array(
                self._layer_ptr(layer_idx, which), shape=(self._seq_capacity, self.stride)
            )[:n]
            out.append(view.reshape(1, n, self._n_kv_heads, self._head_dim).copy())
        return tuple(out)

    def reset(self) -> None:
        if self._cache is not None:
            self._lib.transformer_kv_cache_reset(self._cache)
        self._lens = [0] * self._n_layers
        if self._fallback is not None:
            self._fallback.reset()

    def free(self) -> None:
        """Release the C buffers. No-op on the numpy fallback."""
        if self._cache is not None and self._lib is not None:
            self._lib.transformer_kv_cache_free(self._cache)
        self._cache = None
        self._lib = None
        self._fallback = None

    close = free
