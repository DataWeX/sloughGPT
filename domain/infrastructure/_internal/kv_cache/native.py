"""Native KV state — C backend (ctypes) with an internal numpy fallback.

``NativeKVState`` is the concatenating per-layer KV state used for greedy
decoding. Two backends, one API:

- C path: mirrors the layout of ``libtransformer_forward`` — one flat float
  buffer per K/V, ``n_kv_heads * head_dim`` floats per token step, layers
  laid out contiguously. Driven through ``transformer_kv_cache_init`` /
  ``reset`` / ``free``.
- numpy path: preallocated per-layer buffers with slice-write appends
  (grown on demand) — O(T) copies over a decode where concat is O(T^2).

When the C library is absent or unloadable (it ships as a macOS dylib next
to ``domain.inference._internal.native.bindings``), the numpy backend serves
the same API. This is *native*; the modular default lives in :mod:`.kv_state`.
"""

from __future__ import annotations

import ctypes

import numpy as np


class _NumpyBackend:
    """Internal numpy append backend for NativeKVState — preallocated, no concat.

    Appending writes only the new tokens into a grown-on-demand capacity
    buffer (slice write along the sequence axis), so a T-token decode costs
    O(T) copies instead of ``np.concatenate``'s O(T^2). Rank-agnostic: the
    sequence axis is 1 in both the 3D ``(heads, seq, head_dim)`` and 4D
    ``(1, seq, heads, head_dim)`` conventions.
    """

    __slots__ = (
        "_n_layers",
        "_max_cap",
        "_k",
        "_v",
        "_lens",
        "_cap",
        "_sk",
        "_sv",
        "_quant",
    )

    _MISSING = object()

    def __init__(self, n_layers: int, cap: int = 0, quantized: bool = False):
        self._n_layers = n_layers
        self._max_cap = max(cap, 64)  # hard ceiling for geometric growth
        self._k: list[np.ndarray | None] = [None] * n_layers
        self._v: list[np.ndarray | None] = [None] * n_layers
        self._sk: list[np.ndarray | None] = [None] * n_layers
        self._sv: list[np.ndarray | None] = [None] * n_layers
        self._lens = [0] * n_layers
        self._cap = 0  # per-layer buffers grow to the same capacity
        self._quant = quantized

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def seq_len(self) -> int:
        return max(self._lens)

    # ── raw access (composition surface for callers that bind buffers) ──

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def lens(self) -> list[int]:
        return self._lens

    @property
    def quantized(self) -> bool:
        return self._quant

    @property
    def buffers(self) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
        return self._k, self._v

    @property
    def scales(self) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
        return self._sk, self._sv

    def bind(
        self,
        n_layers: int,
        total_len: int,
        nkv: int | list[int],
        head_dim: int,
        quantized: bool,
    ) -> None:
        """Allocate exact preallocated buffers for every layer (zero-filled).

        ``nkv`` may be a per-layer list (GQA) or a single head count. Sets
        the capacity and resets all fill lengths, keeping the same backend.
        """
        self._n_layers = n_layers
        self._lens = [0] * n_layers
        self._cap = total_len
        self._max_cap = total_len
        self._quant = quantized
        nkv_list = [nkv] * n_layers if isinstance(nkv, int) else list(nkv)
        dtype = np.int8 if quantized else np.float32
        self._k = [np.zeros((1, total_len, nvk, head_dim), dtype=dtype) for nvk in nkv_list]
        self._v = [np.zeros((1, total_len, nvk, head_dim), dtype=dtype) for nvk in nkv_list]
        if quantized:
            self._sk = [
                np.zeros((1, total_len, nvk, 1), dtype=np.float32) for nvk in nkv_list
            ]
            self._sv = [
                np.zeros((1, total_len, nvk, 1), dtype=np.float32) for nvk in nkv_list
            ]
        else:
            self._sk = [None] * n_layers
            self._sv = [None] * n_layers

    def grow_to(self, cap: int) -> None:
        """Resize every layer to at least ``cap``, zero-filling new space."""
        if cap <= self._cap:
            return
        dtype = np.int8 if self._quant else np.float32
        new_k = []
        new_v = []
        for bi in range(self._n_layers):
            b = self._k[bi] if bi < len(self._k) else None
            bv = self._v[bi] if bi < len(self._v) else None
            dl = self._lens[bi] if bi < len(self._lens) else 0
            if b is None:
                new_k.append(None)
                new_v.append(None)
                continue
            nk = np.zeros((1, cap, b.shape[2], b.shape[3]), dtype=dtype)
            nk[:, :dl] = b[:, :dl]
            new_k.append(nk)
            if bv is None:
                new_v.append(np.zeros((1, cap, b.shape[2], b.shape[3]), dtype=dtype))
            else:
                nv = np.zeros((1, cap, bv.shape[2], bv.shape[3]), dtype=dtype)
                nv[:, :dl] = bv[:, :dl]
                new_v.append(nv)
        self._k, self._v = new_k, new_v
        if self._quant:
            new_sk, new_sv = [], []
            for bi in range(self._n_layers):
                sk = self._sk[bi] if bi < len(self._sk) else None
                sv = self._sv[bi] if bi < len(self._sv) else None
                dl = self._lens[bi] if bi < len(self._lens) else 0
                if sk is None:
                    new_sk.append(None)
                    new_sv.append(None)
                    continue
                nsk = np.zeros((1, cap, sk.shape[2], 1), dtype=np.float32)
                nsv = np.zeros((1, cap, sv.shape[2], 1), dtype=np.float32)
                nsk[:, :dl] = sk[:, :dl]
                nsv[:, :dl] = sv[:, :dl]
                new_sk.append(nsk)
                new_sv.append(nsv)
            self._sk, self._sv = new_sk, new_sv
        self._cap = cap

    def adopt(
        self,
        *,
        k=_MISSING,
        v=_MISSING,
        sk=_MISSING,
        sv=_MISSING,
        lens=_MISSING,
        cap=_MISSING,
        quant=_MISSING,
    ) -> None:
        """Replace individual storage fields (composition surface).

        Only the supplied fields are updated; the others keep their current
        value. Assigning ``k`` or ``v`` also syncs the layer count so later
        ``grow_to`` / ``clear`` stay consistent with the adopted buffers.
        """
        if k is not self._MISSING:
            self._k = k
            self._n_layers = len(k)
        if v is not self._MISSING:
            self._v = v
            self._n_layers = len(v)
        if sk is not self._MISSING:
            self._sk = sk
        if sv is not self._MISSING:
            self._sv = sv
        if lens is not self._MISSING:
            self._lens = lens
        if cap is not self._MISSING:
            self._cap = cap
        if quant is not self._MISSING:
            self._quant = quant

    def clear(self) -> None:
        self._k = []
        self._v = []
        self._sk = []
        self._sv = []
        self._lens = []
        self._n_layers = 0
        self._cap = 0
        self._quant = False

    def _grow(self, layer_idx: int, needed: int) -> None:
        cap = self._cap
        while cap < needed:
            cap = min(max(cap * 2, 16), self._max_cap)
            if cap < needed and cap == self._max_cap:
                cap = needed  # bounded ceiling is only a hint — never drop data
        if cap == self._cap:
            return
        old = self._k[layer_idx]
        dl = self._lens[layer_idx]
        new_k = np.empty((*old.shape[:1], cap, *old.shape[2:]), old.dtype)
        new_v = np.empty((*old.shape[:1], cap, *old.shape[2:]), old.dtype)
        new_k[:, :dl] = old[:, :dl]
        new_v[:, :dl] = self._v[layer_idx][:, :dl]
        self._k[layer_idx] = new_k
        self._v[layer_idx] = new_v
        if self._quant:
            nsk = np.empty((*self._sk[layer_idx].shape[:1], cap, *self._sk[layer_idx].shape[2:]), np.float32)
            nsv = np.empty((*self._sv[layer_idx].shape[:1], cap, *self._sv[layer_idx].shape[2:]), np.float32)
            nsk[:, :dl] = self._sk[layer_idx][:, :dl]
            nsv[:, :dl] = self._sv[layer_idx][:, :dl]
            self._sk[layer_idx] = nsk
            self._sv[layer_idx] = nsv
        self._cap = cap

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Append new K, V for a layer; return the full (K, V) as views."""
        if self._k[layer_idx] is None:
            self._cap = min(self._max_cap, max(64, k.shape[1] * 2))
            self._k[layer_idx] = np.empty((*k.shape[:1], self._cap, *k.shape[2:]), k.dtype)
            self._v[layer_idx] = np.empty_like(self._k[layer_idx])

        n = int(k.shape[1])
        base = self._lens[layer_idx]
        if base + n > self._cap:
            self._grow(layer_idx, base + n)
        self._k[layer_idx][:, base : base + n] = k
        self._v[layer_idx][:, base : base + n] = v
        self._lens[layer_idx] = base + n
        return self._k[layer_idx][:, : base + n], self._v[layer_idx][:, : base + n]

    def get(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        if self._k[layer_idx] is None:
            return None
        n = self._lens[layer_idx]
        if n == 0:
            return None
        return self._k[layer_idx][:, :n], self._v[layer_idx][:, :n]

    def reset(self) -> None:
        self._lens = [0] * self._n_layers


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
        self._fallback = _NumpyBackend(n_layers, cap=seq_capacity)

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
    def backend(self):
        """The active numpy storage backend (composition surface).

        When the C library is unavailable this is the storage that serves
        the whole object; consumers that bind exact buffers (e.g. the
        training-side state) compose against it through ``bind``/``grow_to``.
        """
        return self._fallback

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
