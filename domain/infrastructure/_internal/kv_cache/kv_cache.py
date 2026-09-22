"""Unified session + paged KV cache.

``KVCache`` owns exactly two concerns:

  1. Paged mode — block-sliced K/V storage for incremental decoding
  2. Session mode — cross-turn prefix caching with LRU eviction + TTL

The concatenating path and the C backend live in :mod:`native`
(NativeKVCache — ctypes with a numpy fallback). This class deliberately
has no ``get()``: growing one array with ``np.concatenate`` every step is
O(n^2), so forward passes consume blocks via ``get_blocks()``.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

import numpy as np

# ── KVCache (session + paged) ─────────────────────────────────────────────────


class KVCache:
    """Per-layer key-value cache — session + paged only.

    Two modes, shared storage:
      1. Paged mode: update() + get_blocks() — block refs, no concat
      2. Session mode: cross-turn prefix caching with LRU eviction + TTL

    For concatenated K/V access use ``NativeKVCache`` (C + numpy).
    """

    __slots__ = (
        "_n_layers",
        "_paged",
        "_block_size",
        "_k_blocks",
        "_v_blocks",
        "_paged_len",
        "_sessions",
        "_max_sessions",
        "_ttl",
        "_session_lock",
    )

    def __init__(
        self,
        n_layers: int,
        n_heads: int = 0,
        head_dim: int = 0,
        max_seq_len: int = 2048,
        block_size: int = 64,
        max_sessions: int = 20,
        ttl: float = 600.0,
    ):
        self._n_layers = n_layers

        # Paged state (active when n_heads > 0)
        self._paged = n_heads > 0
        self._block_size = block_size
        self._k_blocks: list[list[np.ndarray]] = (
            [[] for _ in range(n_layers)] if self._paged else []
        )
        self._v_blocks: list[list[np.ndarray]] = (
            [[] for _ in range(n_layers)] if self._paged else []
        )
        self._paged_len = 0

        # Session state
        self._sessions: dict[str, tuple[list[int], Any, float]] = {}
        self._max_sessions = max_sessions
        self._ttl = ttl
        self._session_lock = Lock()

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def seq_len(self) -> int:
        if not self._paged:
            return 0
        return self._paged_len

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def is_paged(self) -> bool:
        return self._paged

    # ── Paged: update / reset ─────────────────────────────────────────────

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> None:
        """Append new K, V to the block list. No concat — no return value.

        Requires paged mode (n_heads > 0). Concatenated access is provided
        by NativeKVCache.
        """
        if not self._paged:
            raise ValueError(
                "KVCache is session+paged only; concat access lives in "
                "NativeKVCache (C backend with numpy fallback)"
            )
        self._write_block(layer_idx, k, v)

    def reset(self) -> None:
        """Clear all paged blocks."""
        self._k_blocks = [[] for _ in range(self._n_layers)]
        self._v_blocks = [[] for _ in range(self._n_layers)]
        self._paged_len = 0

    # ── Paged: write_block / get_blocks ──────────────────────────────────

    def _write_block(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> None:
        """Store K/V tensor references into block list — no copy."""
        n_new = k.shape[1]
        bs = self._block_size

        if layer_idx == 0:
            # Store raw tensor refs, slice if needed
            pos = 0
            while pos < n_new:
                take = min(bs - (pos % bs), n_new - pos)
                if take == n_new and pos == 0:
                    # Full tensor fits in one block — store ref directly
                    self._k_blocks[0].append(k)
                    self._v_blocks[0].append(v)
                else:
                    # Partial — store slice (view, not copy)
                    self._k_blocks[0].append(k[:, pos : pos + take])
                    self._v_blocks[0].append(v[:, pos : pos + take])
                pos += take
            self._paged_len += n_new
        else:
            pre_total = self._paged_len - n_new
            pos = 0
            while pos < n_new:
                take = min(bs - ((pre_total + pos) % bs), n_new - pos)
                if take == n_new and pos == 0:
                    self._k_blocks[layer_idx].append(k)
                    self._v_blocks[layer_idx].append(v)
                else:
                    self._k_blocks[layer_idx].append(k[:, pos : pos + take])
                    self._v_blocks[layer_idx].append(v[:, pos : pos + take])
                pos += take

    def get_blocks(self, layer_idx: int) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        """Return (k_blocks, v_blocks, n_valid_tokens) — no concat.

        Blocks are raw tensor refs from model forward, stored directly.
        Requires paged mode.
        """
        if not self._paged:
            return [], [], 0
        return self._k_blocks[layer_idx], self._v_blocks[layer_idx], self._paged_len

    # ── Session: prefix caching with LRU ─────────────────────────────────

    def session_get(self, session_id: str, current_ids: list[int]) -> tuple[Any, int]:
        """Return (past_key_values, prefix_len) if cache hit, else (None, 0)."""
        with self._session_lock:
            entry = self._sessions.get(session_id)
        if entry is None:
            return None, 0
        cached_ids, cached_pkv, _ = entry
        prefix_len = 0
        for a, b in zip(cached_ids, current_ids, strict=False):
            if a != b:
                break
            prefix_len += 1
        if prefix_len == 0:
            return None, 0
        return cached_pkv, prefix_len

    def session_store(self, session_id: str, token_ids: list[int], past_key_values: Any) -> None:
        """Store past_key_values keyed by session + token IDs."""
        with self._session_lock:
            self._session_evict_expired()
            if len(self._sessions) >= self._max_sessions:
                oldest = min(self._sessions, key=lambda k: self._sessions[k][2])
                del self._sessions[oldest]
            self._sessions[session_id] = (token_ids, past_key_values, time.time())

    def session_clear(self, session_id: str) -> None:
        with self._session_lock:
            self._sessions.pop(session_id, None)

    def session_clear_all(self) -> None:
        with self._session_lock:
            self._sessions.clear()

    def _session_evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, v in self._sessions.items() if now - v[2] > self._ttl]
        for k in expired:
            del self._sessions[k]
