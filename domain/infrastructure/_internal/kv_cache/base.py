"""KV cache base — C-backed with paged + session support."""

from __future__ import annotations

import ctypes
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

logger = logging.getLogger("slo.kv_cache")

# ── C library loading ───────────────────────────────────────────────────────

_C_LIB = None
_HAS_C = False


def _load_c_lib():
    global _C_LIB, _HAS_C
    if _C_LIB is not None:
        return _C_LIB

    candidates = [
        Path(__file__).parent.parent.parent.parent
        / "inference"
        / "_internal"
        / "native"
        / "libtransformer_forward.dylib",
        Path(__file__).parent.parent.parent.parent
        / "inference"
        / "_internal"
        / "native"
        / "libtransformer_forward.so",
    ]
    env_path = os.environ.get("MAN_TRANSFORMER_LIB", "")
    if env_path:
        candidates.append(Path(env_path))

    for p in candidates:
        if p.exists():
            try:
                _C_LIB = ctypes.CDLL(str(p))
                _HAS_C = True
                logger.info("Loaded C KV cache backend: %s", p, extra={"tag": "KV"})
                return _C_LIB
            except OSError:
                continue

    _HAS_C = False
    logger.info("C KV cache unavailable, using numpy fallback", extra={"tag": "KV"})
    return None


# ── KVCacheBase ─────────────────────────────────────────────────────────────


class KVCacheBase:
    """Per-layer key-value cache with C backend.

    Supports three modes:
      1. Concat mode: update() + get() — returns concatenated K/V
      2. Paged mode: update() + get_blocks() — returns block refs, no alloc
      3. Session mode: cross-turn prefix caching with LRU eviction

    All modes share the same storage. Switch via API call.
    """

    __slots__ = (
        "_n_layers",
        "_k",
        "_v",
        "_use_c",
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
        self._k: list[np.ndarray | None] = [None] * n_layers
        self._v: list[np.ndarray | None] = [None] * n_layers
        self._use_c = _HAS_C or (_load_c_lib() is not None)

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
        if self._k[0] is None:
            return 0
        return self._k[0].shape[1]

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def is_paged(self) -> bool:
        return self._paged

    # ── Core: update / get / reset ───────────────────────────────────────

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Append new K, V to cache, return full (K, V)."""
        k_prev = self._k[layer_idx]
        if k_prev is None:
            self._k[layer_idx] = k
            self._v[layer_idx] = v
        else:
            self._k[layer_idx] = np.concatenate([k_prev, k], axis=1)
            self._v[layer_idx] = np.concatenate([self._v[layer_idx], v], axis=1)

        if self._paged:
            self._write_block(layer_idx, k, v)

        return self._k[layer_idx], self._v[layer_idx]

    def get(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return concatenated (K, V) for layer_idx, or None."""
        if self._k[layer_idx] is None:
            return None
        return self._k[layer_idx], self._v[layer_idx]

    def reset(self) -> None:
        """Clear all cached data."""
        self._k = [None] * self._n_layers
        self._v = [None] * self._n_layers
        if self._paged:
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
        """
        if not self._paged:
            k, v = self.get(layer_idx)
            if k is None:
                return [], [], 0
            return [k], [v], k.shape[1]

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
