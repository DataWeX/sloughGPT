"""The default KV state — one simple object.

``KVState`` is a per-layer key/value state for decoding:

  - paged: ``update()`` + ``get_blocks()`` — zero-copy block refs, no concat
  - state: generated token ids — ``add_tokens()``, ``position``, ``prefix_match()``
  - session: cross-turn prefix cache with LRU eviction + TTL

Concatenated access lives in :mod:`native` — ``NativeKVState`` (C + numpy)
is what the decode engine ships with. This class deliberately has no
``get()``: growing one array with ``np.concatenate`` every step is O(n^2),
so forward passes consume blocks via ``get_blocks()``.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

import numpy as np


class KVState:
    """Per-layer key/value state — paged storage + tokens + session prefix.

    ``reset()`` clears the paged K/V and the token state; cached sessions
    are untouched.
    """

    __slots__ = (
        "_n_layers",
        "_paged",
        "_block_size",
        "_k_blocks",
        "_v_blocks",
        "_paged_len",
        "_token_ids",
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
        self._paged = n_heads > 0
        self._block_size = block_size
        self._k_blocks: list[list[np.ndarray]] = (
            [[] for _ in range(n_layers)] if self._paged else []
        )
        self._v_blocks: list[list[np.ndarray]] = (
            [[] for _ in range(n_layers)] if self._paged else []
        )
        self._paged_len = 0

        # Generated-token state
        self._token_ids: list[int] = []

        # Session prefix cache (LRU + TTL)
        self._sessions: dict[str, tuple[list[int], Any, float]] = {}
        self._max_sessions = max_sessions
        self._ttl = ttl
        self._session_lock = Lock()

    # ── identity / size ───────────────────────────────────────────────────

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def seq_len(self) -> int:
        return self._paged_len if self._paged else 0

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def is_paged(self) -> bool:
        return self._paged

    # ── state: generated token ids ────────────────────────────────────────

    @property
    def tokens(self) -> list[int]:
        return list(self._token_ids)

    @property
    def token_ids(self) -> list[int]:
        return list(self._token_ids)

    @property
    def position(self) -> int:
        return len(self._token_ids)

    def add_tokens(self, ids: int | list[int]) -> None:
        """Record generated token ids on the decode state."""
        if isinstance(ids, int):
            self._token_ids.append(ids)
        else:
            self._token_ids.extend(ids)

    extend_tokens = add_tokens

    def prefix_match(self, ids: list[int]) -> int:
        """Shared prefix length between ``ids`` and the stored token state.

        Feed the previous turn's ids here to compute a start position for
        incremental decode (the ``prev_ids`` resume ``NumpyKVState`` had).
        """
        n = 0
        for a, b in zip(self._token_ids, ids, strict=False):
            if a != b:
                break
            n += 1
        return n

    # ── paged: update / reset ─────────────────────────────────────────────

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> None:
        """Append new K, V to the block list. No concat — no return value.

        Requires paged mode (n_heads > 0). Concatenated access is provided
        by NativeKVState.
        """
        if not self._paged:
            raise ValueError(
                "KVState is paged+state+session; concat access lives in "
                "NativeKVState (C backend with numpy fallback)"
            )
        self._write_block(layer_idx, k, v)

    def get_blocks(self, layer_idx: int) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        """Return (k_blocks, v_blocks, n_valid_tokens)."""
        if not self._paged:
            return [], [], 0
        return self._k_blocks[layer_idx], self._v_blocks[layer_idx], self._paged_len

    def reset(self) -> None:
        """Clear paged blocks and the token state. Sessions are untouched."""
        self._k_blocks = [[] for _ in range(self._n_layers)] if self._paged else []
        self._v_blocks = [[] for _ in range(self._n_layers)] if self._paged else []
        self._paged_len = 0
        self._token_ids = []

    def _write_block(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> None:
        n_new = k.shape[1]
        bs = self._block_size

        if layer_idx == 0:
            pos = 0
            while pos < n_new:
                take = min(bs - (pos % bs), n_new - pos)
                if take == n_new and pos == 0:
                    self._k_blocks[0].append(k)
                    self._v_blocks[0].append(v)
                else:
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

    # ── session: prefix caching with LRU ──────────────────────────────────

    def session_get(self, session_id: str, current_ids: list[int]) -> tuple[Any, int]:
        """Return (past_key_values, prefix_len) if hit, else (None, 0)."""
        with self._session_lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None, 0
            cached_ids, cached_pkv, _ = entry
            if not cached_ids:
                # Opaque entry (no prefix key): a pure lookup (empty query)
                # returns the state so session managers can delegate their
                # registers onto KVState; a prefix query is a miss. Touched
                # so LRU is by last access.
                if not current_ids:
                    self._sessions[session_id] = (cached_ids, cached_pkv, time.monotonic())
                    return cached_pkv, 0
                return None, 0
            prefix_len = 0
            for a, b in zip(cached_ids, current_ids, strict=False):
                if a != b:
                    break
                prefix_len += 1
            if prefix_len == 0:
                return None, 0
            # Touch the entry so LRU is by last access.
            self._sessions[session_id] = (cached_ids, cached_pkv, time.monotonic())
            return cached_pkv, prefix_len

    def session_store(self, session_id: str, token_ids: list[int], past_key_values: Any) -> None:
        """Store past_key_values keyed by session + token IDs."""
        with self._session_lock:
            self._session_evict_expired()
            if session_id not in self._sessions and len(self._sessions) >= self._max_sessions:
                oldest = min(self._sessions, key=lambda k: self._sessions[k][2])
                del self._sessions[oldest]
            self._sessions[session_id] = (token_ids, past_key_values, time.monotonic())

    def session_clear(self, session_id: str) -> None:
        with self._session_lock:
            self._sessions.pop(session_id, None)

    def session_clear_all(self) -> None:
        with self._session_lock:
            self._sessions.clear()

    def _session_evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, v in self._sessions.items() if now - v[2] > self._ttl]
        for k in expired:
            del self._sessions[k]
