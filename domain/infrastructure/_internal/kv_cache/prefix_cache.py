"""Content-addressed KV prefix table — share one token prefix across sessions.

Why this exists (card 20261102_023531): cross-turn KV reuse is keyed by
``session_id`` (``SessionKVManager`` -> ``KVState._sessions``), so two clients
— or one client reconnecting, or two routes — that send the *same* system
prompt each pay a full prefill, because their session ids differ. The route
differs; the tokens up to some point do not. Measured reference points for the
same 482-token prompt on the live model: **43.02 s** from an empty state,
**1.26 s** from a warm one.

Why it is safe (unlike the semantic response tier): K/V is a pure function of
the token ids *and their positions*, so a block-aligned prefix hit is
token-exact **by construction**. There is no threshold and no similarity
score; every hit is verified by comparing the ids, which also removes the
hash-collision hazard. ``semantic_cache`` may propose a candidate and must
never decide one; this tier decides, and decides exactly.

What a hit costs: one copy of the prefix K/V into the caller's buffers — a
memcpy. What it replaces: a forward pass over those same tokens. The ratio is
why the copy is acceptable (measured in ``scripts/benchmark_kv_prefix_cache.py``).

Two lookups, in order:

1. **Exact** (O(1)): the block-aligned prefix of the query, hashed. This is the
   repeated-prompt case.
2. **Partial** (vectorised scan over at most ``max_entries`` snapshots): the
   longest shared block-aligned prefix. This is the headline case — same
   system prompt, different user turn, different session.

Storage is capped by *both* entry count and total cached tokens, because a
snapshot is ~24 KB/token for a 24-layer GQA model (2 KV heads, head_dim 64,
float32) and an unbounded table would be a memory leak with a friendly name.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

#: Tokens per cache line. Must be a power of two in spirit only — any value
#: works; 64 matches the default KV block size and keeps snapshots meaningful
#: (a 16-token prefix saves no meaningful prefill).
DEFAULT_BLOCK = 64

#: Below this, a prefix is not worth a snapshot: the copy and the LRU churn
#: cost more than the prefill it saves.
DEFAULT_MIN_TOKENS = 64

#: Snapshot count cap. Small on purpose — the table holds whole K/V prefixes,
#: not indices, and real traffic has a handful of distinct system prompts.
DEFAULT_MAX_ENTRIES = 8

#: Total cached tokens across all snapshots. 1024 tokens ≈ 25 MB on the
#: 0.5B reference model; raise deliberately, not by accident.
DEFAULT_MAX_TOKENS = 1024


class KVSnapshot:
    """One immutable, block-aligned copy of a filled KV prefix.

    ``ids`` has exactly ``filled`` entries: the snapshot is cut at a block
    boundary, so what is stored is exactly what can be served. The arrays are
    **copies**; a snapshot never aliases a live ``NumpyKVState``, whose buffers
    are mutated in place by the next turn.
    """

    __slots__ = ("ids", "k", "v", "sk", "sv", "filled", "quantized")

    def __init__(
        self,
        ids: np.ndarray,
        k: tuple[np.ndarray, ...],
        v: tuple[np.ndarray, ...],
        sk: tuple[np.ndarray | None, ...],
        sv: tuple[np.ndarray | None, ...],
        filled: int,
        quantized: bool,
    ) -> None:
        self.ids = ids
        self.k = k
        self.v = v
        self.sk = sk
        self.sv = sv
        self.filled = filled
        self.quantized = quantized

    @property
    def n_tokens(self) -> int:
        return self.filled

    @property
    def nbytes(self) -> int:
        total = 0
        for group in (self.k, self.v, self.sk, self.sv):
            for arr in group:
                if arr is not None:
                    total += arr.nbytes
        return total

    @property
    def n_layers(self) -> int:
        return len(self.k)

    @property
    def head_dim(self) -> int:
        return int(self.k[0].shape[-1]) if self.k else 0

    def __repr__(self) -> str:
        return (
            f"KVSnapshot(tokens={self.filled}, layers={self.n_layers}, "
            f"head_dim={self.head_dim}, quantized={self.quantized}, "
            f"bytes={self.nbytes})"
        )


def _digest(ids: np.ndarray, cut: int) -> bytes:
    """Content address for ``ids[:cut]`` — the token bytes, nothing else."""
    window = np.ascontiguousarray(ids.reshape(-1)[:cut], dtype=np.int64)
    return hashlib.blake2b(window.tobytes(), digest_size=16).digest()


class PrefixCache:
    """LRU table of block-aligned KV prefixes, keyed by their token content.

    Thread-safe: the server can hold one generation lock today, but an LRU
    dict mutated from two threads is a data race waiting for the day it is
    lifted, and a lock around a handful of dict operations costs nothing next
    to the prefill it saves.
    """

    __slots__ = (
        "_entries",
        "_max_entries",
        "_max_tokens",
        "_min_tokens",
        "_block",
        "_lock",
        "_hits",
        "_partial_hits",
        "_misses",
        "_stores",
        "_evictions",
        "_rejected",
    )

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        block: int = DEFAULT_BLOCK,
    ) -> None:
        self._entries: OrderedDict[bytes, KVSnapshot] = OrderedDict()
        self._max_entries = max(1, int(max_entries))
        self._max_tokens = max(0, int(max_tokens))
        self._min_tokens = max(1, int(min_tokens))
        self._block = max(1, int(block))
        self._lock = threading.Lock()
        self._hits = 0
        self._partial_hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0
        self._rejected = 0

    # ── configuration ─────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> PrefixCache | None:
        """Build from the environment; ``SLO_KV_PREFIX_CACHE=0`` disables it.

        A kill switch matters here: this tier sits in the hottest path in the
        repo, so it must be possible to switch it off on a live box without a
        revert.
        """
        import os

        if os.environ.get("SLO_KV_PREFIX_CACHE", "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            return None

        def _int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None or not raw.strip():
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        return cls(
            max_entries=_int("SLO_KV_PREFIX_CACHE_ENTRIES", DEFAULT_MAX_ENTRIES),
            max_tokens=_int("SLO_KV_PREFIX_CACHE_TOKENS", DEFAULT_MAX_TOKENS),
            min_tokens=_int("SLO_KV_PREFIX_CACHE_MIN_TOKENS", DEFAULT_MIN_TOKENS),
            block=_int("SLO_KV_PREFIX_CACHE_BLOCK", DEFAULT_BLOCK),
        )

    @property
    def block(self) -> int:
        return self._block

    @property
    def min_tokens(self) -> int:
        return self._min_tokens

    # ── write ─────────────────────────────────────────────────────────────

    def store(
        self,
        ids: np.ndarray,
        k: Any,
        v: Any,
        sk: Any = None,
        sv: Any = None,
        filled: int = 0,
        quantized: bool = False,
    ) -> bool:
        """Copy the block-aligned prefix of a filled state into the table.

        Returns ``True`` when a snapshot was stored. Refuses (with a counted
        ``rejected``) when there is no complete block, when the buffers are
        shorter than ``filled``, or when the snapshot would exceed the token
        budget on its own.
        """
        try:
            filled = int(filled)
            if filled < self._min_tokens or not k or k[0] is None:
                with self._lock:
                    self._rejected += 1
                return False
            cut = (filled // self._block) * self._block
            if cut < self._min_tokens or cut > int(k[0].shape[1]):
                with self._lock:
                    self._rejected += 1
                return False
            if self._max_tokens and cut > self._max_tokens:
                with self._lock:
                    self._rejected += 1
                return False

            ids_cut = np.array(ids.reshape(-1)[:cut], dtype=np.int64, copy=True).reshape(1, cut)
            snapshot = KVSnapshot(
                ids=ids_cut,
                k=tuple(np.array(a[:, :cut], copy=True) for a in k),
                v=tuple(np.array(a[:, :cut], copy=True) for a in v),
                sk=tuple(
                    None if a is None else np.array(a[:, :cut], copy=True) for a in (sk or ())
                ),
                sv=tuple(
                    None if a is None else np.array(a[:, :cut], copy=True) for a in (sv or ())
                ),
                filled=cut,
                quantized=bool(quantized),
            )
            key = _digest(ids_cut, cut)
        except Exception:  # pragma: no cover - a cache must never break generation
            with self._lock:
                self._rejected += 1
            return False

        with self._lock:
            self._entries[key] = snapshot
            self._entries.move_to_end(key)
            self._stores += 1
            self._evict_locked()
        return True

    def _evict_locked(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1
        while self._total_tokens_locked() > self._max_tokens and self._entries:
            self._entries.popitem(last=False)
            self._evictions += 1

    def _total_tokens_locked(self) -> int:
        return sum(s.filled for s in self._entries.values())

    # ── read ──────────────────────────────────────────────────────────────

    def match(self, input_ids: np.ndarray) -> tuple[KVSnapshot, int] | None:
        """Longest block-aligned prefix of ``input_ids`` present in the table.

        Returns ``(snapshot, matched_tokens)`` or ``None``. Both paths verify
        by comparing ids, so a hash collision cannot serve a wrong prefix; the
        digest is an index, never the evidence.
        """
        try:
            ids = np.asarray(input_ids).reshape(-1)
            length = int(ids.shape[0])
            if length < self._min_tokens:
                with self._lock:
                    self._misses += 1
                return None
            aligned = (length // self._block) * self._block
            key = _digest(ids, aligned) if aligned >= self._min_tokens else None
        except Exception:  # pragma: no cover - defensive
            with self._lock:
                self._misses += 1
            return None

        with self._lock:
            if key is not None:
                snap = self._entries.get(key)
                if (
                    snap is not None
                    and snap.filled == aligned
                    and np.array_equal(snap.ids[0], ids[:aligned])
                ):
                    self._entries.move_to_end(key)
                    self._hits += 1
                    return snap, aligned

            best: tuple[KVSnapshot, int] | None = None
            for snap in self._entries.values():
                lim = min(snap.filled, length)
                if lim < self._min_tokens:
                    continue
                # Vectorised common-prefix length: the first mismatch index.
                equal = snap.ids[0, :lim] == ids[:lim]
                shared = lim if bool(equal.all()) else int(np.argmin(equal))
                shared = (shared // self._block) * self._block
                if shared < self._min_tokens:
                    continue
                if best is None or shared > best[1]:
                    best = (snap, shared)

            if best is None:
                self._misses += 1
                return None
            self._entries.move_to_end(_digest(best[0].ids, best[0].filled))
            self._partial_hits += 1
            return best

    # ── lifecycle ─────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Drop every snapshot. Called when the weights change."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get_stats(self) -> dict[str, Any]:
        """Counters for the health/status surface."""
        with self._lock:
            entries = len(self._entries)
            tokens = self._total_tokens_locked()
            lookups = self._hits + self._partial_hits + self._misses
            return {
                "entries": entries,
                "cached_tokens": tokens,
                "bytes": sum(s.nbytes for s in self._entries.values()),
                "max_entries": self._max_entries,
                "max_tokens": self._max_tokens,
                "min_tokens": self._min_tokens,
                "block": self._block,
                "hits": self._hits,
                "partial_hits": self._partial_hits,
                "misses": self._misses,
                "stores": self._stores,
                "evictions": self._evictions,
                "rejected": self._rejected,
                "hit_rate": ((self._hits + self._partial_hits) / lookups) if lookups else 0.0,
            }
