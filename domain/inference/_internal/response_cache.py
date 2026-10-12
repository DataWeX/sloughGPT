"""
Response Cache — unified exact-key LLM response caching for the serving path.

Why this exists:
  ``SemanticCache`` (same package) does similarity matching and is wired only
  into ``SloEngine``/``soul.py`` — it never sees the real chat serving path
  (``ChatDomain.respond`` -> provider). The serving path had no response-level
  cache, and the one cache that did sit near it (``SessionKVCache`` in
  model_server) poisoned benchmarks because its key omitted infrastructure
  state (model identity/params), so identical prompts replayed stale tokens.

The invariant this module enforces: **a cache key is valid only if every input
that can change the output is in the key.** The exact key is a sha256 over the
canonicalized messages + every sampling parameter + the model id + a
process-wide generation token. Model load/unload/swap bumps the generation
token, which invalidates both tiers at once (coherence by version bump, not
lazy per-entry revalidation).

Two tiers, one stats surface:
  * Exact tier (always on): content-addressed lookup. Deterministic replay of
    a previously generated response under an identical request tuple.
  * Semantic tier (opt-in, off by default): reuses ``SemanticCache`` for
    paraphrase matching, gated behind the existing ``semantic_cache`` feature
    flag AND restricted to deterministic requests. Semantic matching is unsafe
    for general chat (near-identical phrasing can differ in meaning), so it is
    never the primary lookup.

Correctness gates for storing a response:
  * Non-streaming only (a cached response cannot serve a half-stream).
  * Deterministic requests only (temperature below the cutoff) — replaying a
    stochastic sample would be a correctness lie, not a hit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("slo.response_cache")

# Requests at or below this temperature are treated as deterministic enough to
# cache and replay. Above it, sampling is stochastic and a replay would not be
# a faithful response to the request.
DETERMINISTIC_TEMPERATURE = 0.2

# Process-wide generation token. Bumped on every model load/unload/swap. It is
# baked into every key, so a bump invalidates both tiers coherently.
_generation_lock = threading.Lock()
_generation: int = 0


def bump_cache_generation() -> int:
    """Invalidate every cached response by advancing the generation token.

    Called on model load/unload/swap. Returns the new generation value.
    """
    global _generation
    with _generation_lock:
        _generation += 1
        gen = _generation
    logger.debug("response-cache generation bumped to %s", gen)
    return gen


def current_generation() -> int:
    return _generation


@dataclass(slots=True)
class ResponseEntry:
    """A cached deterministic response for one exact request tuple."""

    key: str
    response: str
    model: str
    created_at: float
    last_accessed: float
    hit_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class ResponseCache:
    """Thread-safe LRU+TTL exact response cache with an optional semantic tier.

    Args:
        max_entries: LRU capacity for the exact tier.
        ttl_seconds: Entries older than this are treated as absent.
        semantic_enabled: Opt-in paraphrase tier. When True and the
            ``semantic_cache`` feature flag is enabled, a miss on the exact
            tier falls back to ``SemanticCache`` similarity matching.
    """

    def __init__(
        self,
        max_entries: int = 512,
        ttl_seconds: float = 1800.0,
        semantic_enabled: bool = False,
    ):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.semantic_enabled = semantic_enabled

        self._entries: OrderedDict[str, ResponseEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._semantic: Any | None = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "expirations": 0,
            "bypass_stochastic": 0,
            "bypass_stream": 0,
            "semantic_hits": 0,
        }

    # -- key discipline ---------------------------------------------------

    @staticmethod
    def build_key(
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str = "",
        generation: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Canonicalize the full request tuple and hash it.

        Every input that can change the output participates: the ordered
        messages (role+content), the system prompt, the model id, the sampling
        params, and the generation token. Changing any one yields a different
        key (a miss), which is the correctness property the old SessionKVCache
        key lacked.
        """
        canon_msgs = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
        ]
        payload = {
            "messages": canon_msgs,
            "model": model,
            "temperature": round(float(temperature), 4),
            "max_tokens": int(max_tokens),
            "system_prompt": system_prompt,
            "generation": current_generation() if generation is None else generation,
            "extra": extra or {},
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def is_deterministic(temperature: float) -> bool:
        return float(temperature) <= DETERMINISTIC_TEMPERATURE

    # -- lookup -----------------------------------------------------------

    def get(self, key: str) -> str | None:
        """Return the cached response for an exact key, or None."""
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if now - entry.created_at > self.ttl_seconds:
                del self._entries[key]
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return None
            self._entries.move_to_end(key)
            entry.hit_count += 1
            entry.last_accessed = now
            self._stats["hits"] += 1
            return entry.response

    def get_semantic(self, query: str, model: str, temperature: float) -> str | None:
        """Paraphrase fallback. Only when enabled, deterministic, and flagged."""
        if not self.semantic_enabled:
            return None
        if not self.is_deterministic(temperature):
            return None
        try:
            from domain.shared import is_enabled as _flag_enabled

            if not _flag_enabled("semantic_cache"):
                return None
        except Exception:
            return None

        sem = self._get_semantic_cache()
        if sem is None:
            return None
        response = sem.get(query)
        if response is not None:
            with self._lock:
                self._stats["semantic_hits"] += 1
        return response

    def _get_semantic_cache(self):
        if self._semantic is None:
            try:
                from domain.inference._internal.semantic_cache import SemanticCache

                # High threshold: only strong paraphrases qualify, keeping the
                # opt-in semantic tier conservative.
                self._semantic = SemanticCache(
                    max_entries=self.max_entries,
                    similarity_threshold=0.85,
                    ttl_seconds=self.ttl_seconds,
                )
            except Exception as e:
                logger.debug("semantic tier unavailable: %s", e)
                return None
        return self._semantic

    # -- store ------------------------------------------------------------

    def put(
        self,
        key: str,
        response: str,
        model: str,
        prompt_for_semantic: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a response under an exact key. Returns True if stored."""
        if not response:
            return False
        now = time.time()
        with self._lock:
            if len(self._entries) >= self.max_entries:
                self._evict_lru_locked()
            self._entries[key] = ResponseEntry(
                key=key,
                response=response,
                model=model,
                created_at=now,
                last_accessed=now,
                metadata=metadata or {},
            )
            self._stats["stores"] += 1

        if self.semantic_enabled and prompt_for_semantic:
            sem = self._get_semantic_cache()
            if sem is not None:
                try:
                    sem.put(prompt_for_semantic, response, metadata={"model": model})
                except Exception as e:
                    logger.debug("semantic store failed: %s", e)
        return True

    def _evict_lru_locked(self) -> bool:
        if not self._entries:
            return False
        self._entries.popitem(last=False)
        self._stats["evictions"] += 1
        return True

    # -- bypass counters (for observability of requests we deliberately skip) -

    def record_bypass_stochastic(self) -> None:
        with self._lock:
            self._stats["bypass_stochastic"] += 1

    def record_bypass_stream(self) -> None:
        with self._lock:
            self._stats["bypass_stream"] += 1

    # -- lifecycle --------------------------------------------------------

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            if self._semantic is not None:
                try:
                    self._semantic.clear()
                except Exception:
                    pass
        return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0
            return {
                "generation": current_generation(),
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "semantic_enabled": self.semantic_enabled,
                "hit_rate": f"{hit_rate:.1%}",
                **self._stats,
            }

    def reset_stats(self) -> None:
        with self._lock:
            for k in self._stats:
                self._stats[k] = 0


# -- process-wide singleton used by the chat seam ---------------------------

_singleton: ResponseCache | None = None
_singleton_lock = threading.Lock()


def get_response_cache() -> ResponseCache:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ResponseCache()
    return _singleton


def reset_response_cache_for_tests() -> None:
    global _singleton
    with _singleton_lock:
        _singleton = None


__all__ = [
    "ResponseCache",
    "ResponseEntry",
    "DETERMINISTIC_TEMPERATURE",
    "bump_cache_generation",
    "current_generation",
    "get_response_cache",
    "reset_response_cache_for_tests",
]
