"""Unified knowledge state: one store, one API, feeds the model.

Consolidates HDMemoryStore, WorkingMemory, FormalLogicEngine,
KnowledgeGraph, and model weights behind a single KnowledgeManager.

Scoring: entries are scored by recency (exponential time decay) and
frequency (access count), so the worker trains high-value knowledge first.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """A single piece of knowledge in the unified store."""

    text: str
    source: str = "user"
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    trained: bool = False
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def to_training_line(self) -> str:
        return self.text.strip()

    def score(self, half_life_hours: float = 24.0) -> float:
        """Priority score: recency (exponential decay) × frequency (log-scaled).

        - Recency: 1.0 at creation, halves every `half_life_hours`.
        - Frequency: log2(1 + access_count), so 0 accesses = 0, 1 = 1, 3 = 2, 7 = 3.
        - Combined: recency * (1 + frequency), range ~[0, ∞).
        """
        age_hours = (time.time() - self.timestamp) / 3600.0
        recency = math.pow(0.5, age_hours / half_life_hours)
        frequency = math.log2(1 + self.access_count)
        return recency * (1.0 + frequency)

    def record_access(self) -> None:
        """Record that this entry was accessed (query hit)."""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class QueryResult:
    """Merged result from all knowledge stores."""

    text: str
    source: str
    score: float
    store: str  # which store produced this
    entry_ref: KnowledgeEntry | None = None  # for access tracking on cache hits


class KnowledgeCache:
    """LRU cache for frequent queries."""

    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._cache: OrderedDict[str, list[QueryResult]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> list[QueryResult] | None:
        if query in self._cache:
            self._cache.move_to_end(query)
            self.hits += 1
            return self._cache[query]
        self.misses += 1
        return None

    def put(self, query: str, results: list[QueryResult]) -> None:
        self._cache[query] = results
        self._cache.move_to_end(query)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def invalidate(self) -> None:
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class KnowledgeState:
    """Unified state across all knowledge stores.

    Write-through: add() goes to HD memory + working memory + KG + queue.
    Query: searches HD + KG + working memory, returns merged results.
    """

    def __init__(
        self,
        hd_memory: Any = None,
        working_memory: Any = None,
        logic_engine: Any = None,
        knowledge_graph: Any = None,
        max_entries: int = 10000,
    ):
        self._hd_memory = hd_memory
        self._working_memory = working_memory
        self._logic_engine = logic_engine
        self._knowledge_graph = knowledge_graph
        self._max_entries = max_entries

        self._entries: list[KnowledgeEntry] = []
        self._train_queue: list[KnowledgeEntry] = []
        self._lock = threading.Lock()

    def add(self, text: str, source: str = "user", tags: list[str] | None = None) -> KnowledgeEntry:
        """Add knowledge to all stores. Returns the entry."""
        entry = KnowledgeEntry(text=text, source=source, tags=tags or [])

        with self._lock:
            self._entries.append(entry)
            self._train_queue.append(entry)

            # Evict oldest if over limit
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]

        # Write-through to external stores (no lock held)
        if self._hd_memory is not None:
            try:
                self._hd_memory.add(text, role=source)
            except Exception:
                logger.debug("HD memory add failed", exc_info=True)

        if self._working_memory is not None:
            try:
                self._working_memory.add(text)
            except Exception:
                logger.debug("Working memory add failed", exc_info=True)

        if self._knowledge_graph is not None:
            try:
                self._knowledge_graph.add_fact(
                    subject=source, predicate="states", obj=text[:100]
                )
            except Exception:
                logger.debug("KG add failed", exc_info=True)

        return entry

    def query(self, query: str, top_k: int = 5) -> list[QueryResult]:
        """Search all stores, return merged results sorted by score."""
        results: list[QueryResult] = []

        # HD memory search
        if self._hd_memory is not None:
            try:
                hd_results = self._hd_memory.search(query, top_k=top_k)
                for _item_id, content, score in hd_results:
                    results.append(QueryResult(text=content, source="hd", score=score, store="hd_memory"))
            except Exception:
                logger.debug("HD search failed", exc_info=True)

        # Knowledge graph query
        if self._knowledge_graph is not None:
            try:
                kg_results = self._knowledge_graph.query(subject=query)
                for fact in kg_results:
                    results.append(QueryResult(
                        text=f"{fact.subject} {fact.predicate} {fact.object}",
                        source="kg",
                        score=0.8,
                        store="knowledge_graph",
                    ))
            except Exception:
                logger.debug("KG query failed", exc_info=True)

        # Working memory recent
        if self._working_memory is not None:
            try:
                recent = self._working_memory.get_recent()
                for item in recent:
                    if query.lower() in item.lower():
                        results.append(QueryResult(
                            text=item, source="working", score=0.6, store="working_memory"
                        ))
            except Exception:
                logger.debug("WM query failed", exc_info=True)

        # Text match on entries — record access for frequency scoring
        query_lower = query.lower()
        for entry in reversed(self._entries):
            if query_lower in entry.text.lower():
                entry.record_access()
                results.append(QueryResult(
                    text=entry.text, source=entry.source, score=0.7, store="entries",
                    entry_ref=entry,
                ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get_context(self, query: str, max_chars: int = 1000) -> str:
        """Get merged context string for prompt injection."""
        results = self.query(query, top_k=10)
        if not results:
            return ""

        lines = []
        total = 0
        for r in results:
            line = f"[{r.source}]: {r.text}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)

        return "\n".join(lines)

    def get_training_data(self, max_entries: int = 0) -> str:
        """Export untrained entries as training text, highest priority first.

        Sorted by score (recency × frequency) so the model trains on
        the most valuable knowledge first.
        """
        with self._lock:
            untrained = [e for e in self._train_queue if not e.trained]
            # Priority: highest score first
            untrained.sort(key=lambda e: e.score(), reverse=True)
            if max_entries > 0:
                untrained = untrained[:max_entries]
            for e in untrained:
                e.trained = True
            self._train_queue = [e for e in self._train_queue if not e.trained]

        return "\n".join(e.to_training_line() for e in untrained)

    def stats(self) -> dict[str, Any]:
        """Return stats from all stores."""
        with self._lock:
            entry_count = len(self._entries)
            queue_size = len(self._train_queue)
            trained_count = sum(1 for e in self._entries if e.trained)

        stats: dict[str, Any] = {
            "entries": entry_count,
            "train_queue": queue_size,
            "trained": trained_count,
        }

        # Scoring summary
        with self._lock:
            if self._entries:
                scores = [e.score() for e in self._entries]
                accesses = [e.access_count for e in self._entries]
                ages_hours = [(time.time() - e.timestamp) / 3600.0 for e in self._entries]
                stats["scoring"] = {
                    "avg_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                    "avg_access_count": sum(accesses) / len(accesses),
                    "max_access_count": max(accesses),
                    "avg_age_hours": sum(ages_hours) / len(ages_hours),
                }
                # Freshness: entries accessed in last hour
                recent = sum(1 for e in self._entries if (time.time() - e.last_accessed) < 3600)
                stats["scoring"]["recently_accessed"] = recent

        if self._hd_memory is not None:
            try:
                stats["hd_memory"] = self._hd_memory.get_stats()
            except Exception:
                pass

        if self._knowledge_graph is not None:
            try:
                stats["knowledge_graph"] = self._knowledge_graph.stats()
            except Exception:
                pass

        return stats

    def clear(self) -> None:
        """Clear all stores."""
        with self._lock:
            self._entries.clear()
            self._train_queue.clear()

        if self._hd_memory is not None:
            try:
                self._hd_memory.clear()
            except Exception:
                pass

        if self._working_memory is not None:
            try:
                self._working_memory.clear()
            except Exception:
                pass


class KnowledgeWorker:
    """Background worker that trains the model on queued knowledge."""

    def __init__(self, state: KnowledgeState, trainer_factory: Any = None):
        self._state = state
        self._trainer_factory = trainer_factory
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_train_time: float = 0
        self._train_count: int = 0

    def start(self, interval: float = 60.0) -> None:
        """Start background training thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(interval,), daemon=True, name="knowledge-worker"
        )
        self._thread.start()
        logger.info("KnowledgeWorker started (interval=%ss)", interval)

    def stop(self) -> None:
        """Stop background training."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def train_now(self) -> dict[str, Any] | None:
        """Train immediately on queued knowledge."""
        training_text = self._state.get_training_data()
        if not training_text.strip():
            return None

        if self._trainer_factory is None:
            logger.debug("No trainer factory, skipping train")
            return None

        try:
            trainer = self._trainer_factory(training_text)
            result = trainer.train()
            self._train_count += 1
            self._last_train_time = time.time()
            return {
                "success": result.success,
                "final_loss": result.final_loss,
                "steps": result.global_step,
                "entries_trained": training_text.count("\n") + 1,
            }
        except Exception as e:
            logger.error("Training failed: %s", e)
            return {"success": False, "error": str(e)}

    def _loop(self, interval: float) -> None:
        while not self._stop_event.wait(interval):
            self.train_now()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "train_count": self._train_count,
            "last_train_time": self._last_train_time,
            "running": self._thread is not None and self._thread.is_alive(),
        }


class KnowledgeManager:
    """Unified facade over all knowledge stores.

    Usage:
        km = KnowledgeManager(hd_memory=..., working_memory=..., ...)
        km.add("Paris is the capital of France")
        context = km.get_context("What is the capital of France?")
        results = km.query("capital of France")
    """

    def __init__(
        self,
        hd_memory: Any = None,
        working_memory: Any = None,
        logic_engine: Any = None,
        knowledge_graph: Any = None,
        cache_size: int = 256,
        max_entries: int = 10000,
    ):
        self.state = KnowledgeState(
            hd_memory=hd_memory,
            working_memory=working_memory,
            logic_engine=logic_engine,
            knowledge_graph=knowledge_graph,
            max_entries=max_entries,
        )
        self.cache = KnowledgeCache(max_size=cache_size)
        self.worker = KnowledgeWorker(self.state)

    def add(self, text: str, source: str = "user", tags: list[str] | None = None) -> KnowledgeEntry:
        entry = self.state.add(text, source=source, tags=tags)
        self.cache.invalidate()
        return entry

    def query(self, query: str, top_k: int = 5) -> list[QueryResult]:
        cached = self.cache.get(query)
        if cached is not None:
            # Record access on cache hits too (frequency tracking)
            for r in cached:
                if r.entry_ref is not None:
                    r.entry_ref.record_access()
            return cached

        results = self.state.query(query, top_k=top_k)
        self.cache.put(query, results)
        return results

    def get_context(self, query: str, max_chars: int = 1000) -> str:
        return self.state.get_context(query, max_chars=max_chars)

    def train(self) -> dict[str, Any] | None:
        return self.worker.train_now()

    def start_worker(self, interval: float = 60.0) -> None:
        self.worker.start(interval=interval)

    def stop_worker(self) -> None:
        self.worker.stop()

    def stats(self) -> dict[str, Any]:
        s = self.state.stats()
        s["cache"] = {
            "hits": self.cache.hits,
            "misses": self.cache.misses,
            "hit_rate": self.cache.hit_rate,
        }
        s["worker"] = self.worker.stats
        return s

    def clear(self) -> None:
        self.state.clear()
        self.cache.invalidate()
