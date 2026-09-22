"""KnowledgeEngine — unified feature facade for knowledge management.

Wraps KnowledgeMemory, KnowledgeIngestor, and DataFilter into a single
cohesive API. Routers should use this, not import from domain.knowledge._internal
or domain.learner._internal directly.

Usage:
    from domain.knowledge.engine import KnowledgeEngine

    engine = KnowledgeEngine()
    item = engine.store("The sky is blue", topic="science")
    results = engine.query("What color is the sky?")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("slo.knowledge.engine")


@dataclass
class KnowledgeResult:
    """Result from a knowledge operation."""

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeEngine:
    """Unified knowledge feature API.

    Provides a single entry point for all knowledge operations:
    - CRUD (store, query, get, update, delete)
    - Ingestion (URL, file, batch)
    - Intelligence (topic suggestion, gap detection, duplicate check)
    - Search (semantic + keyword)
    """

    def __init__(self) -> None:
        self._memory: Any = None
        self._ingestor: Any = None
        self._filter: Any = None

    def _get_memory(self) -> Any:
        if self._memory is None:
            from domain.knowledge import get_knowledge_memory

            self._memory = get_knowledge_memory()
        return self._memory

    def _get_ingestor(self) -> Any:
        if self._ingestor is None:
            from domain.knowledge import get_knowledge_ingestor

            self._ingestor = get_knowledge_ingestor()
        return self._ingestor

    def _get_filter(self) -> Any:
        if self._filter is None:
            from domain.knowledge import get_data_filter

            self._filter = get_data_filter()
        return self._filter

    def get_memory(self) -> Any:
        """Return the underlying KnowledgeMemory instance.

        Bridge for legacy callers still built against the memory API. New
        code should use the facade methods (store/query/update/...) which
        wrap the same memory behind KnowledgeResult.
        """
        return self._get_memory()

    def get_ingestor(self) -> Any:
        """Return the underlying KnowledgeIngestor instance.

        Bridge for legacy callers built against the ingestor API.
        """
        return self._get_ingestor()

    def store(
        self,
        content: str,
        topic: str = "general",
        source: str = "manual",
        importance: float = 0.7,
    ) -> KnowledgeResult:
        """Store a knowledge item.

        Args:
            content: The knowledge content.
            topic: Topic category.
            source: Source of the knowledge.
            importance: Importance score (0-1).

        Returns:
            KnowledgeResult with stored item data.
        """
        try:
            from domain.knowledge import KnowledgeFact

            memory = self._get_memory()
            fact = KnowledgeFact(
                content=content,
                topic=topic,
                source=source,
                importance=importance,
            )
            stored = memory.add_fact(fact)
            return KnowledgeResult(
                success=True,
                data={
                    "stored": stored,
                    "content": content,
                    "topic": topic,
                    "source": source,
                    "importance": importance,
                },
            )
        except Exception as e:
            logger.error("Knowledge store failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def query(self, search: str, top_k: int = 10) -> KnowledgeResult:
        """Search knowledge items.

        Args:
            search: Search query.
            top_k: Maximum results to return.

        Returns:
            KnowledgeResult with list of matching items.
        """
        try:
            memory = self._get_memory()
            results = memory.search(search, top_k=top_k)
            return KnowledgeResult(
                success=True,
                data=results,
                metadata={"query": search, "count": len(results)},
            )
        except Exception as e:
            logger.error("Knowledge query failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def get(self, item_id: str) -> KnowledgeResult:
        """Get a knowledge item by ID.

        Args:
            item_id: The item ID.

        Returns:
            KnowledgeResult with item data.
        """
        try:
            memory = self._get_memory()
            item = next(
                (i for i in memory.list_all(top_k=10000) if str(i.get("id")) == str(item_id)),
                None,
            )
            if item is None:
                return KnowledgeResult(success=False, error=f"Item not found: {item_id}")
            return KnowledgeResult(success=True, data=item)
        except Exception as e:
            logger.error("Knowledge get failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def delete(self, item_id: str) -> KnowledgeResult:
        """Delete a knowledge item.

        Args:
            item_id: The item ID.

        Returns:
            KnowledgeResult with deletion status.
        """
        try:
            memory = self._get_memory()
            deleted = memory.delete_by_id(item_id)
            if not deleted:
                return KnowledgeResult(success=False, error=f"Item not found: {item_id}")
            return KnowledgeResult(success=True, data={"deleted": True})
        except Exception as e:
            logger.error("Knowledge delete failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def ingest_url(self, url: str, source: str = "direct") -> KnowledgeResult:
        """Ingest knowledge from a URL.

        Args:
            url: The URL to ingest.
            source: Source label.

        Returns:
            KnowledgeResult with ingestion status.
        """
        try:
            ingestor = self._get_ingestor()
            result = ingestor.ingest_url(url)
            return KnowledgeResult(
                success=True,
                data=result,
                metadata={"url": url, "source": source},
            )
        except Exception as e:
            logger.error("URL ingestion failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def suggest_topic(self, content: str) -> KnowledgeResult:
        """Suggest a topic for given content.

        Args:
            content: The content to categorize.

        Returns:
            KnowledgeResult with suggested topic.
        """
        try:
            memory = self._get_memory()
            topic = memory.suggest_topic(content) if hasattr(memory, "suggest_topic") else "general"
            return KnowledgeResult(success=True, data=topic)
        except Exception as e:
            logger.error("Topic suggestion failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def find_gaps(self) -> KnowledgeResult:
        """Find knowledge gaps — topics with few items.

        Returns:
            KnowledgeResult with list of gaps.
        """
        try:
            memory = self._get_memory()
            gaps = memory.find_gaps() if hasattr(memory, "find_gaps") else []
            return KnowledgeResult(success=True, data=gaps)
        except Exception as e:
            logger.error("Gap detection failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def stats(self) -> KnowledgeResult:
        """Get knowledge base statistics.

        Returns:
            KnowledgeResult with stats dict.
        """
        try:
            memory = self._get_memory()
            facts = memory.list_all(top_k=5000)
            total = len(facts)
            topics: dict[str, int] = {}
            sources: dict[str, int] = {}
            importance_total = 0.0
            for f in facts:
                topics[f.get("topic", "general")] = topics.get(f.get("topic", "general"), 0) + 1
                src = f.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
                importance_total += float(f.get("importance", 0.5))
            return KnowledgeResult(
                success=True,
                data={
                    "total_items": total,
                    "topic_count": len(topics),
                    "topics": topics,
                    "sources": sources,
                    "avg_importance": round(importance_total / total, 4) if total else 0.0,
                    "total_facts": total,
                    "visited_urls": len(memory._visited) if hasattr(memory, "_visited") else 0,
                },
            )
        except Exception as e:
            logger.error("Knowledge stats failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def list_topics(self) -> KnowledgeResult:
        """List all topics with item counts.

        Returns:
            KnowledgeResult with list of {name, count} dicts.
        """
        try:
            memory = self._get_memory()
            topics: dict[str, int] = {}
            for f in memory.list_all(top_k=5000):
                topics[f.get("topic", "general")] = topics.get(f.get("topic", "general"), 0) + 1
            return KnowledgeResult(success=True, data={"topics": topics, "total": len(topics)})
        except Exception as e:
            logger.error("Topic listing failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def list_all(self, top_k: int = 200) -> KnowledgeResult:
        """List all knowledge items.

        Args:
            top_k: Maximum items to return.

        Returns:
            KnowledgeResult with list of item dicts.
        """
        try:
            memory = self._get_memory()
            items = memory.list_all(top_k=top_k) if hasattr(memory, "list_all") else []
            return KnowledgeResult(success=True, data=items)
        except Exception as e:
            logger.error("List all failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def update(
        self,
        item_id: str,
        content: str | None = None,
        topic: str | None = None,
        importance: float | None = None,
    ) -> KnowledgeResult:
        """Update a knowledge item.

        Args:
            item_id: The item ID.
            content: New content (optional).
            topic: New topic (optional).
            importance: New importance (optional).

        Returns:
            KnowledgeResult with updated item data.
        """
        try:
            memory = self._get_memory()
            current_content = content
            if current_content is None:
                current = next(
                    (i for i in memory.list_all(top_k=10000) if str(i.get("id")) == str(item_id)),
                    None,
                )
                current_content = (current or {}).get("content", "")
            ok = memory.update_fact(
                item_id,
                current_content,
                topic=topic,
                importance=importance,
            )
            return KnowledgeResult(success=True, data={"updated": ok, "id": item_id})
        except Exception as e:
            logger.error("Knowledge update failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def batch_store(
        self, items: list[dict], topic: str = "general", source: str = "batch"
    ) -> KnowledgeResult:
        """Batch store knowledge items.

        Args:
            items: List of dicts with 'content' key and optional 'topic', 'source', 'importance'.
            topic: Default topic for items without one.
            source: Default source for items without one.

        Returns:
            KnowledgeResult with count of stored items.
        """
        try:
            from domain.knowledge import KnowledgeFact

            memory = self._get_memory()
            facts = [
                KnowledgeFact(
                    content=item["content"],
                    topic=item.get("topic", topic),
                    source=item.get("source", source),
                    importance=item.get("importance", 0.7),
                )
                for item in items
                if item.get("content")
            ]
            count = memory.add_facts(facts)
            return KnowledgeResult(success=True, data={"stored": count})
        except Exception as e:
            logger.error("Batch store failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def batch_delete(self, item_ids: list[str]) -> KnowledgeResult:
        """Batch delete knowledge items.

        Args:
            item_ids: List of item IDs to delete.

        Returns:
            KnowledgeResult with count of deleted items.
        """
        try:
            memory = self._get_memory()
            count = 0
            for item_id in item_ids:
                if memory.delete_by_id(item_id):
                    count += 1
            return KnowledgeResult(success=True, data={"deleted": count})
        except Exception as e:
            logger.error("Batch delete failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def search_files(
        self, query: str, path: str = ".", extensions: list[str] | None = None, top_k: int = 10
    ) -> KnowledgeResult:
        """Search files by content.

        Args:
            query: Search query.
            path: Root path to search.
            extensions: File extensions to include.
            top_k: Maximum results.

        Returns:
            KnowledgeResult with matching files.
        """
        try:
            from pathlib import Path as _Path

            from domain.knowledge import FileIndex

            index = FileIndex()
            root = _Path(path or ".")
            if root.is_file():
                index.index_file(str(root))
            else:
                index.index_directory(str(root), extensions=set(extensions or []))
            results = index.search(query, top_k=top_k)
            return KnowledgeResult(
                success=True,
                data={"results": results, "indexed_files": index.file_count},
            )
        except Exception as e:
            logger.error("File search failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def check_duplicate(self, content: str, threshold: float = 0.85) -> KnowledgeResult:
        """Check for duplicate knowledge items.

        Args:
            content: Content to check.
            threshold: Similarity threshold (0-1).

        Returns:
            KnowledgeResult with duplicate info.
        """
        try:
            from domain.knowledge import DuplicateDetector

            detector = DuplicateDetector(threshold=threshold)
            detector.load_from_store(self._get_memory()._vector_store)
            is_dup, dup_of, score = detector.check(content)
            return KnowledgeResult(
                success=True,
                data={"is_duplicate": is_dup, "duplicate_of": dup_of, "score": score},
            )
        except Exception as e:
            logger.error("Duplicate check failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def categorize(self, content: str) -> KnowledgeResult:
        """Auto-categorize content.

        Args:
            content: Content to categorize.

        Returns:
            KnowledgeResult with suggested category.
        """
        try:
            from domain.knowledge import AutoCategorizer

            categorizer = AutoCategorizer()
            categorizer.load_from_store(self._get_memory()._vector_store)
            topic = categorizer.categorize(content)
            return KnowledgeResult(success=True, data={"topic": topic})
        except Exception as e:
            logger.error("Categorization failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def bulk_ingest(
        self,
        items: list[dict],
        topic: str = "imported",
        source: str = "bulk",
        dedup_threshold: float = 0.85,
    ) -> KnowledgeResult:
        """Bulk ingest items with deduplication.

        Args:
            items: List of dicts with 'content' key.
            topic: Default topic.
            source: Default source.
            dedup_threshold: Dedup similarity threshold.

        Returns:
            KnowledgeResult with ingest stats.
        """
        try:
            from domain.knowledge import BulkProcessor

            processor = BulkProcessor(knowledge_memory=self._get_memory())
            result = processor.ingest_texts(
                [i.get("content", "") if isinstance(i, dict) else str(i) for i in items],
                topic=topic,
                source=source,
                dedup_threshold=dedup_threshold,
            )
            return KnowledgeResult(success=True, data=result)
        except Exception as e:
            logger.error("Bulk ingest failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def categorize_with_store(self, content: str) -> KnowledgeResult:
        """Categorize content and return topic plus scored suggestions.

        Uses the existing topic vocabulary from the knowledge store.

        Returns:
            KnowledgeResult with {topic, suggestions}.
        """
        try:
            from domain.knowledge import AutoCategorizer

            categorizer = AutoCategorizer()
            categorizer.load_from_store(self._get_memory()._vector_store)
            topic = categorizer.categorize(content)
            suggestions = [
                {"topic": name, "score": round(score, 4)}
                for name, score in categorizer.suggest_topics(content, top_k=3)
            ]
            return KnowledgeResult(
                success=True,
                data={"topic": topic, "suggestions": suggestions},
            )
        except Exception as e:
            logger.error("Categorization with store failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def find_gaps_with_store(self) -> KnowledgeResult:
        """Find under-represented topics in the knowledge store.

        Returns:
            KnowledgeResult with {gaps, total_facts}.
        """
        try:
            from domain.knowledge import KnowledgeGapDetector

            memory = self._get_memory()
            detector = KnowledgeGapDetector()
            detector.load_from_store(memory._vector_store)
            return KnowledgeResult(
                success=True,
                data={"gaps": detector.find_gaps(), "total_facts": len(memory.list_all())},
            )
        except Exception as e:
            logger.error("Gap detection failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def bulk_ingest_with_memory(
        self,
        items: list[str],
        topic: str = "imported",
        source: str = "bulk",
        dedup_threshold: float = 0.85,
    ) -> KnowledgeResult:
        """Ingest a batch of raw texts with deduplication.

        Returns:
            KnowledgeResult with {added, skipped, errors}.
        """
        try:
            from domain.knowledge import BulkProcessor

            processor = BulkProcessor(knowledge_memory=self._get_memory())
            report = processor.ingest_texts(
                [str(i) for i in items],
                topic=topic,
                source=source,
                dedup_threshold=dedup_threshold,
            )
            return KnowledgeResult(success=True, data=report)
        except Exception as e:
            logger.error("Bulk ingest (memory) failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def related(self, item_id: str, top_k: int = 5) -> KnowledgeResult:
        """Get related knowledge items.

        Args:
            item_id: The item ID.
            top_k: Maximum related items.

        Returns:
            KnowledgeResult with related items.
        """
        try:
            memory = self._get_memory()
            related_items = (
                memory.get_related(item_id, top_k=top_k) if hasattr(memory, "get_related") else []
            )
            return KnowledgeResult(success=True, data=related_items)
        except Exception as e:
            logger.error("Related items failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))

    def context(self, query: str, top_k: int = 5) -> KnowledgeResult:
        """Get context items for a query.

        Args:
            query: The query.
            top_k: Maximum context items.

        Returns:
            KnowledgeResult with context items.
        """
        try:
            memory = self._get_memory()
            context_str = (
                memory.get_context_string(max_items=top_k)
                if hasattr(memory, "get_context_string")
                else ""
            )
            return KnowledgeResult(success=True, data=context_str)
        except Exception as e:
            logger.error("Context retrieval failed: %s", e)
            return KnowledgeResult(success=False, error=str(e))


_engine: KnowledgeEngine | None = None


def get_knowledge_engine() -> KnowledgeEngine:
    """Singleton accessor for KnowledgeEngine."""
    global _engine
    if _engine is None:
        _engine = KnowledgeEngine()
    return _engine
