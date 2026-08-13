"""Storage seam for the memory layer.

Producers (chat loop, future task executor) talk to ``MemoryService`` only.
This module owns the adapter to a concrete store. Swapping stores - e.g. a
task-queue-backed persistence layer for option 3 - means implementing
``MemoryProvider``, never touching the producers.
"""

import logging
from typing import Any, Dict, List, Optional, Protocol

from domains.learner.knowledge import KnowledgeFact, get_knowledge_memory

logger = logging.getLogger(__name__)


class MemoryProvider(Protocol):
    """Storage adapter contract implemented by concrete memory stores."""

    def store_turn(self, user_message: str, assistant_response: str) -> bool:
        """Persist one turn as durable memory. True when new facts stored."""

    def store_turn_facts(self, user_message: str, assistant_response: str) -> List[str]:
        """Persist one turn; return the newly stored fact texts."""

    def store(self, content: str, topic: str, source: str) -> bool:
        """Persist a single raw fact. True when newly stored."""

    def retrieve(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Return up to ``limit`` memory items relevant to ``query``."""

    def list_all(self, limit: int) -> List[Dict[str, Any]]:
        """Return stored items (most recent first), up to ``limit``."""

    def clear(self) -> int:
        """Remove every stored item; return the number removed."""

    def delete(self, ids: List[str]) -> int:
        """Remove the stored items with the given entry ids; return the count removed."""

    def update(self, item_id: str, content: str, topic: Optional[str] = None,
               importance: Optional[float] = None) -> bool:
        """Edit an existing item's text (and optionally its topic/importance)."""

    def stats(self) -> Dict[str, Any]:
        """Return provider-level statistics."""


class KnowledgeMemoryProvider:
    """MemoryProvider backed by the zero-dependency KnowledgeMemory store.

    ``store_turn`` delegates to ``auto_ingest_from_chat`` - the existing fact
    extractor that pulls declarative statements out of the assistant response,
    infers a topic from the user message, and deduplicates via content hash.

    The store is injectable so tests can point at an isolated instance;
    production callers use the module-level ``get_knowledge_memory()``.
    """

    def __init__(self, store: Optional[Any] = None):
        self._store = store

    def _get_store(self):
        """Resolve the concrete store (injected or module singleton)."""
        if self._store is not None:
            return self._store
        return get_knowledge_memory()

    def store_turn(self, user_message: str, assistant_response: str) -> bool:
        """
        Extract and persist facts from one completed turn.

        Args:
            user_message: the user's prompt text (used for topic inference).
            assistant_response: the assistant's reply to mine facts from.

        Returns:
            True when at least one new fact was stored.

        Side effects:
            - writes extracted facts into the knowledge store (persisted).
        """
        if not user_message or not assistant_response:
            return False
        try:
            added = self._get_store().auto_ingest_from_chat(user_message, assistant_response)
            return added > 0
        except Exception as e:
            logger.debug("Memory store_turn failed: %s", e)
            return False

    def store_turn_facts(self, user_message: str, assistant_response: str) -> List[str]:
        """
        Extract and persist facts from one completed turn, returning them.

        Args:
            user_message: the user's prompt text (used for topic inference).
            assistant_response: the assistant's reply to mine facts from.

        Returns:
            List of the newly stored fact texts; empty when inputs are blank,
            nothing was extracted, or storage failed.

        Side effects:
            - writes extracted facts into the knowledge store (persisted).
        """
        if not user_message or not assistant_response:
            return []
        try:
            return self._get_store().ingest_from_chat(user_message, assistant_response)
        except Exception as e:
            logger.debug("Memory store_turn failed: %s", e)
            return []

    def store(self, content: str, topic: str, source: str) -> bool:
        """
        Persist a single explicit fact (used by the future task layer).

        Args:
            content: the fact text.
            topic: knowledge topic label.
            source: provenance label, e.g. ``"task"`` or ``"manual"``.

        Returns:
            True when newly stored, False when duplicate or failed.
        """
        if not content or not content.strip():
            return False
        try:
            return self._get_store().add_fact(
                KnowledgeFact(content=content, topic=topic, source=source)
            )
        except Exception as e:
            logger.debug("Memory store failed: %s", e)
            return False

    def retrieve(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Semantic search for memory items relevant to ``query``.

        Args:
            query: the lookup text (typically the user message).
            limit: maximum number of results.

        Returns:
            List of fact dicts (content/topic/source/url/timestamp/importance/score).

        Side effects:
            - none; read-only.
        """
        try:
            return self._get_store().search(query, top_k=limit)
        except Exception as e:
            logger.debug("Memory retrieve failed: %s", e)
            return []

    def stats(self) -> Dict[str, Any]:
        """Return provider statistics (total facts, topics)."""
        try:
            return self._get_store().stats()
        except Exception as e:
            logger.debug("Memory stats failed: %s", e)
            return {}

    def list_all(self, limit: int) -> List[Dict[str, Any]]:
        """
        Return stored memory items (most recent first).

        Args:
            limit: maximum number of items to return.

        Returns:
            List of fact dicts (content/topic/source/url/timestamp/importance).

        Side effects:
            - none; read-only.
        """
        try:
            return self._get_store().list_all(top_k=limit)
        except Exception as e:
            logger.debug("Memory list failed: %s", e)
            return []

    def clear(self) -> int:
        """
        Remove every stored item.

        Returns:
            Number of items removed.

        Side effects:
            - wipes the underlying knowledge store.
        """
        try:
            return int(self._get_store().clear_all())
        except Exception as e:
            logger.debug("Memory clear failed: %s", e)
            return 0

    def delete(self, ids: List[str]) -> int:
        """
        Remove specific stored items by entry id.

        Args:
            ids: vector-store entry ids to delete.

        Returns:
            Number of items actually removed.

        Side effects:
            - removes the matching facts from the knowledge store (persisted).
        """
        if not ids:
            return 0
        removed = 0
        store = self._get_store()
        for item_id in ids:
            try:
                if store.delete_by_id(item_id):
                    removed += 1
            except Exception as e:
                logger.debug("Memory delete %s failed: %s", item_id, e)
        return removed

    def update(self, item_id: str, content: str,
               topic: Optional[str] = None,
               importance: Optional[float] = None) -> bool:
        """
        Edit an existing item's text (and optionally its topic/importance).

        Args:
            item_id: vector-store entry id of the item to edit.
            content: new fact text.
            topic: optional new topic label; ``None`` keeps the existing one.
            importance: optional importance score in [0, 1] (clamped);
                ``None`` keeps the existing one.

        Returns:
            True when the item was updated, False when the id is unknown,
            the text is empty, or it duplicates another stored fact.

        Side effects:
            - replaces the fact's text/embedding in the knowledge store.
        """
        if not content or not content.strip():
            return False
        try:
            return self._get_store().update_fact(
                item_id, content.strip(), topic=topic, importance=importance)
        except Exception as e:
            logger.debug("Memory update %s failed: %s", item_id, e)
            return False
