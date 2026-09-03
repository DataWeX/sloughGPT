"""Facade that makes long-term memory a first-class service.

Layering (kept producer-agnostic so the task-execution layer can reuse it):

    chat loop / task executor        (producers)
      -> MemoryService               (this module: config + gating)
      -> MemoryProvider              (storage seam)
      -> KnowledgeMemory             (concrete store)

Nothing here knows about HTTP, chat schemas, or tasks. ``remember()`` is the
turn-saver the chat loop was missing; ``retrieve()`` feeds the existing
knowledge-enrichment attach path; ``store()`` is available to any producer
(e.g. the future persistent-task layer) for explicit fact writes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from domains.memory.memory_config import MemoryConfig
from domains.memory.memory_provider import KnowledgeMemoryProvider, MemoryProvider

logger = logging.getLogger(__name__)


class MemoryService:
    """Auto-memory entry point for any producer (chat loop, task runner, CLI).

    All public methods fail closed: exceptions are logged at debug level so a
    memory hiccup can never break a chat turn. Methods are synchronous;
    ``remember_async`` offloads ``remember`` onto ``asyncio.to_thread`` unless
    ``config.sync_remember`` requests inline execution.
    """

    def __init__(self, provider: Optional[MemoryProvider] = None,
                 config: Optional[MemoryConfig] = None):
        """
        Args:
            provider: storage adapter; defaults to KnowledgeMemoryProvider.
            config: runtime config; defaults to the MemoryConfig singleton.
        """
        self._provider: MemoryProvider = provider or KnowledgeMemoryProvider()
        self._config = config or MemoryConfig.get()

    @property
    def enabled(self) -> bool:
        """Whether the memory layer is active."""
        return self._config.enabled

    def set_enabled(self, enabled: bool) -> None:
        """
        Toggle the memory master switch at runtime.

        Args:
            enabled: ``True`` enables the memory layer; ``False`` no-ops every
                subsequent method (store/remember/retrieve/list/clear/delete).

        Side effects:
            - updates the shared ``MemoryConfig`` singleton.
        """
        self._config.set_enabled(enabled)

    def set_archive_retention(self, days: float) -> None:
        """
        Override the archive retention window at runtime.

        Args:
            days: retention window in days for ``prune_archive()``; ``0``
                prunes everything.

        Side effects:
            - updates the shared ``MemoryConfig`` singleton.
        """
        self._config.set_archive_retention_days(days)

    def config_snapshot(self) -> dict:
        """
        Return the current runtime memory settings.

        Returns:
            dict: keys ``enabled``, ``min_chars``, ``max_facts``,
                ``store_path``, ``sync_remember``,
                ``consolidation_threshold``, ``maintenance_interval_minutes``,
                ``archive_retention_days``.

        Side effects:
            - none; read-only.
        """
        return self._config.snapshot()

    def remember(self, user_message: str, assistant_response: str) -> bool:
        """
        Silently persist one completed turn as durable memory.

        Args:
            user_message: the user's prompt/instruction text.
            assistant_response: the assistant's reply to mine facts from.

        Returns:
            True when at least one new fact was stored; False when skipped
            (disabled, empty, too short, or nothing new learned).

        Side effects:
            - writes extracted facts into the underlying knowledge store.
        """
        if not self.enabled:
            return False
        combined = (user_message or "") + (assistant_response or "")
        if len(combined.strip()) < self._config.min_chars:
            return False
        return self._provider.store_turn(user_message, assistant_response)

    async def remember_async(self, user_message: str, assistant_response: str) -> bool:
        """
        Non-blocking variant of ``remember`` for async producers.

        The synchronous store call runs on a worker thread unless
        ``config.sync_remember`` is set (tests and task runners want the
        result inline without event-loop gymnastics).

        Args:
            user_message: the user's prompt/instruction text.
            assistant_response: the assistant's reply to mine facts from.

        Returns:
            True when at least one new fact was stored; False when skipped.

        Side effects:
            - same as ``remember``; extraction runs on a worker thread.
        """
        if self._config.sync_remember:
            return self.remember(user_message, assistant_response)
        return await asyncio.to_thread(self.remember, user_message, assistant_response)

    def remember_facts(self, user_message: str, assistant_response: str) -> List[str]:
        """
        Silently persist one completed turn and return the newly stored facts.

        Args:
            user_message: the user's prompt/instruction text.
            assistant_response: the assistant's reply to mine facts from.

        Returns:
            List of the newly stored fact texts; empty when skipped (disabled,
            empty, too short, or nothing new learned).

        Side effects:
            - writes extracted facts into the underlying knowledge store.
        """
        if not self.enabled:
            return []
        combined = (user_message or "") + (assistant_response or "")
        if len(combined.strip()) < self._config.min_chars:
            return []
        return self._provider.store_turn_facts(user_message, assistant_response)

    async def remember_facts_async(self, user_message: str, assistant_response: str) -> List[str]:
        """
        Non-blocking variant of ``remember_facts`` for async producers.

        The synchronous store call runs on a worker thread unless
        ``config.sync_remember`` is set (tests and task runners want the
        result inline without event-loop gymnastics).

        Args:
            user_message: the user's prompt/instruction text.
            assistant_response: the assistant's reply to mine facts from.

        Returns:
            List of the newly stored fact texts; empty when nothing was stored.

        Side effects:
            - same as ``remember_facts``; extraction runs on a worker thread.
        """
        if self._config.sync_remember:
            return self.remember_facts(user_message, assistant_response)
        return await asyncio.to_thread(self.remember_facts, user_message, assistant_response)

    def retrieve(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return memory items relevant to ``query``.

        Args:
            query: the lookup text (typically the user message).
            limit: max results; defaults to ``config.max_facts``.

        Returns:
            List of fact dicts; empty when disabled or ``query`` is blank.

        Side effects:
            - none; read-only.
        """
        if not self.enabled or not query:
            return []
        return self._provider.retrieve(query, limit or self._config.max_facts)

    def store(self, content: str, topic: str, source: str) -> bool:
        """
        Persist a single explicit fact (used by the task layer).

        Args:
            content: the fact text.
            topic: knowledge topic label.
            source: provenance label, e.g. ``"task"``.

        Returns:
            True when newly stored, False when disabled/duplicate/failed.

        Side effects:
            - writes the fact into the underlying knowledge store.
        """
        if not self.enabled:
            return False
        return self._provider.store(content, topic, source)

    def stats(self) -> Dict[str, Any]:
        """Return memory statistics (total facts, topics)."""
        return self._provider.stats()

    def list_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Return stored memory items (most recent first).

        Args:
            limit: maximum number of items to return.

        Returns:
            List of fact dicts; empty when disabled.

        Side effects:
            - none; read-only.
        """
        if not self.enabled:
            return []
        return self._provider.list_all(limit)

    def clear(self) -> int:
        """
        Remove every stored item.

        Returns:
            Number of items removed; 0 when disabled.

        Side effects:
            - wipes the underlying knowledge store.
        """
        if not self.enabled:
            return 0
        return self._provider.clear()

    def delete(self, ids: List[str]) -> int:
        """
        Remove specific stored items by entry id.

        Args:
            ids: entry ids to delete (from ``list_all``/``retrieve``).

        Returns:
            Number of items actually removed; 0 when disabled.

        Side effects:
            - removes the matching facts from the underlying knowledge store.
        """
        if not self.enabled:
            return 0
        return self._provider.delete(ids)

    def update(self, item_id: str, content: str,
               topic: Optional[str] = None,
               importance: Optional[float] = None) -> bool:
        """
        Edit a stored item's text (and optionally its topic/importance).

        Args:
            item_id: entry id of the item to edit.
            content: new fact text.
            topic: optional new topic label; ``None`` keeps the existing one.
            importance: optional importance score in [0, 1] (clamped);
                ``None`` keeps the existing one.

        Returns:
            True when the item was updated; False when disabled, the id is
            unknown, or the new text duplicates another stored fact.

        Side effects:
            - replaces the fact's text/embedding in the knowledge store.
        """
        if not self.enabled:
            return False
        return self._provider.update(item_id, content, topic=topic,
                                     importance=importance)


_service: Optional[MemoryService] = None


def get_memory_service(provider: Optional[MemoryProvider] = None,
                       config: Optional[MemoryConfig] = None) -> MemoryService:
    """Return the process-wide MemoryService singleton.

    Created once with the production provider/config; later calls return the
    existing instance regardless of arguments.
    """
    global _service
    if _service is None:
        _service = MemoryService(provider=provider, config=config)
    return _service
