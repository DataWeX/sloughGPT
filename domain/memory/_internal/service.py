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
import os
import time
from typing import Any

from domain.memory._internal import memory_card
from domain.memory._internal.config import MemoryConfig
from domain.memory._internal.provider import KnowledgeMemoryProvider, MemoryProvider

logger = logging.getLogger(__name__)

ACTIVE_CARD_ENV = "SLO_MEMORY_ACTIVE_CARD"
_CARD_FETCH_LIMIT = 1_000_000


class MemoryService:
    """Auto-memory entry point for any producer (chat loop, task runner, CLI).

    All public methods fail closed: exceptions are logged at debug level so a
    memory hiccup can never break a chat turn. Methods are synchronous;
    ``remember_async`` offloads ``remember`` onto ``asyncio.to_thread`` unless
    ``config.sync_remember`` requests inline execution.
    """

    def __init__(self, provider: MemoryProvider | None = None, config: MemoryConfig | None = None):
        """
        Args:
            provider: storage adapter; defaults to KnowledgeMemoryProvider.
            config: runtime config; defaults to the MemoryConfig singleton.
        """
        self._provider: MemoryProvider = provider or KnowledgeMemoryProvider()
        self._config = config or MemoryConfig.get()
        self._active_card_loaded = False

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

    def remember_facts(self, user_message: str, assistant_response: str) -> list[str]:
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

    async def remember_facts_async(self, user_message: str, assistant_response: str) -> list[str]:
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

    def retrieve(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
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

    def stats(self) -> dict[str, Any]:
        """Return memory statistics (total facts, topics)."""
        return self._provider.stats()

    def list_all(self, limit: int = 100) -> list[dict[str, Any]]:
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

    def delete(self, ids: list[str]) -> int:
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

    def update(
        self, item_id: str, content: str, topic: str | None = None, importance: float | None = None
    ) -> bool:
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
        return self._provider.update(item_id, content, topic=topic, importance=importance)

    # ── loadable memory cards ───────────────────────────────────────────────

    def save_card(
        self, name: str | None = None, overwrite: bool = False, cards_dir: str | None = None
    ) -> dict[str, Any]:
        """
        Snapshot current long-term memory into a portable card file.

        Args:
            name: card name; ``None`` auto-generates a unique ``card-<ts>``.
            overwrite: allow replacing an existing card of the same name.
            cards_dir: override cards directory (default
                ``SLO_MEMORY_CARDS_DIR`` or ``data/memory_cards``).

        Returns:
            dict: ``ok=True`` with ``name``/``facts_count``/``created_at``/
            ``size_bytes``, or ``ok=False`` with ``error``/``error_code``
            (``disabled``/``exists``/``invalid_name``).

        Side effects:
            - writes one atomic card file; memory store untouched.
        """
        if not self.enabled:
            return {"ok": False, "error": "memory is disabled", "error_code": "disabled"}
        facts = self._provider.list_all(limit=_CARD_FETCH_LIMIT)
        try:
            info = memory_card.save_card(facts, name, cards_dir=cards_dir, overwrite=overwrite)
        except memory_card.MemoryCardError as e:
            return {"ok": False, "error": str(e), "error_code": e.error_code}
        return {
            "ok": True,
            "name": info.name,
            "facts_count": info.facts_count,
            "created_at": info.created_at,
            "size_bytes": info.size_bytes,
        }

    def load_card(
        self, name: str, mode: str = "replace", cards_dir: str | None = None
    ) -> dict[str, Any]:
        """
        Load a card back into long-term memory.

        The card is fully read and checksum-verified BEFORE any store
        mutation. ``mode="replace"`` first autosaves current memory to an
        ``autosave-<ts>`` card and aborts (store untouched) if that backup
        cannot be written; ``mode="merge"`` imports alongside existing facts
        (content-hash dedup makes it idempotent).

        Args:
            name: card to load.
            mode: ``"replace"`` (default) or ``"merge"``.
            cards_dir: override cards directory.

        Returns:
            dict: ``ok=True`` with ``name``/``mode``/``facts_count``/
            ``imported``/``skipped``/``backup`` (replace only) or ``ok=False``
            with ``error``/``error_code`` (``disabled``/``not_found``/
            ``corrupt``/``invalid_mode``/``backup_failed``).

        Side effects:
            - replace: autosave card written, store cleared, facts re-imported.
            - merge: facts imported (duplicates skipped).
        """
        if not self.enabled:
            return {"ok": False, "error": "memory is disabled", "error_code": "disabled"}
        if mode not in ("replace", "merge"):
            return {
                "ok": False,
                "error": f"invalid mode {mode!r} (expected replace|merge)",
                "error_code": "invalid_mode",
            }
        try:
            facts = memory_card.load_card(name, cards_dir=cards_dir)  # verify FIRST
        except memory_card.MemoryCardError as e:
            return {"ok": False, "error": str(e), "error_code": e.error_code}

        backup: str | None = None
        replaced = 0
        if mode == "replace":
            current = self._provider.list_all(limit=_CARD_FETCH_LIMIT)
            if current:
                try:
                    autosave_name = (
                        "autosave-"
                        + time.strftime("%Y%m%d-%H%M%S", time.localtime())
                        + f"-{time.time_ns() % 1_000_000:06d}"
                    )
                    backup_info = memory_card.save_card(current, autosave_name, cards_dir=cards_dir)
                    backup = backup_info.name
                except Exception as e:
                    logger.warning("memory card load aborted, autosave failed: %s", e)
                    return {
                        "ok": False,
                        "error": f"autosave backup failed, load aborted: {e}",
                        "error_code": "backup_failed",
                    }
            replaced = self._provider.clear()

        imported, skipped = self._provider.store_facts(facts)
        logger.info(
            "memory card loaded: %s mode=%s imported=%d skipped=%d backup=%s",
            name,
            mode,
            imported,
            skipped,
            backup,
        )
        return {
            "ok": True,
            "name": name,
            "mode": mode,
            "facts_count": len(facts),
            "imported": imported,
            "skipped": skipped,
            "replaced": replaced,
            "backup": backup,
        }

    def list_cards(self, cards_dir: str | None = None) -> dict[str, Any]:
        """
        List saved memory cards, newest first.

        Available even while memory is disabled (file management only).

        Returns:
            dict: ``ok=True`` with ``cards`` — list of CardInfo dicts
            (``name``/``created_at``/``facts_count``/``size_bytes``/``valid``).

        Side effects:
            - none; read-only.
        """
        try:
            infos = memory_card.list_cards(cards_dir=cards_dir)
        except Exception as e:
            logger.warning("memory card listing failed: %s", e)
            return {"ok": False, "error": str(e), "error_code": "card_error"}
        return {
            "ok": True,
            "cards": [
                {
                    "name": i.name,
                    "created_at": i.created_at,
                    "facts_count": i.facts_count,
                    "size_bytes": i.size_bytes,
                    "valid": i.valid,
                }
                for i in infos
            ],
        }

    def delete_card(self, name: str, cards_dir: str | None = None) -> dict[str, Any]:
        """
        Delete a saved memory card (memory store untouched).

        Args:
            name: card to delete.
            cards_dir: override cards directory.

        Returns:
            dict: ``ok=True`` with ``deleted`` (bool); ``ok=False`` with
            ``error``/``error_code`` for invalid names.

        Side effects:
            - removes the card file when present.
        """
        try:
            deleted = memory_card.delete_card(name, cards_dir=cards_dir)
        except memory_card.MemoryCardError as e:
            return {"ok": False, "error": str(e), "error_code": e.error_code}
        return {"ok": True, "deleted": deleted}

    def ensure_active_card(self) -> bool:
        """
        Load ``SLO_MEMORY_ACTIVE_CARD`` once per process — the model-load hook.

        Called after a model loads so the model wakes up remembering. No-op
        when the env var is unset, the card already loaded, memory is
        disabled, or loading fails (fail-closed: a broken card never breaks
        model load; failure is logged and retried on the next load).

        Returns:
            bool: True only when this call actually loaded the card.

        Side effects:
            - merge-loads the active card into long-term memory on success.
        """
        name = os.environ.get(ACTIVE_CARD_ENV, "").strip()
        if not name or self._active_card_loaded:
            return False
        result = self.load_card(name, mode="merge")
        if not result.get("ok"):
            logger.warning("active memory card %r not loaded: %s", name, result.get("error"))
            return False
        self._active_card_loaded = True
        logger.info(
            "active memory card loaded: %s (%d imported)",
            name,
            result.get("imported", 0),
        )
        return True


_service: MemoryService | None = None


def get_memory_service(
    provider: MemoryProvider | None = None, config: MemoryConfig | None = None
) -> MemoryService:
    """Return the process-wide MemoryService singleton.

    Created once with the production provider/config; later calls return the
    existing instance regardless of arguments.
    """
    global _service
    if _service is None:
        _service = MemoryService(provider=provider, config=config)
    return _service
