"""Configuration for the auto-memory layer.

Behaviour is sourced from environment variables so the chat loop stays
declarative - enabling or tuning memory never requires a code change.

| Variable                | Default        | Meaning                                      |
|-------------------------|----------------|----------------------------------------------|
| ``SLO_MEMORY_ENABLED``  | ``true``       | Master switch; ``false`` no-ops every method |
| ``SLO_MEMORY_MIN_CHARS``| ``80``         | Min combined turn length worth remembering   |
| ``SLO_MEMORY_MAX_FACTS``| ``5``          | Max facts returned by ``retrieve()``         |
| ``SLO_MEMORY_STORE_PATH`` | ``data/memory`` | Task-backed store dir for ``*.jsonl`` provenance |
| ``SLO_MEMORY_SYNC``     | ``false``      | Remember synchronously (for tests/tasks)     |
| ``SLO_MEMORY_CONSOLIDATION_THRESHOLD`` | ``0.80`` | Min n-gram cosine for near-dup merge in ``memory.consolidate`` |
| ``SLO_MEMORY_MAINTENANCE_INTERVAL_MINUTES`` | ``60`` | How often the server schedules a ``memory.consolidate`` pass; ``0`` disables |
| ``SLO_MEMORY_ARCHIVE_RETENTION_DAYS`` | ``30`` | Default retention window for ``prune_archive`` / ``memory archive --prune-days`` |
"""

import os
import threading
from typing import Optional


class MemoryConfig:
    """Runtime configuration for the auto-memory layer.

    Attributes:
        enabled: master switch; when False the service no-ops everywhere.
        min_chars: minimum combined user+assistant length before a turn is
            stored (short small-talk turns are noise, not knowledge).
        max_facts: maximum facts returned by retrieve().
        store_path: directory used by task-backed stores.
        sync_remember: when True remember() stores inline; otherwise callers
            offload it to a worker (``asyncio.to_thread``).
        consolidation_threshold: min n-gram cosine similarity for two facts in
            the same topic to be treated as near-duplicates by the
            ``memory.consolidate`` task. The default ``0.80`` merges
            near-verbatim copies (measured ~0.845) while keeping genuine
            paraphrases (~0.586) and cross-topic facts distinct.
        maintenance_interval_minutes: how often the server enqueues a
            ``memory.consolidate`` maintenance pass; ``0`` disables it.
        archive_retention_days: default retention window (in days) applied by
            ``prune_archive()`` and the ``memory archive --prune-days`` flag.
    """

    DEFAULT_ENABLED = True
    DEFAULT_MIN_CHARS = 80
    DEFAULT_MAX_FACTS = 5
    DEFAULT_STORE_PATH = "data/memory"
    DEFAULT_CONSOLIDATION_THRESHOLD = 0.80
    DEFAULT_MAINTENANCE_INTERVAL_MINUTES = 60
    DEFAULT_ARCHIVE_RETENTION_DAYS = 30

    _instance: Optional["MemoryConfig"] = None
    _lock = threading.Lock()

    def __init__(self, **kwargs):
        self.enabled = bool(kwargs.get("enabled", self._from_bool("SLO_MEMORY_ENABLED", self.DEFAULT_ENABLED)))
        self.min_chars = int(kwargs.get("min_chars", os.environ.get("SLO_MEMORY_MIN_CHARS", self.DEFAULT_MIN_CHARS)))
        self.max_facts = int(kwargs.get("max_facts", os.environ.get("SLO_MEMORY_MAX_FACTS", self.DEFAULT_MAX_FACTS)))
        self.store_path = kwargs.get("store_path", os.environ.get("SLO_MEMORY_STORE_PATH", self.DEFAULT_STORE_PATH))
        self.sync_remember = bool(kwargs.get("sync_remember", self._from_bool("SLO_MEMORY_SYNC", False)))
        self.consolidation_threshold = float(kwargs.get(
            "consolidation_threshold",
            os.environ.get("SLO_MEMORY_CONSOLIDATION_THRESHOLD", self.DEFAULT_CONSOLIDATION_THRESHOLD),
        ))
        self.maintenance_interval_minutes = float(kwargs.get(
            "maintenance_interval_minutes",
            os.environ.get("SLO_MEMORY_MAINTENANCE_INTERVAL_MINUTES", self.DEFAULT_MAINTENANCE_INTERVAL_MINUTES),
        ))
        self.archive_retention_days = float(kwargs.get(
            "archive_retention_days",
            os.environ.get("SLO_MEMORY_ARCHIVE_RETENTION_DAYS", self.DEFAULT_ARCHIVE_RETENTION_DAYS),
        ))

    @staticmethod
    def _from_bool(name: str, default: bool) -> bool:
        """Parse an env var as a boolean, returning ``default`` when unset."""
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in ("1", "true", "yes", "on")

    @classmethod
    def get(cls) -> "MemoryConfig":
        """Return the process-wide MemoryConfig singleton."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def set_enabled(self, enabled: bool) -> None:
        """
        Toggle the memory master switch at runtime.

        Args:
            enabled: ``True`` enables storing/retrieving memory; ``False``
                no-ops every service method.

        Side effects:
            - mutates the shared singleton; persists for the process lifetime
              (does not rewrite ``SLO_MEMORY_ENABLED`` in the environment).
        """
        with self._lock:
            self.enabled = bool(enabled)

    def set_archive_retention_days(self, days: float) -> None:
        """
        Override the archive retention window at runtime.

        Args:
            days: retention window in days applied by ``prune_archive()``
                when no explicit ``retain_days`` is passed. ``0`` prunes
                everything.

        Side effects:
            - mutates the shared singleton; persists for the process lifetime
              (does not rewrite ``SLO_MEMORY_ARCHIVE_RETENTION_DAYS``).
        """
        with self._lock:
            self.archive_retention_days = max(0.0, float(days))

    def snapshot(self) -> dict:
        """
        Return the current runtime settings as a plain dict.

        Returns:
            dict: keys ``enabled``, ``min_chars``, ``max_facts``,
                ``store_path``, ``sync_remember``,
                ``consolidation_threshold``, ``maintenance_interval_minutes``,
                ``archive_retention_days``.

        Side effects:
            - none; read-only.
        """
        with self._lock:
            return {
                "enabled": self.enabled,
                "min_chars": self.min_chars,
                "max_facts": self.max_facts,
                "store_path": self.store_path,
                "sync_remember": self.sync_remember,
                "consolidation_threshold": self.consolidation_threshold,
                "maintenance_interval_minutes": self.maintenance_interval_minutes,
                "archive_retention_days": self.archive_retention_days,
            }
