"""Memory layer - chat- and task-agnostic auto-memory facade.

Layers (kept modular so the future task-execution layer can plug in):
    producer (chat loop / task executor)
      -> MemoryService.remember() / retrieve() / store()
      -> MemoryProvider (storage seam)
      -> KnowledgeMemory (concrete zero-dependency store)

Nothing in this package knows about HTTP, chat schemas, or tasks.

Usage:
    from memory import get_memory_service, MemoryConfig

    svc = get_memory_service()
    svc.remember("user msg", "assistant reply")
    facts = svc.retrieve("user msg")
"""

from __future__ import annotations

from memory._internal.config import MemoryConfig
from memory._internal.provider import KnowledgeMemoryProvider, MemoryProvider
from memory._internal.service import MemoryService, get_memory_service
from memory._internal.maintenance import (
    maintenance_tick,
    start_memory_maintenance,
    stop_memory_maintenance,
)
from memory._internal.task_memory import (
    TASK_CONSOLIDATE,
    TASK_REMEMBER,
    TASK_STORE,
    archive_stats,
    list_archive,
    prune_archive,
    register_memory_handlers,
    submit_memory_consolidate,
    submit_memory_remember,
    submit_memory_store,
    unregister_memory_handlers,
)
from memory._internal.consolidation import plan_consolidation

__all__ = [
    "MemoryConfig",
    "MemoryProvider",
    "KnowledgeMemoryProvider",
    "MemoryService",
    "get_memory_service",
    "plan_consolidation",
    "TASK_REMEMBER",
    "TASK_STORE",
    "TASK_CONSOLIDATE",
    "register_memory_handlers",
    "unregister_memory_handlers",
    "submit_memory_remember",
    "submit_memory_store",
    "submit_memory_consolidate",
    "maintenance_tick",
    "start_memory_maintenance",
    "stop_memory_maintenance",
    "list_archive",
    "archive_stats",
    "prune_archive",
]
