"""Memory layer - chat- and task-agnostic auto-memory facade.

Layers (kept modular so the future task-execution layer can plug in):
    producer (chat loop / task executor)
      -> MemoryService.remember() / retrieve() / store()
      -> MemoryProvider (storage seam)
      -> KnowledgeMemory (concrete zero-dependency store)

Nothing in this package knows about HTTP, chat schemas, or tasks.
"""

from domains.memory.memory_config import MemoryConfig
from domains.memory.memory_provider import KnowledgeMemoryProvider, MemoryProvider
from domains.memory.memory_service import MemoryService, get_memory_service
from domains.memory.maintenance import (
    maintenance_tick,
    start_memory_maintenance,
    stop_memory_maintenance,
)
from domains.memory.task_memory import (
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

__all__ = [
    "MemoryConfig",
    "MemoryProvider",
    "KnowledgeMemoryProvider",
    "MemoryService",
    "get_memory_service",
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
