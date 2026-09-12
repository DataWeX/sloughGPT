"""Memory layer - DEPRECATED: import from `memory` package instead.

This module is a compatibility shim. All code should migrate to:
    from memory import get_memory_service, MemoryConfig, ...
"""

from __future__ import annotations

# Re-export everything from the new package for backward compatibility
from memory import (
    MemoryConfig,
    MemoryProvider,
    KnowledgeMemoryProvider,
    MemoryService,
    get_memory_service,
    plan_consolidation,
    TASK_REMEMBER,
    TASK_STORE,
    TASK_CONSOLIDATE,
    register_memory_handlers,
    unregister_memory_handlers,
    submit_memory_remember,
    submit_memory_store,
    submit_memory_consolidate,
    maintenance_tick,
    start_memory_maintenance,
    stop_memory_maintenance,
    list_archive,
    archive_stats,
    prune_archive,
)

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
