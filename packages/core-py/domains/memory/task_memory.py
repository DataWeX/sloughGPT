"""Backward-compatibility — allows ``from domain.memory._internal.task_memory import ...``."""
from domain.memory._internal.task_memory import (
    TASK_CONSOLIDATE,
    TASK_REMEMBER,
    TASK_STORE,
    archive_stats,
    list_archive,
    prune_archive,
    register_memory_handlers,
    unregister_memory_handlers,
)

__all__ = [
    "TASK_CONSOLIDATE",
    "TASK_REMEMBER",
    "TASK_STORE",
    "archive_stats",
    "list_archive",
    "prune_archive",
    "register_memory_handlers",
    "unregister_memory_handlers",
]