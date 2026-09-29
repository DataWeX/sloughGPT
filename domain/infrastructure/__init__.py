"""infrastructure — Core infrastructure (config, event bus, lifecycle, errors).

Public API:
    AppConfig, get_config, reload_config
    EventBus, get_event_bus
    LifecycleManager, LifecyclePhase
    AppError, ErrorCode
    get_pool, get_model_registry, get_server_state
    get_knowledge_repository, get_dataset_repository
    get_event_buffer, get_server_buffer
"""

from domain.infrastructure._internal.config import (
    AppConfig,
    ConfigManager,
    get_config,
    get_config_manager,
    reload_config,
    set_config_manager,
)
from domain.infrastructure._internal.database import get_db
from domain.infrastructure._internal.entity_repositories import (
    get_dataset_repository,
    get_knowledge_repository,
)
from domain.infrastructure._internal.errors import (
    AppError,
    ErrorCode,
    classify_exception,
)
from domain.infrastructure._internal.event_buffer import get_event_buffer
from domain.infrastructure._internal.event_bus import (
    EventBus,
    get_event_bus,
)
from domain.infrastructure._internal.fire_and_forget import get_pool
from domain.infrastructure._internal.lifecycle import (
    LifecycleManager,
    LifecyclePhase,
    get_lifecycle_manager,
)
from domain.infrastructure._internal.model_registry import get_model_registry
from domain.infrastructure._internal.output_buffer import get_server_buffer
from domain.infrastructure._internal.server_state import get_server_state

# Lazy exports — heavy or optional modules (import on first attribute access).
_LAZY_INFRA = {
    "SLNCCompiler": ("domain.infrastructure._internal.slnc.compiler", "SLNCCompiler"),
    "try_register": ("domain.infrastructure._internal.artifact_registry", "try_register"),
}


def __getattr__(name):
    if name in _LAZY_INFRA:
        import importlib

        module_path, attr = _LAZY_INFRA[name]
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AppConfig",
    "get_config",
    "reload_config",
    "ConfigManager",
    "get_config_manager",
    "set_config_manager",
    "get_db",
    "EventBus",
    "get_event_bus",
    "LifecycleManager",
    "LifecyclePhase",
    "get_lifecycle_manager",
    "AppError",
    "ErrorCode",
    "classify_exception",
    "get_db",
    "get_lifecycle_manager",
    "get_knowledge_repository",
    "get_dataset_repository",
    "get_event_buffer",
    "get_server_buffer",
    "get_pool",
    "get_model_registry",
    "get_server_state",
    "SLNCCompiler",
    "try_register",
]
