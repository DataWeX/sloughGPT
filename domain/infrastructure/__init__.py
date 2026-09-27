"""infrastructure — Core infrastructure (config, event bus, lifecycle, errors).

Public API:
    AppConfig, get_config, reload_config
    EventBus, get_event_bus
    LifecycleManager, LifecyclePhase
    AppError, ErrorCode
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
from domain.infrastructure._internal.errors import (
    AppError,
    ErrorCode,
    classify_exception,
)
from domain.infrastructure._internal.event_bus import (
    EventBus,
    get_event_bus,
)
from domain.infrastructure._internal.lifecycle import (
    LifecycleManager,
    LifecyclePhase,
    get_lifecycle_manager,
)

_LAZY_IMPORTS = {
    "get_event_buffer": ("._internal.event_buffer", "get_event_buffer"),
    "get_server_buffer": ("._internal.output_buffer", "get_server_buffer"),
    "DatasetRepository": ("._internal.entity_repositories", "DatasetRepository"),
    "KnowledgeRepository": ("._internal.entity_repositories", "KnowledgeRepository"),
    "artifact_registry": ("._internal.artifact_registry", None),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, package=__name__)
        return mod if attr is None else getattr(mod, attr)
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
    "get_event_buffer",
    "get_server_buffer",
    "DatasetRepository",
    "KnowledgeRepository",
    "artifact_registry",
]
