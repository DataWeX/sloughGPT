"""infrastructure — Core infrastructure (config, event bus, lifecycle, errors).

Public API:
    AppConfig, get_config, reload_config
    EventBus, get_event_bus
    LifecycleManager, LifecyclePhase
    AppError, ErrorCode, ERROR_REGISTRY, get_error_class,
    all error subclasses (ValidationError, AuthError, NotFoundError, ...)
    get_pool, get_model_registry, get_server_state
    get_knowledge_repository, get_dataset_repository
    get_event_buffer, get_server_buffer
"""

from domain.infrastructure._internal import errors as _errors_mod
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
    ERROR_REGISTRY,
    AppError,
    AuthError,
    ConfigError,
    ConflictError,
    ErrorCode,
    FatalError,
    ModelError,
    ModelOOMError,
    ModelTimeoutError,
    NotFoundError,
    NotImplementedAppError,
    RecoverableError,
    ResourceExhaustedError,
    TaskError,
    TimeoutAppError,
    ValidationError,
    classify_exception,
    emit_error_event,
    get_error_info,
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


def get_error_class(class_name: str) -> type[AppError]:
    """Resolve an error class by registry class-name (falls back to AppError)."""
    return getattr(_errors_mod, class_name, AppError)


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
    "ERROR_REGISTRY",
    "classify_exception",
    "get_error_class",
    "get_error_info",
    "emit_error_event",
    "RecoverableError",
    "FatalError",
    "ValidationError",
    "ConfigError",
    "ModelError",
    "ModelOOMError",
    "ModelTimeoutError",
    "TaskError",
    "ResourceExhaustedError",
    "NotFoundError",
    "AuthError",
    "ConflictError",
    "TimeoutAppError",
    "NotImplementedAppError",
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
