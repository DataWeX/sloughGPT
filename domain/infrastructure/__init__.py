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
from domain.infrastructure._internal.entity_repositories import (
    get_dataset_repository,
    get_knowledge_repository,
)
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
]
