"""infrastructure — Core infrastructure (config, event bus, lifecycle, errors).

Public API:
    AppConfig, get_config, reload_config
    EventBus, get_event_bus
    LifecycleManager, LifecyclePhase
    AppError, ErrorCode
"""

from domain.infrastructure._internal.config import (
    AppConfig,
    get_config,
    reload_config,
    ConfigManager,
    get_config_manager,
    set_config_manager,
)
from domain.infrastructure._internal.event_bus import (
    EventBus,
    get_event_bus,
)
from domain.infrastructure._internal.lifecycle import (
    LifecycleManager,
    LifecyclePhase,
)
from domain.infrastructure._internal.errors import (
    AppError,
    ErrorCode,
)

__all__ = [
    "AppConfig",
    "get_config",
    "reload_config",
    "ConfigManager",
    "get_config_manager",
    "set_config_manager",
    "EventBus",
    "get_event_bus",
    "LifecycleManager",
    "LifecyclePhase",
    "AppError",
    "ErrorCode",
]
