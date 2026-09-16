"""Backward-compatibility shim — imports from the new ``domain.infrastructure`` package."""

try:
    from domain.infrastructure._internal.config import (
        AppConfig,
        ConfigManager,
        get_config,
        get_config_manager,
        reload_config,
        set_config_manager,
    )
    from domain.infrastructure._internal.errors import (
        AppError,
        ErrorCode,
    )
    from domain.infrastructure._internal.event_bus import (
        EventBus,
        get_event_bus,
    )
    from domain.infrastructure._internal.lifecycle import (
        LifecycleManager,
        LifecyclePhase,
    )
except (ImportError, AttributeError):
    AppConfig = None
    get_config = None
    reload_config = None
    ConfigManager = None
    get_config_manager = None
    set_config_manager = None
    EventBus = None
    get_event_bus = None
    LifecycleManager = None
    LifecyclePhase = None
    AppError = Exception
    ErrorCode = None

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
