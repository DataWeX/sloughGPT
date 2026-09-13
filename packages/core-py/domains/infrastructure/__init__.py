"""Backward-compatibility shim — imports from the new ``domain.infrastructure`` package."""

try:
    from domains.infrastructure.config import (
        AppConfig,
        get_config,
        reload_config,
        ConfigManager,
        get_config_manager,
        set_config_manager,
    )
    from domains.infrastructure.event_bus import (
        EventBus,
        get_event_bus,
    )
    from domains.infrastructure.lifecycle import (
        LifecycleManager,
        LifecyclePhase,
    )
    from domains.infrastructure.errors import (
        AppError,
        ErrorCode,
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
