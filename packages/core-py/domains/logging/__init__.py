"""
Logging package — OOP logger hierarchy for all interfaces.

Classes:
    LogLevel         — DEBUG, INFO, WARNING, ERROR, CRITICAL
    LogRecord        — immutable log event dataclass
    Logger           — abstract base class (ABC)
    ChildLogger      — delegates emit() to parent
    TaggedLogger     — attaches type tags to every record
    CompositeLogger  — emits to multiple downstream loggers
    ConsoleLogger    — colored terminal output (API server)
    CLILogger        — native ANSI output (CLI commands)
    ShellLogger      — ANSI output (interactive REPL)
    WebLogger        — structured JSON (browser / SSR) + UI event tracking
    BridgeHandler    — routes Python logging.getLogger() through our Logger
    ErrorCode        — structured error codes (E_AUTH_*, E_MODEL_*, etc.)
    LogTag           — type tags (REQ, MODEL, AUTH, etc.)

Centralized config:
    setup_logging()       — configure all logging in one place (recommended)
    get_request_id()      — get current request correlation ID
    set_request_id()      — set current request correlation ID
    get_log_context()     — get current structured logging context
    set_log_context()     — merge fields into logging context

Factory:
    get_logger()     — create a logger by interface name
    set_global()     — set the global default logger
    get_global()     — get the global default logger
"""

from __future__ import annotations

from typing import Optional

from .base import Logger, LogLevel, LogRecord, ChildLogger, TaggedLogger, CompositeLogger, ErrorCode, LogTag
from .console_logger import ConsoleLogger
from .cli_logger import CLILogger
from .shell_logger import ShellLogger
from .web_logger import WebLogger
from .bridge import BridgeHandler
from .config import (
    setup_logging,
    get_request_id,
    set_request_id,
    get_log_context,
    set_log_context,
    clear_log_context,
)

WebEventLogger = WebLogger  # backward compat alias

__all__ = [
    "LogLevel",
    "LogRecord",
    "Logger",
    "ChildLogger",
    "TaggedLogger",
    "CompositeLogger",
    "ErrorCode",
    "LogTag",
    "ConsoleLogger",
    "CLILogger",
    "ShellLogger",
    "WebLogger",
    "WebEventLogger",
    "BridgeHandler",
    "setup_logging",
    "get_request_id",
    "set_request_id",
    "get_log_context",
    "set_log_context",
    "clear_log_context",
    "get_logger",
    "set_global",
    "get_global",
]

# ── Global singleton ───────────────────────────────────────────────────

_global_logger: Optional[Logger] = None


def set_global(logger: Logger) -> None:
    """Set the global default logger used by ``get_global()``."""
    global _global_logger
    _global_logger = logger


def get_global() -> Logger:
    """Get the global default logger.  Creates a ConsoleLogger if none set."""
    global _global_logger
    if _global_logger is None:
        _global_logger = ConsoleLogger("slo")
    return _global_logger


# ── Factory ────────────────────────────────────────────────────────────

_INTERFACE_MAP = {
    "api":     ConsoleLogger,
    "server":  ConsoleLogger,
    "console": ConsoleLogger,
    "cli":     CLILogger,
    "shell":   ShellLogger,
    "repl":    ShellLogger,
    "web":     WebLogger,
    "browser": WebLogger,
}


def get_logger(
    interface: str = "api",
    name: str = "slo",
    level: LogLevel = LogLevel.INFO,
    **kwargs,
) -> Logger:
    """Create a logger for the given interface.

    Parameters:
        interface: One of ``"api"``, ``"cli"``, ``"shell"``, ``"web"``
                   (or aliases: ``"server"``, ``"console"``, ``"repl"``,
                   ``"browser"``).
        name:      Logger name (e.g. ``"slo.api.inference"``).
        level:     Minimum log level.
        **kwargs:  Passed to the logger constructor.

    Returns:
        A Logger subclass instance for the requested interface.

    Raises:
        ValueError: If the interface name is unknown.
    """
    cls = _INTERFACE_MAP.get(interface.lower())
    if cls is None:
        valid = ", ".join(sorted(set(_INTERFACE_MAP.keys())))
        raise ValueError(f"Unknown logger interface {interface!r}. Choose from: {valid}")
    return cls(name=name, level=level, **kwargs)
