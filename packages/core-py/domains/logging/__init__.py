"""
Logging package — OOP logger hierarchy for all interfaces.

Classes:
    LogLevel         — DEBUG, INFO, WARNING, ERROR, CRITICAL
    LogRecord        — immutable log event dataclass
    Logger           — abstract base class (ABC)
    ChildLogger      — delegates emit() to parent
    TaggedLogger     — attaches type tags to every record
    ConsoleLogger    — colored terminal output (API server)
    CLILogger        — Rich-powered output (CLI commands)
    ShellLogger      — ANSI output (interactive REPL)
    WebLogger        — structured JSON (browser / SSR)
    BridgeHandler    — routes Python logging.getLogger() through our Logger
    ErrorCode        — structured error codes (E_AUTH_*, E_MODEL_*, etc.)
    LogTag           — type tags (REQ, MODEL, AUTH, etc.)

Factory:
    get_logger()     — create a logger by interface name
    set_global()     — set the global default logger
    get_global()     — get the global default logger
"""

from typing import Optional

from .base import Logger, LogLevel, LogRecord, ChildLogger, TaggedLogger, ErrorCode, LogTag
from .console_logger import ConsoleLogger
from .cli_logger import CLILogger
from .shell_logger import ShellLogger
from .web_logger import WebLogger
from .bridge import BridgeHandler

__all__ = [
    "LogLevel",
    "LogRecord",
    "Logger",
    "ChildLogger",
    "TaggedLogger",
    "ErrorCode",
    "LogTag",
    "ConsoleLogger",
    "CLILogger",
    "ShellLogger",
    "WebLogger",
    "BridgeHandler",
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
