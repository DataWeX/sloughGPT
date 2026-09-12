"""logging — OOP logger hierarchy for all interfaces.

Public API:
    LogLevel, LogRecord, Logger, ChildLogger, TaggedLogger, CompositeLogger
    ErrorCode, LogTag, ConsoleLogger, CLILogger, ShellLogger, WebLogger
    BridgeHandler, setup_logging, get_request_id, set_request_id
    get_logger, set_global, get_global
"""

from __future__ import annotations

from typing import Optional

from domain.logging._internal.base import (
    Logger, LogLevel, LogRecord, ChildLogger, TaggedLogger,
    CompositeLogger, ErrorCode, LogTag,
)
from domain.logging._internal.console_logger import ConsoleLogger
from domain.logging._internal.cli_logger import CLILogger
from domain.logging._internal.shell_logger import ShellLogger
from domain.logging._internal.web_logger import WebLogger
from domain.logging._internal.bridge import BridgeHandler
from domain.logging._internal.config import (
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

_global_logger: Optional[Logger] = None


def set_global(logger: Logger) -> None:
    global _global_logger
    _global_logger = logger


def get_global() -> Logger:
    global _global_logger
    if _global_logger is None:
        _global_logger = ConsoleLogger("slo")
    return _global_logger


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
    cls = _INTERFACE_MAP.get(interface.lower())
    if cls is None:
        valid = ", ".join(sorted(set(_INTERFACE_MAP.keys())))
        raise ValueError(f"Unknown logger interface {interface!r}. Choose from: {valid}")
    return cls(name=name, level=level, **kwargs)
