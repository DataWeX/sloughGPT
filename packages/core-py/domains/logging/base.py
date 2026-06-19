"""
Base logging interfaces — LogLevel, LogRecord, and Logger ABC.

Every interface (API server, CLI, shell REPL, web) inherits from Logger
and implements ``emit()`` to route output to its native destination.

Usage::

    from domains.logging import Logger, LogLevel, LogRecord

    class MyLogger(Logger):
        def emit(self, record: LogRecord) -> None:
            print(f"[{record.level.value}] {record.message}")
"""

from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class LogLevel(Enum):
    """Standard log levels, ordered by severity."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, LogLevel):
            return NotImplemented
        _order = {self.DEBUG: 0, self.INFO: 1, self.WARNING: 2, self.ERROR: 3, self.CRITICAL: 4}
        return _order[self] >= _order[other]

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, LogLevel):
            return NotImplemented
        _order = {self.DEBUG: 0, self.INFO: 1, self.WARNING: 2, self.ERROR: 3, self.CRITICAL: 4}
        return _order[self] > _order[other]

    def __le__(self, other: object) -> bool:
        if not isinstance(other, LogLevel):
            return NotImplemented
        return not self.__gt__(other)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, LogLevel):
            return NotImplemented
        return not self.__ge__(other)


@dataclass(frozen=True)
class LogRecord:
    """A single log event — immutable, passed from caller to emitter.

    Attributes:
        level:    Severity of the message.
        message:  Human-readable text.
        logger:   Logical logger name (e.g. ``"man.api.inference"``).
        timestamp: Unix timestamp (seconds) of when the record was created.
        context:  Arbitrary key-value metadata (request_id, model name, etc.).
        exception: Captured exception info as ``"Type: message"`` or ``None``.
    """
    level: LogLevel
    message: str
    logger: str = "man"
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None


class Logger(ABC):
    """Abstract base for all interface-specific loggers.

    Subclass and implement ``emit()`` to route records to your output
    (terminal colors, Rich console, browser console, structured JSON, etc.).

    The convenience methods (``debug``, ``info``, ``warning``, ``error``,
    ``critical``) build a ``LogRecord`` and call ``emit()``.  Override them
    only if you need a different call signature.

    Parameters:
        name:     Logger name, typically ``"man.<domain>"``.
        level:    Minimum level to emit (below this is silently dropped).
        context:  Default context attached to every record.
    """

    def __init__(
        self,
        name: str = "man",
        level: LogLevel = LogLevel.INFO,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._name = name
        self._level = level
        self._context = dict(context) if context else {}
        self._lock = threading.Lock()

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def level(self) -> LogLevel:
        return self._level

    @level.setter
    def level(self, value: LogLevel) -> None:
        self._level = value

    @property
    def context(self) -> Dict[str, Any]:
        return self._context

    def set_context(self, **kwargs: Any) -> None:
        """Merge key-value pairs into the default context."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        self._context.clear()

    # ── Abstract ────────────────────────────────────────────────────────

    @abstractmethod
    def emit(self, record: LogRecord) -> None:
        """Write a log record to this interface's output.

        Must be thread-safe — ``emit`` may be called from any thread.
        """

    # ── Convenience ─────────────────────────────────────────────────────

    def _make_record(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[str] = None,
    ) -> LogRecord:
        merged = {**self._context, **(context or {})}
        return LogRecord(
            level=level,
            message=message,
            logger=self._name,
            context=merged,
            exception=exception,
        )

    def _should_emit(self, level: LogLevel) -> bool:
        return level >= self._level

    def debug(self, msg: str, **ctx: Any) -> None:
        if self._should_emit(LogLevel.DEBUG):
            self.emit(self._make_record(LogLevel.DEBUG, msg, ctx))

    def info(self, msg: str, **ctx: Any) -> None:
        if self._should_emit(LogLevel.INFO):
            self.emit(self._make_record(LogLevel.INFO, msg, ctx))

    def warning(self, msg: str, **ctx: Any) -> None:
        if self._should_emit(LogLevel.WARNING):
            self.emit(self._make_record(LogLevel.WARNING, msg, ctx))

    def error(self, msg: str, exception: Optional[str] = None, **ctx: Any) -> None:
        if self._should_emit(LogLevel.ERROR):
            self.emit(self._make_record(LogLevel.ERROR, msg, ctx, exception=exception))

    def critical(self, msg: str, exception: Optional[str] = None, **ctx: Any) -> None:
        if self._should_emit(LogLevel.CRITICAL):
            self.emit(self._make_record(LogLevel.CRITICAL, msg, ctx, exception=exception))

    def exception(self, msg: str, exc: BaseException, **ctx: Any) -> None:
        """Log an error with the exception's string representation."""
        exc_str = f"{type(exc).__name__}: {exc}"
        self.error(msg, exception=exc_str, **ctx)

    # ── Child loggers ───────────────────────────────────────────────────

    def child(self, suffix: str, **ctx: Any) -> ChildLogger:
        """Create a child logger with ``name = self.name.suffix``.

        The child inherits the parent's level and emits through the
        parent's ``emit()``, so output formatting is shared.

        Example::

            log = ConsoleLogger("man.api")
            inference_log = log.child("inference")
            inference_log.info("generating")  # [api.inference] generating
        """
        child_logger = ChildLogger(
            name=f"{self._name}.{suffix}",
            parent=self,
            context={**self._context, **ctx},
        )
        return child_logger

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self._name!r} level={self._level.value}>"


class ChildLogger(Logger):
    """A child logger that delegates ``emit()`` to its parent.

    Created via ``Logger.child()``.  Shares the parent's ``emit()``
    method so output formatting is consistent.
    """

    def __init__(
        self,
        name: str,
        parent: Logger,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, level=parent.level, context=context)
        self._parent = parent

    @property
    def level(self) -> LogLevel:
        return self._parent.level

    @level.setter
    def level(self, value: LogLevel) -> None:
        self._parent.level = value

    def emit(self, record: LogRecord) -> None:
        self._parent.emit(record)
