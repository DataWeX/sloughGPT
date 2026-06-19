"""
BridgeHandler — routes Python ``logging`` calls through a ``Logger`` instance.

Allows the rest of the codebase (which uses ``logging.getLogger("man.xxx")``)
to output through the new OOP logger hierarchy without changing every call site.

Usage::

    import logging
    from domains.logging import ConsoleLogger, BridgeHandler

    log = ConsoleLogger("man", level=LogLevel.DEBUG)
    handler = BridgeHandler(log)
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG)

    # Now all logging.getLogger("man.xxx").info(...) routes through ConsoleLogger
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import Logger, LogLevel


# ── Standard logging → our LogLevel mapping ────────────────────────────

_LEVEL_MAP = {
    logging.DEBUG:    LogLevel.DEBUG,
    logging.INFO:     LogLevel.INFO,
    logging.WARNING:  LogLevel.WARNING,
    logging.ERROR:    LogLevel.ERROR,
    logging.CRITICAL: LogLevel.CRITICAL,
}


class BridgeHandler(logging.Handler):
    """A ``logging.Handler`` that delegates to a ``Logger`` instance.

    This lets any module using ``logging.getLogger("man.xxx")`` output
    through the new OOP logger hierarchy.

    Parameters:
        logger: The ``Logger`` instance to delegate to.
    """

    def __init__(self, logger: Logger) -> None:
        super().__init__()
        self._logger = logger

    def emit(self, record: logging.LogRecord) -> None:
        """Convert a standard ``logging.LogRecord`` and emit via our Logger."""
        level = _LEVEL_MAP.get(record.levelno, LogLevel.INFO)

        # Build context from standard logging attributes
        ctx = {}
        if hasattr(record, "pathname"):
            ctx["path"] = record.pathname
        if hasattr(record, "lineno"):
            ctx["line"] = record.lineno

        # Capture exception if present
        exception = None
        if record.exc_info and record.exc_info[1]:
            exception = f"{type(record.exc_info[1]).__name__}: {record.exc_info[1]}"
        elif record.exc_text:
            exception = record.exc_text

        # Create and emit our LogRecord
        from .base import LogRecord
        log_record = LogRecord(
            level=level,
            message=record.getMessage(),
            logger=record.name,
            timestamp=record.created,
            context=ctx,
            exception=exception,
        )
        self._logger.emit(log_record)
