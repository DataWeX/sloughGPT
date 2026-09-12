"""
BridgeHandler — routes Python ``logging`` calls through a ``Logger`` instance.

Allows the rest of the codebase (which uses ``logging.getLogger("slo.xxx")``)
to output through the new OOP logger hierarchy without changing every call site.

Supports passing ``error_code`` and ``tag`` via ``extra``::

    logger.error("Model OOM", extra={"error_code": "E_MODEL_OOM", "tag": "MODEL"})
    logger.info("loaded", extra={"context": {"model": "gpt2"}})

Usage::

    import logging
    from domains.logging import ConsoleLogger, BridgeHandler

    log = ConsoleLogger("slo", level=LogLevel.DEBUG)
    handler = BridgeHandler(log)
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG)

    # Now all logging.getLogger("slo.xxx").info(...) routes through ConsoleLogger
"""

from __future__ import annotations

import logging

from .base import Logger, LogLevel

# ── Standard logging → our LogLevel mapping ────────────────────────────

_LEVEL_MAP = {
    logging.DEBUG:    LogLevel.DEBUG,
    logging.INFO:     LogLevel.INFO,
    logging.WARNING:  LogLevel.WARNING,
    logging.ERROR:    LogLevel.ERROR,
    logging.CRITICAL: LogLevel.CRITICAL,
}

# Standard logging.LogRecord attributes.  Anything on the record beyond
# these (plus the explicitly handled context/error_code/tag) was injected
# via ``extra={...}`` and belongs in the record's context — mirroring the
# native Logger API where ``log.info(msg, **ctx)`` merges kwargs into context.
_STANDARD_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
})

_HANDLED_ATTRS = frozenset({"context", "error_code", "tag"})


def record_extra_context(record: logging.LogRecord) -> dict:
    """Collect structured context from a stdlib ``LogRecord``.

    Merges the explicit ``context`` extra dict, then auto-captures any other
    non-standard ``extra`` fields (injected by stdlib as record attributes)
    so structured telemetry renders as ``key=value`` context — mirroring the
    native ``Logger`` API where ``log.info(msg, **ctx)`` merges kwargs into
    context.  Explicit context keys win over stray top-level fields.
    """
    ctx: dict = {}
    extra_ctx = getattr(record, "context", None)
    if isinstance(extra_ctx, dict):
        ctx.update(extra_ctx)
    for key, value in record.__dict__.items():
        if key in _STANDARD_ATTRS or key in _HANDLED_ATTRS:
            continue
        if key not in ctx:
            ctx[key] = value
    return ctx


class BridgeHandler(logging.Handler):
    """A ``logging.Handler`` that delegates to a ``Logger`` instance.

    This lets any module using ``logging.getLogger("slo.xxx")`` output
    through the new OOP logger hierarchy.

    Supports ``extra`` dict keys:
        - ``context``: dict merged into the record's context
        - ``error_code``: str error code (e.g. ``"E_MODEL_OOM"``)
        - ``tag``: str type tag (e.g. ``"MODEL"``, ``"REQ"``)

    Any other ``extra`` keys are captured automatically into the record's
    context (mirroring ``Logger.info(msg, **ctx)``), so structured fields
    such as ``extra={"mode": "guard", "elapsed_ms": 511}`` render as
    ``key=value`` context without nesting them under ``context`` manually.

    Parameters:
        logger: The ``Logger`` instance to delegate to.
    """

    def __init__(self, logger: Logger) -> None:
        super().__init__()
        self._logger = logger

    def emit(self, record: logging.LogRecord) -> None:
        """Convert a standard ``logging.LogRecord`` and emit via our Logger."""
        level = _LEVEL_MAP.get(record.levelno, LogLevel.INFO)

        # Build context from standard logging attributes (skip noisy path/lineno
        # at INFO+ — only useful for DEBUG-level tracing)
        ctx = {}
        if level < LogLevel.INFO:
            if hasattr(record, "pathname"):
                ctx["path"] = record.pathname
            if hasattr(record, "lineno"):
                ctx["line"] = record.lineno

        # Merge explicit context + auto-capture non-standard extra fields
        ctx.update(record_extra_context(record))

        # Capture exception if present
        exception = None
        if record.exc_info and record.exc_info[1]:
            exception = f"{type(record.exc_info[1]).__name__}: {record.exc_info[1]}"
        elif record.exc_text:
            exception = record.exc_text

        # Extract error_code and tag from extra
        error_code = getattr(record, "error_code", None)
        tag = getattr(record, "tag", None)

        # Create and emit our LogRecord
        from .base import LogRecord
        log_record = LogRecord(
            level=level,
            message=record.getMessage(),
            logger=record.name,
            timestamp=record.created,
            context=ctx,
            exception=exception,
            error_code=error_code,
            tag=tag,
        )
        self._logger.emit(log_record)
