"""
Structured logging with request-context, JSON formatting, and latency tracking.

Usage::

    from domains.infrastructure.structured_log import StructuredLogger, LogContext

    log = StructuredLogger("slo.models")

    with LogContext(request_id="abc-123", model_id="gpt2"):
        log.info("Generating", tokens=50, latency_ms=123.4)
        # → {"ts": "...", "level": "INFO", "logger": "slo.models",
        #    "request_id": "abc-123", "model_id": "gpt2",
        #    "msg": "Generating", "tokens": 50, "latency_ms": 123.4}

Timing helpers::

    from domains.infrastructure.structured_log import log_timer, timed

    # Context manager
    with log_timer(log, "model load"):
        load_model()

    # Decorator
    @timed(log)
    def load_model():
        ...

FastAPI middleware::

    from domains.infrastructure.structured_log import request_log_middleware

    app.middleware("http")(request_log_middleware)
    # Adds request_id + timing to every request
"""

from __future__ import annotations

import functools
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Optional


# ── Thread-local request context ──────────────────────────────────────

_context = threading.local()


def get_request_id() -> Optional[str]:
    """Return the current thread's request ID (set by middleware or LogContext)."""
    return getattr(_context, "request_id", None)


def get_log_context() -> dict:
    """Return the current thread's full logging context dict."""
    return getattr(_context, "extra", {}).copy()


class LogContext:
    """Context manager that enriches every log message with context fields.

    Nested contexts merge — inner fields override outer.
    """

    def __init__(self, **kwargs: Any):
        self._new = kwargs
        self._prior: dict = {}

    def __enter__(self):
        self._prior = getattr(_context, "extra", {}).copy()
        merged = {**self._prior, **self._new}
        if "request_id" not in merged:
            merged["request_id"] = str(uuid.uuid4())[:8]
        _context.extra = merged
        _context.request_id = merged.get("request_id")
        return self

    def __exit__(self, *args):
        _context.extra = self._prior
        _context.request_id = self._prior.get("request_id")


# ── JSON formatter ───────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Includes timestamp, level, logger name, message, and any ``extra``
    dict merged from the thread-local context.
    """

    _KNOWN = frozenset({
        "name", "levelno", "levelname", "pathname", "filename", "module",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "process", "processName", "args", "msg",
        "exc_info", "exc_text", "stack_info", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        ctx = getattr(_context, "extra", {})
        obj: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if ctx:
            obj.update(ctx)
        # Merge extra fields injected via Logger.log(..., extra={...})
        # or via StructuredLogger.info("...", tag="foo")
        for key, val in record.__dict__.items():
            if key not in self._KNOWN:
                obj[key] = val
        if record.exc_info and record.exc_info[0]:
            obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(obj, default=str)


# ── StructuredLogger wrapper ─────────────────────────────────────────

class StructuredLogger:
    """Drop-in replacement for ``logging.getLogger()`` with structured extras.

    Supports both new-style keyword extras and legacy positional-arg + ``extra``
    dict usage::

        log = StructuredLogger("slo.models")

        # New style
        log.info("Loaded model", model="gpt2", load_time_ms=3200)

        # Legacy style (still works)
        log.info("Loaded %s in %.1fms", "gpt2", 3200, extra={"tag": "MODEL"})
    """

    def __init__(self, name: str, level: int = logging.INFO, **tags: Any):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._tags = dict(tags)

    def __repr__(self) -> str:
        tags = f", tags={self._tags}" if self._tags else ""
        return f"StructuredLogger({self._logger.name!r}, level={logging.getLevelName(self._logger.level)}{tags})"

    def __getattr__(self, attr: str):
        return getattr(self._logger, attr)

    def child(self, suffix: str, **tags: Any) -> "StructuredLogger":
        """Create a child logger with a dotted name suffix and extra tags.

        Tags merge with parent tags (child overrides parent)::

            parent = StructuredLogger("slo.training")
            child = parent.child("optimizer", phase="train")
            child.info("lr updated")
            # → {"logger": "slo.training.optimizer", "phase": "train", ...}
        """
        child_name = f"{self._logger.name}.{suffix}"
        merged_tags = {**self._tags, **tags}
        return StructuredLogger(child_name, level=self._logger.level, **merged_tags)

    def _log(self, level: int, msg: str, *args: Any, **extra: Any) -> None:
        extra_dict = extra.pop("extra", {})
        if isinstance(extra_dict, dict):
            extra_dict.update(extra)
        else:
            extra_dict = dict(extra)
        if self._tags:
            merged = dict(self._tags)
            merged.update(extra_dict)
            extra_dict = merged
        if args:
            self._logger.log(level, msg, *args, extra=extra_dict)
        else:
            self._logger.log(level, msg, extra=extra_dict)

    def debug(self, msg: str, *args: Any, **extra: Any) -> None:
        self._log(logging.DEBUG, msg, *args, **extra)

    def info(self, msg: str, *args: Any, **extra: Any) -> None:
        self._log(logging.INFO, msg, *args, **extra)

    def warning(self, msg: str, *args: Any, **extra: Any) -> None:
        self._log(logging.WARNING, msg, *args, **extra)

    def error(self, msg: str, *args: Any, **extra: Any) -> None:
        self._log(logging.ERROR, msg, *args, **extra)

    def critical(self, msg: str, *args: Any, **extra: Any) -> None:
        self._log(logging.CRITICAL, msg, *args, **extra)


# ── Timing helpers ────────────────────────────────────────────────────

@contextmanager
def log_timer(
    logger: StructuredLogger,
    label: str,
    level: int = logging.INFO,
    **extra: Any,
):
    """Context manager that logs elapsed time on exit.

    Usage::

        with log_timer(log, "model load"):
            load_model()
        # → "model load completed in 3.2s"
    """
    start = time.monotonic()
    try:
        yield
    except Exception:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger._log(
            logging.ERROR,
            f"{label} failed after {elapsed_ms:.1f}ms",
            elapsed_ms=round(elapsed_ms, 1),
            **extra,
        )
        raise
    else:
        elapsed_ms = (time.monotonic() - start) * 1000
        if elapsed_ms >= 1000:
            elapsed_str = f"{elapsed_ms / 1000:.1f}s"
        else:
            elapsed_str = f"{elapsed_ms:.0f}ms"
        logger._log(
            level,
            f"{label} completed in {elapsed_str}",
            elapsed_ms=round(elapsed_ms, 1),
            **extra,
        )


def timed(
    logger: StructuredLogger,
    level: int = logging.INFO,
    **extra: Any,
) -> Callable:
    """Decorator that logs function execution time.

    Usage::

        @timed(log)
        def load_model():
            ...

        @timed(log, level=logging.WARNING)
        def slow_operation():
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                result = fn(*args, **kwargs)
            except Exception:
                elapsed_ms = (time.monotonic() - start) * 1000
                logger._log(
                    logging.ERROR,
                    f"{fn.__name__} failed after {elapsed_ms:.1f}ms",
                    elapsed_ms=round(elapsed_ms, 1),
                    **extra,
                )
                raise
            else:
                elapsed_ms = (time.monotonic() - start) * 1000
                if elapsed_ms >= 1000:
                    elapsed_str = f"{elapsed_ms / 1000:.1f}s"
                else:
                    elapsed_str = f"{elapsed_ms:.0f}ms"
                logger._log(
                    level,
                    f"{fn.__name__} completed in {elapsed_str}",
                    elapsed_ms=round(elapsed_ms, 1),
                    **extra,
                )
                return result
        return wrapper
    return decorator


def tagged(
    logger: StructuredLogger,
    **tags: Any,
) -> StructuredLogger:
    """Return a proxy that injects ``tags`` into every log call.

    Usage::

        log = tagged(StructuredLogger("slo.training"), phase="train")
        log.info("Starting", epochs=10)
        # → {"msg": "Starting", "phase": "train", "epochs": 10}
    """
    class _TaggedProxy:
        __slots__ = ("_inner", "_tags")

        def __init__(self, inner: StructuredLogger, tags: dict):
            object.__setattr__(self, "_inner", inner)
            object.__setattr__(self, "_tags", tags)

        def _merge(self, extra: dict) -> dict:
            merged = dict(self._tags)
            merged.update(extra)
            return merged

        def debug(self, msg: str, *args: Any, **extra: Any) -> None:
            self._inner._log(logging.DEBUG, msg, *args, **self._merge(extra))

        def info(self, msg: str, *args: Any, **extra: Any) -> None:
            self._inner._log(logging.INFO, msg, *args, **self._merge(extra))

        def warning(self, msg: str, *args: Any, **extra: Any) -> None:
            self._inner._log(logging.WARNING, msg, *args, **self._merge(extra))

        def error(self, msg: str, *args: Any, **extra: Any) -> None:
            self._inner._log(logging.ERROR, msg, *args, **self._merge(extra))

        def critical(self, msg: str, *args: Any, **extra: Any) -> None:
            self._inner._log(logging.CRITICAL, msg, *args, **self._merge(extra))

    return _TaggedProxy(logger, tags)


def setup_structured_logging(
    root_level: int = logging.INFO,
    fmt: Optional[logging.Formatter] = None,
) -> None:
    """Install the JSON formatter on the root logger and all children.

    Call once at startup::

        from domains.infrastructure.structured_log import setup_structured_logging
        setup_structured_logging()
    """
    handler = logging.StreamHandler()
    handler.setFormatter(fmt or JSONFormatter())
    root = logging.getLogger()
    # Remove existing handlers to avoid duplicate output
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(root_level)
    # Quiet noisy third-party loggers
    for name in ("requests", "urllib3", "httpx", "asyncio", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── FastAPI middleware ────────────────────────────────────────────────

async def request_log_middleware(request, call_next):
    """FastAPI middleware — adds ``request_id`` + timing to every request.

    Usage::

        from domains.infrastructure.structured_log import request_log_middleware
        app.middleware("http")(request_log_middleware)
    """
    rid = request.headers.get("X-Request-Id") or str(uuid.uuid4())[:8]
    start = time.time()
    with LogContext(request_id=rid, path=request.url.path, method=request.method):
        response = await call_next(request)
        elapsed_ms = (time.time() - start) * 1000
        # Log basic request info
        logger = logging.getLogger("slo.http")
        logger.info(
            "%s %s → %d (%.1fms)",
            request.method, request.url.path, response.status_code, elapsed_ms,
            extra={"status": response.status_code, "elapsed_ms": round(elapsed_ms, 1)},
        )
    response.headers["X-Request-Id"] = rid
    return response
