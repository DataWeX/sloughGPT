"""
Centralized logging configuration — stdlib only, no external dependencies.

Replaces the scattered logging setup across main.py, cli.py, and bridge.py
with a single entry point that configures:

  - Console output (human-readable or JSON)
  - File output with rotation (JSON lines)
  - Request correlation IDs (via contextvars)
  - Third-party logger suppression
  - OutputBuffer integration (SSE streaming)

Usage::

    from domains.logging.config import setup_logging

    # At startup (once):
    setup_logging()  # reads SLO_LOG_LEVEL, SLO_LOG_FORMAT, SLO_LOG_DIR env vars

    # In any module:
    import logging
    logger = logging.getLogger("slo.mymodule")
    logger.info("doing work", extra={"tag": "INFRA", "request_id": get_request_id()})

Environment variables:
    SLO_LOG_LEVEL   — root log level (default: INFO)
    SLO_LOG_FORMAT  — "human" (colored terminal) or "json" (structured) (default: human)
    SLO_LOG_DIR     — directory for log files (default: logs/ relative to repo root)
    SLO_LOG_NO_FILE — set to "1" to disable file logging
    NO_COLOR        — set to "1" to disable ANSI colors
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import os
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── Correlation ID via contextvars (thread-safe, async-safe) ──────────

_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "slo_request_id", default=None
)

_log_context: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "slo_log_context", default={}
)


def get_request_id() -> Optional[str]:
    """Return the current request's correlation ID."""
    return _request_id.get()


def set_request_id(request_id: str) -> None:
    """Set the current request's correlation ID."""
    _request_id.set(request_id)


def get_log_context() -> dict:
    """Return the current thread's structured logging context."""
    return _log_context.get().copy()


def set_log_context(**kwargs: Any) -> None:
    """Merge key-value pairs into the current logging context."""
    current = _log_context.get().copy()
    current.update(kwargs)
    _log_context.set(current)


def clear_log_context() -> None:
    """Clear the current logging context."""
    _log_context.set({})


# ── ANSI codes ────────────────────────────────────────────────────────

_NO_COLOR = os.environ.get("NO_COLOR", "").strip() == "1"
_SLO_LOG_COLOR = os.environ.get("SLO_LOG_COLOR", "").strip().lower()


def _color_enabled(stream=None) -> bool:
    if _NO_COLOR:
        return False
    if _SLO_LOG_COLOR in ("1", "true", "yes", "on"):
        return True
    if _SLO_LOG_COLOR in ("0", "false", "no", "off"):
        return False
    try:
        return bool((stream or sys.stderr).isatty())
    except (AttributeError, ValueError):
        return False


class _A:
    """ANSI escape codes."""
    RESET    = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    RED      = "\033[31m"
    GREEN    = "\033[32m"
    YELLOW   = "\033[33m"
    BLUE     = "\033[34m"
    MAGENTA  = "\033[35m"
    CYAN     = "\033[36m"
    WHITE    = "\033[37m"
    GREY     = "\033[90m"
    BG_RED   = "\033[41m"


# ── Level formatting ──────────────────────────────────────────────────

_LEVEL_ABBR = {
    logging.DEBUG:    "DBG",
    logging.INFO:     "INF",
    logging.WARNING:  "WRN",
    logging.ERROR:    "ERR",
    logging.CRITICAL: "CRI",
}

_LEVEL_STYLE = {
    logging.DEBUG:    (_A.DIM,                            "DBG"),
    logging.INFO:     (_A.GREEN,                          "INF"),
    logging.WARNING:  (_A.YELLOW + _A.BOLD,               "WRN"),
    logging.ERROR:    (_A.RED + _A.BOLD,                  "ERR"),
    logging.CRITICAL: (_A.BG_RED + _A.BOLD + _A.WHITE,   "CRI"),
}

_TAG_STYLE = {
    "REQ":    (_A.CYAN + _A.BOLD,     "REQ"),
    "AUTH":   (_A.MAGENTA + _A.BOLD,  "AUTH"),
    "MODEL":  (_A.BLUE + _A.BOLD,     "MODEL"),
    "SOUL":   (_A.CYAN,               "SOUL"),
    "TRAIN":  (_A.GREEN + _A.BOLD,    "TRAIN"),
    "INFRA":  (_A.GREY + _A.BOLD,     "INFRA"),
    "START":  (_A.GREEN + _A.BOLD,    "START"),
    "SLOW":   (_A.YELLOW + _A.DIM,    "SLOW"),
    "ERROR":  (_A.RED + _A.BOLD,      "ERROR"),
    "WARN":   (_A.YELLOW + _A.BOLD,   "WARN"),
    "OK":     (_A.GREEN,              "OK"),
    "INF":    (_A.CYAN,               "INF"),
    "COG":    (_A.MAGENTA,            "COG"),
    "IDLE":   (_A.GREY,               "IDLE"),
    "KV":     (_A.BLUE,               "KV"),
    "GPU":    (_A.YELLOW,             "GPU"),
    "LEARN":  (_A.GREEN,              "LEARN"),
    "BENCH":  (_A.MAGENTA,            "BENCH"),
    "EVENT":  (_A.CYAN,               "EVENT"),
}


# ── Record factory — auto-injects correlation ID ─────────────────────

_original_record_factory = logging.getLogRecordFactory()


def _enriched_record_factory(*args, **kwargs):
    """LogRecord factory that auto-injects request_id from contextvars."""
    record = _original_record_factory(*args, **kwargs)
    rid = _request_id.get()
    # Fallback: check schemas.common contextvar if set_request_id wasn't called
    if not rid:
        try:
            from schemas.common import get_correlation_id
            rid = get_correlation_id()
        except (ImportError, AttributeError):
            pass
    if rid and not hasattr(record, "request_id"):
        record.request_id = rid
    # Merge structured context into record extras
    ctx = _log_context.get()
    if ctx:
        for k, v in ctx.items():
            if not hasattr(record, k):
                setattr(record, k, v)
    return record


# ── Human-readable formatter ─────────────────────────────────────────

class HumanFormatter(logging.Formatter):
    """Colored terminal output: HH:MM:SS LVL [TAG] logger message key=val"""

    def __init__(self, colors: bool = True):
        super().__init__()
        self._colors = colors

    def format(self, record: logging.LogRecord) -> str:
        parts = []
        c = self._colors

        # Timestamp
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        parts.append(f"{_A.GREY}{ts}{_A.RESET}" if c else ts)

        # Level badge
        color, abbrev = _LEVEL_STYLE.get(record.levelno, (_A.WHITE, "???"))
        parts.append(f"{color}{_A.BOLD}{abbrev:>3}{_A.RESET}" if c else abbrev.rjust(3))

        # Tag
        tag = getattr(record, "tag", None)
        if tag:
            tc, tt = _TAG_STYLE.get(tag, (_A.CYAN, tag))
            parts.append(f"{tc}{_A.BOLD}[{tt}]{_A.RESET}" if c else f"[{tt}]")

        # Logger name (last component only)
        logger_name = record.name.split(".")[-1] if record.name else ""
        if logger_name:
            parts.append(f"{_A.GREY}{_A.DIM}{logger_name}{_A.RESET}" if c else logger_name)

        # Message
        parts.append(record.getMessage())

        # Request ID (if present and not already in tag)
        rid = getattr(record, "request_id", None)
        if rid:
            parts.append(f"{_A.DIM}req={rid}{_A.RESET}" if c else f"req={rid}")

        # Structured context — collect non-standard fields
        ctx = _collect_extras(record)
        if ctx:
            ctx_parts = []
            for k, v in ctx.items():
                if c:
                    ctx_parts.append(f"{_A.DIM}{k}={_A.WHITE}{v}{_A.RESET}")
                else:
                    ctx_parts.append(f"{k}={v}")
            parts.append(" ".join(ctx_parts))

        # Exception
        if record.exc_info and record.exc_info[1]:
            exc_type = type(record.exc_info[1]).__name__
            exc_msg = str(record.exc_info[1])
            if c:
                parts.append(f"{_A.RED}{_A.BOLD}[{exc_type}]{_A.RESET} {_A.RED}{exc_msg}{_A.RESET}")
            else:
                parts.append(f"[{exc_type}] {exc_msg}")
        elif record.exc_text:
            parts.append(record.exc_text)

        return " ".join(parts)


# ── JSON formatter ────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """Structured JSON lines: one JSON object per line."""

    _KNOWN = frozenset({
        "name", "levelno", "levelname", "pathname", "filename", "module",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "process", "processName", "args", "msg",
        "exc_info", "exc_text", "stack_info", "taskName", "message",
    })

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Correlation ID
        rid = getattr(record, "request_id", None)
        if rid:
            entry["request_id"] = rid

        # Tag
        tag = getattr(record, "tag", None)
        if tag:
            entry["tag"] = tag

        # Structured extras
        ctx = _collect_extras(record)
        if ctx:
            entry["ctx"] = ctx

        # Exception
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str, ensure_ascii=False)


def _collect_extras(record: logging.LogRecord) -> dict:
    """Extract non-standard extra fields from a LogRecord."""
    ctx = {}
    for key, val in record.__dict__.items():
        if key in _KNOWN_KEYS or key.startswith("_"):
            continue
        # Skip standard attributes
        if key in ("msg", "args", "levelname", "levelno", "pathname", "filename",
                    "module", "exc_info", "exc_text", "stack_info", "lineno",
                    "funcName", "created", "msecs", "relativeCreated", "thread",
                    "threadName", "processName", "process", "taskName", "message",
                    "name", "asctime"):
            continue
        # Skip already-handled fields
        if key in ("tag", "request_id", "error_code"):
            continue
        ctx[key] = val
    return ctx


_KNOWN_KEYS = frozenset({
    "name", "levelno", "levelname", "pathname", "filename", "module",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "process", "processName", "args", "msg",
    "exc_info", "exc_text", "stack_info", "taskName", "message",
    "asctime", "tag", "request_id", "error_code",
})


# ── Third-party logger suppression ───────────────────────────────────

_NOISY_LOGGERS = {
    "httpx":                  logging.WARNING,
    "httpcore":               logging.WARNING,
    "urllib3":                logging.WARNING,
    "uvicorn.access":         logging.WARNING,
    "watchfiles":             logging.WARNING,
    "asyncio":                logging.WARNING,
    "PIL":                    logging.WARNING,
    "urllib3.connectionpool":  logging.WARNING,
}


# ── Client extension filter ───────────────────────────────────────────

class ClientExtensionFilter(logging.Filter):
    """Suppress noisy client errors from browser extensions."""
    _PATTERNS = ("CLIENT ERROR", "0 0", "chrome-extension://", "moz-extension://")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._PATTERNS)


# ── File handler with rotation ────────────────────────────────────────

def _create_file_handler(
    log_dir: Path,
    level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.handlers.RotatingFileHandler:
    """Create a rotating file handler for structured JSON log output."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sloughgpt.log"

    handler = logging.handlers.RotatingFileHandler(
        str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter())
    handler.addFilter(ClientExtensionFilter())
    return handler


# ── OutputBuffer integration ──────────────────────────────────────────

def _install_output_buffer_bridge(root: logging.Logger) -> Optional[Any]:
    """Install the OutputBuffer log handler if available."""
    try:
        from domains.infrastructure.output_buffer import install_log_bridge, install_stdio_bridge
        buf_handler = install_log_bridge()
        install_stdio_bridge()
        return buf_handler
    except Exception:
        return None


# ── Main setup function ──────────────────────────────────────────────

def setup_logging(
    level: Optional[str] = None,
    format: Optional[str] = None,
    log_dir: Optional[str] = None,
    enable_file: Optional[bool] = None,
    enable_console: bool = True,
    enable_output_buffer: bool = True,
) -> dict[str, Any]:
    """Configure all logging in one place.

    Call once at startup (main.py or cli.py). Reads env vars for defaults.

    Args:
        level:            Log level (default: SLO_LOG_LEVEL env or "INFO")
        format:           "human" or "json" (default: SLO_LOG_FORMAT env or "human")
        log_dir:          Directory for log files (default: SLO_LOG_DIR env or "logs/")
        enable_file:      Enable file logging (default: SLO_LOG_NO_FILE != "1")
        enable_console:   Install console stderr handler (default: True).
                          Set False when using BridgeHandler + CLILogger (CLI mode).
        enable_output_buffer: Install OutputBuffer bridge (default: True)

    Returns:
        dict with setup info: {"level", "format", "log_dir", "file_handler", "bridge"}
    """
    # Resolve config from env / args
    level_name = (level or os.environ.get("SLO_LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)
    fmt = (format or os.environ.get("SLO_LOG_FORMAT", "human")).lower()
    no_file = os.environ.get("SLO_LOG_NO_FILE", "").strip() == "1"
    use_file = enable_file if enable_file is not None else not no_file

    # Determine log directory
    if log_dir:
        log_path = Path(log_dir)
    else:
        env_dir = os.environ.get("SLO_LOG_DIR", "").strip()
        if env_dir:
            log_path = Path(env_dir)
        else:
            # Default: logs/ relative to repo root
            try:
                from domains.shared import find_repo_root
                repo = find_repo_root(Path(__file__).resolve())
                log_path = repo / "logs"
            except Exception:
                log_path = Path("logs")

    # Install record factory for correlation IDs
    logging.setLogRecordFactory(_enriched_record_factory)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove all existing handlers to avoid duplicates
    for h in list(root.handlers):
        root.removeHandler(h)

    # Console handler
    if enable_console:
        colors = _color_enabled(sys.stderr)
        if fmt == "json":
            console_formatter = JSONFormatter()
        else:
            console_formatter = HumanFormatter(colors=colors)

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(ClientExtensionFilter())
        root.addHandler(console_handler)

    # File handler
    file_handler = None
    if use_file:
        try:
            file_handler = _create_file_handler(log_path, level=logging.DEBUG)
            root.addHandler(file_handler)
        except Exception:
            pass  # File logging is best-effort

    # OutputBuffer bridge (for SSE streaming)
    bridge = None
    if enable_output_buffer:
        bridge = _install_output_buffer_bridge(root)

    # Suppress noisy third-party loggers
    for logger_name, logger_level in _NOISY_LOGGERS.items():
        logging.getLogger(logger_name).setLevel(logger_level)

    return {
        "level": level_name,
        "format": fmt,
        "log_dir": str(log_path),
        "file_handler": file_handler,
        "bridge": bridge,
    }
