"""
Centralized logging configuration — stdlib only, no external dependencies.

Replaces the scattered logging setup across main.py, cli.py, and bridge.py
with a single entry point that configures:

  - Console output (human-readable, JSON, or slo.log JSON)
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
    SLO_LOG_FORMAT  — "human" (colored terminal), "json" (legacy structured),
                      or "slo" (slo.log v1 typed schema) (default: human)
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
    "UI":     (_A.BLUE + _A.BOLD,     "UI"),
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


# ── Unified formatter ─────────────────────────────────────────────────

class LogFormatter(logging.Formatter):
    """Single formatter for all output types: human, json, slo.

    Works with both stdlib ``logging.LogRecord`` and OOP ``LogRecord``.

    Args:
        fmt:    Output format — ``"human"`` (colored terminal), ``"json"``
                (structured JSON lines), or ``"slo"`` (slo.log v1 envelope).
        colors: Enable ANSI colors in human mode.
    """

    def __init__(self, fmt: str = "human", colors: bool = True):
        super().__init__()
        self._mode = fmt
        self._colors = colors

    def format(self, record: logging.LogRecord) -> str:
        if self._mode == "json":
            return self._format_json(record)
        if self._mode == "slo":
            return self._format_slo(record)
        return self._format_human(record)

    def format_oop(self, record: "LogRecord") -> str:
        """Format an OOP LogRecord (from domains.logging.base)."""
        if self._mode == "json":
            return self._format_json_oop(record)
        if self._mode == "slo":
            return self._format_slo_oop(record)
        if self._mode == "cli":
            return self._format_cli_oop(record)
        if self._mode == "shell":
            return self._format_shell_oop(record)
        return self._format_human_oop(record)

    # ── human ────────────────────────────────────────────────────────

    def _format_human(self, record: logging.LogRecord) -> str:
        parts = []
        c = self._colors

        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        parts.append(f"{_A.GREY}{ts}{_A.RESET}" if c else ts)

        color, abbrev = _LEVEL_STYLE.get(record.levelno, (_A.WHITE, "???"))
        parts.append(f"{color}{_A.BOLD}{abbrev:>3}{_A.RESET}" if c else abbrev.rjust(3))

        tag = getattr(record, "op", None)
        if tag:
            domain = tag.split(".")[0].upper() if "." in tag else tag.upper()
            tc, tt = _TAG_STYLE.get(domain, (_A.CYAN, domain))
            parts.append(f"{tc}{_A.BOLD}[{tt}]{_A.RESET}" if c else f"[{tt}]")
        else:
            tag = getattr(record, "tag", None)
            if tag:
                tc, tt = _TAG_STYLE.get(tag, (_A.CYAN, tag))
                parts.append(f"{tc}{_A.BOLD}[{tt}]{_A.RESET}" if c else f"[{tt}]")

        logger_name = record.name.split(".")[-1] if record.name else ""
        if logger_name:
            parts.append(f"{_A.GREY}{_A.DIM}{logger_name}{_A.RESET}" if c else logger_name)

        parts.append(record.getMessage())

        rid = getattr(record, "request_id", None)
        if rid:
            parts.append(f"{_A.DIM}req={rid}{_A.RESET}" if c else f"req={rid}")

        ctx = _collect_extras(record)
        if ctx:
            ctx_parts = []
            for k, v in ctx.items():
                if c:
                    ctx_parts.append(f"{_A.DIM}{k}={_A.WHITE}{v}{_A.RESET}")
                else:
                    ctx_parts.append(f"{k}={v}")
            parts.append(" ".join(ctx_parts))

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

    # ── json ─────────────────────────────────────────────────────────

    def _format_json(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        rid = getattr(record, "request_id", None)
        if rid:
            entry["request_id"] = rid

        tag = getattr(record, "tag", None)
        if tag:
            entry["tag"] = tag

        ctx = _collect_extras(record)
        if ctx:
            entry["ctx"] = ctx

        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str, ensure_ascii=False)

    # ── slo ──────────────────────────────────────────────────────────

    def _format_slo(self, record: logging.LogRecord) -> str:
        op = _derive_op(record)
        rid = getattr(record, "request_id", None)
        ok = getattr(record, "ok", True)
        dur_ms = getattr(record, "dur_ms", None)
        err = getattr(record, "err", None)

        entry: dict[str, Any] = {
            "v": 1,
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "lvl": record.levelname,
            "op": op,
            "corr": rid,
            "dur_ms": dur_ms,
            "ok": ok,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        if err:
            entry["err"] = err

        tag = getattr(record, "tag", None)
        if tag:
            entry["tag"] = tag

        error_code = getattr(record, "error_code", None)
        if error_code:
            entry["error_code"] = error_code

        extras = _collect_extras(record)
        if extras:
            entry["ctx"] = extras

        domain, _, verb = op.partition(".")
        domain_payload = _collect_domain_payload(record, domain)
        if domain_payload:
            entry[domain] = domain_payload

        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str, ensure_ascii=False)

    # ── OOP LogRecord methods ───────────────────────────────────────

    def _format_human_oop(self, record: "LogRecord") -> str:
        parts = []
        c = self._colors

        ts = datetime.fromtimestamp(record.timestamp).strftime("%H:%M:%S")
        parts.append(f"{_A.GREY}{ts}{_A.RESET}" if c else ts)

        _level_map = {
            "debug": (_A.DIM, "DBG"), "info": (_A.GREEN, "INF"),
            "warning": (_A.YELLOW + _A.BOLD, "WRN"), "error": (_A.RED + _A.BOLD, "ERR"),
            "critical": (_A.BG_RED + _A.BOLD + _A.WHITE, "CRI"),
        }
        level_val = getattr(record.level, "value", str(record.level)) if record.level else "info"
        color, abbrev = _level_map.get(level_val, (_A.WHITE, "???"))
        parts.append(f"{color}{_A.BOLD}{abbrev:>3}{_A.RESET}" if c else abbrev.rjust(3))

        if record.tag:
            tc, tt = _TAG_STYLE.get(record.tag, (_A.CYAN, record.tag))
            parts.append(f"{tc}{_A.BOLD}[{tt}]{_A.RESET}" if c else f"[{tt}]")

        logger_name = record.logger.split(".")[-1] if record.logger else ""
        if logger_name:
            parts.append(f"{_A.GREY}{_A.DIM}{logger_name}{_A.RESET}" if c else logger_name)

        parts.append(record.message)

        if record.error_code:
            parts.append(f"({_A.YELLOW}{record.error_code}{_A.RESET})" if c else f"({record.error_code})")

        if record.context:
            ctx_parts = []
            for k, v in record.context.items():
                if c:
                    ctx_parts.append(f"{_A.DIM}{k}={_A.WHITE}{v}{_A.RESET}")
                else:
                    ctx_parts.append(f"{k}={v}")
            parts.append(" ".join(ctx_parts))

        if record.exception:
            exc_type = record.exception.split(":")[0].strip() if ":" in record.exception else record.exception
            if c:
                parts.append(f"{_A.RED}{_A.BOLD}[{exc_type}]{_A.RESET} {_A.RED}{record.exception}{_A.RESET}")
            else:
                parts.append(f"[{exc_type}] {record.exception}")

        return " ".join(parts)

    def _format_json_oop(self, record: "LogRecord") -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.timestamp, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.level.value.upper(),
            "logger": record.logger,
            "msg": record.message,
        }
        rid = getattr(record, "request_id", None)
        if rid:
            entry["corr"] = rid
        if record.tag:
            entry["tag"] = record.tag
        if record.error_code:
            entry["code"] = record.error_code
        if record.context:
            entry["ctx"] = record.context
        if record.exception:
            entry["err"] = record.exception
        return json.dumps(entry, default=str, ensure_ascii=False)

    def _format_slo_oop(self, record: "LogRecord") -> str:
        tag = record.tag or "sys.info"
        entry: dict[str, Any] = {
            "v": 1,
            "ts": datetime.fromtimestamp(record.timestamp, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "lvl": record.level.value.upper(),
            "op": tag,
            "corr": record.context.get("request_id") if record.context else None,
            "dur_ms": None,
            "ok": record.level.value not in ("error", "critical"),
            "logger": record.logger,
            "msg": record.message,
        }
        if record.error_code:
            entry["error_code"] = record.error_code
        if record.context:
            entry["ctx"] = record.context
        if record.exception:
            entry["err"] = record.exception
        return json.dumps(entry, default=str, ensure_ascii=False)

    def _format_cli_oop(self, record: "LogRecord") -> str:
        """CLI format: timestamp + icon + message, two-line output."""
        c = self._colors

        def _wrap(text: str, style: str) -> str:
            return f"{style}{text}{_A.RESET}" if c else text

        _cli_icons = {
            "debug": (_A.DIM + _A.CYAN, "·"),
            "info": (_A.GREEN, "ℹ"),
            "warning": (_A.BOLD + _A.YELLOW, "!"),
            "error": (_A.BOLD + _A.RED, "✗"),
            "critical": (_A.BG_RED + _A.BOLD + _A.WHITE, "✗"),
        }
        color, icon = _cli_icons.get(record.level.value, (_A.WHITE, "·"))

        ts = datetime.fromtimestamp(record.timestamp).strftime("%H:%M:%S")
        msg = record.message
        if record.exception:
            msg += _wrap(f" — {record.exception}", _A.RED)
        primary = f"  {_wrap(ts, _A.DIM)} {_wrap(icon, color)} {msg}"

        meta_parts = []
        if record.logger:
            meta_parts.append(_wrap(record.logger, _A.DIM))
        if record.context:
            ctx_str = " ".join(f"{k}={v}" for k, v in record.context.items())
            meta_parts.append(_wrap(ctx_str, _A.DIM))

        lines = [primary]
        if meta_parts:
            lines.append("    " + " ".join(meta_parts))
        return "\n".join(lines)

    def _format_shell_oop(self, record: "LogRecord") -> str:
        """Shell format: timestamp + icon + level + logger + context + message."""
        c = self._colors

        def _wrap(text: str, style: str) -> str:
            return f"{style}{text}{_A.RESET}" if c else text

        parts = []

        ts = datetime.fromtimestamp(record.timestamp).strftime("%H:%M:%S")
        parts.append(_wrap(ts, _A.DIM))

        _shell_icons = {
            "debug": (_A.DIM + _A.CYAN, "·"),
            "info": (_A.GREEN, "ℹ"),
            "warning": (_A.YELLOW + _A.BOLD, "!"),
            "error": (_A.RED + _A.BOLD, "✗"),
            "critical": (_A.RED + _A.BOLD, "✗"),
        }
        color, icon = _shell_icons.get(record.level.value, ("", "·"))
        level_label = record.level.value.upper()
        parts.append(_wrap(f"{icon} {level_label}", color))

        parts.append(_wrap(record.logger, _A.DIM + _A.CYAN))

        if record.context:
            ctx_str = " ".join(f"{k}={v}" for k, v in record.context.items())
            parts.append(_wrap(ctx_str, _A.DIM))

        parts.append(record.message)

        if record.exception:
            parts.append(_wrap(record.exception, _A.RED))

        return " ".join(parts)


# ── Legacy tag → slo.log op mapping (backward compat) ────────────────

_LEGACY_TAG_TO_OP = {
    "REQ":    "http.request",
    "AUTH":   "http.auth",
    "MODEL":  "model.load",
    "SOUL":   "model.load",
    "TRAIN":  "train.step",
    "INFRA":  "infra.error",
    "START":  "sys.startup",
    "SLOW":   "http.request",
    "ERROR":  "infra.error",
    "WARN":   "infra.error",
    "OK":     "sys.info",
    "INF":    "sys.info",
    "COG":    "rag.query",
    "IDLE":   "sys.info",
    "KV":     "sys.info",
    "GPU":    "infer.generate",
    "LEARN":  "train.step",
    "BENCH":  "sys.info",
    "EVENT":  "sys.info",
    "CHAT":   "http.request",
    "DOWNLOAD": "download.start",
    "WORKFLOW": "workflow.start",
    "SYSTEM":   "sys.info",
}


def _derive_op(record: logging.LogRecord) -> str:
    """Derive the slo.log ``op`` field from record extras or legacy tag.

    Priority:
      1. ``record.op`` (explicit — callers that have migrated)
      2. ``record.tag`` → lookup in ``_LEGACY_TAG_TO_OP``
      3. Fallback: ``"sys.info"``
    """
    explicit = getattr(record, "op", None)
    if explicit:
        return explicit
    tag = getattr(record, "tag", None)
    if tag:
        return _LEGACY_TAG_TO_OP.get(tag, "sys.info")
    return "sys.info"





def _collect_domain_payload(record: logging.LogRecord, domain: str) -> dict:
    """Extract domain-specific payload from record extras."""
    _DOMAIN_KEYS = {
        "http":    {"method", "path", "status", "elapsed_s"},
        "train":   {"job_id", "epoch", "step", "total_steps", "loss", "lr"},
        "model":   {"id", "layers", "weights_count", "file_mb", "source"},
        "infer":   {"model_id", "tokens", "session_id", "prompt_len", "timeout_s"},
        "infra":   {"component", "worker_id", "model_id", "reason", "restart_count", "max_restarts"},
        "sys":     {"phase", "signal", "version"},
        "web":     {"event", "path"},
        "rag":     {"chunks", "chars", "top_k", "results", "verified", "confidence", "citations"},
        "download": {"resource", "elapsed_s", "url", "bytes", "speed"},
        "workflow": {"job_id", "kind", "status"},
    }
    keys = _DOMAIN_KEYS.get(domain, set())
    if not keys:
        return {}
    payload = {}
    for key in keys:
        val = getattr(record, key, None)
        if val is not None:
            payload[key] = val
    return payload or {}


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
        if key in ("tag", "request_id", "error_code", "op", "corr", "dur_ms", "ok", "err"):
            continue
        ctx[key] = val
    return ctx


_KNOWN_KEYS = frozenset({
    "name", "levelno", "levelname", "pathname", "filename", "module",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "process", "processName", "args", "msg",
    "exc_info", "exc_text", "stack_info", "taskName", "message",
    "asctime", "tag", "request_id", "error_code",
    # slo.log v1 envelope fields
    "op", "corr", "dur_ms", "ok", "err",
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
    fmt: str = "json",
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
    handler.setFormatter(LogFormatter(fmt=fmt, colors=False))
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
    except Exception as e:
        logger.debug("OutputBuffer bridge unavailable: %s", e)
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
            except Exception as e:
                logger.debug("find_repo_root failed, using default log path: %s", e)
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
        console_formatter = LogFormatter(fmt=fmt, colors=colors)

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(ClientExtensionFilter())
        root.addHandler(console_handler)

    # File handler
    file_handler = None
    if use_file:
        try:
            file_handler = _create_file_handler(log_path, level=logging.DEBUG, fmt=fmt)
            root.addHandler(file_handler)
        except Exception as e:
            # File logging is best-effort — warn once so operators know it failed
            logging.getLogger(__name__).warning("Could not create log file handler: %s", e)

    # OutputBuffer bridge (for SSE streaming)
    bridge = None
    if enable_output_buffer:
        bridge = _install_output_buffer_bridge(root)

    # Dashboard event buffer filter (captures tagged events for CLI monitor)
    try:
        from domains.logging.dashboard_filter import DashboardFilter
        root.addFilter(DashboardFilter())
    except Exception:
        pass

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



