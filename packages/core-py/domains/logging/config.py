"""
Centralized logging configuration — stdlib only, no external dependencies.

Single unified formatter (SloFormatter) handles both console and file output:
  - Console: colored human-readable with op-based tag badges
  - File: slo.log v1 JSON with envelope + domain payloads
  - Legacy tag backward compatibility via _derive_op()

Usage::

    from domains.logging.config import setup_logging

    # At startup (once):
    setup_logging()  # reads SLO_LOG_LEVEL, SLO_LOG_DIR env vars

    # In any module:
    import logging
    logger = logging.getLogger("slo.mymodule")
    logger.info("doing work", extra={"op": "infra.startup"})

Environment variables:
    SLO_LOG_LEVEL   — root log level (default: INFO)
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
    if not rid:
        try:
            from domains.infrastructure.correlation import get_correlation_id
            rid = get_correlation_id()
        except (ImportError, AttributeError):
            pass
    if rid and not hasattr(record, "request_id"):
        record.request_id = rid
    ctx = _log_context.get()
    if ctx:
        for k, v in ctx.items():
            if not hasattr(record, k):
                setattr(record, k, v)
    return record


# ── Legacy tag -> slo.log op mapping (backward compat) ────────────────

_LEGACY_TAG_TO_OP = {
    "REQ":      "http.request",
    "AUTH":     "http.auth",
    "MODEL":    "model.load",
    "SOUL":     "model.load",
    "TRAIN":    "train.step",
    "INFRA":    "infra.error",
    "START":    "sys.startup",
    "SLOW":     "http.request",
    "ERROR":    "infra.error",
    "WARN":     "infra.error",
    "OK":       "sys.info",
    "INF":      "sys.info",
    "COG":      "rag.query",
    "IDLE":     "sys.info",
    "KV":       "sys.info",
    "GPU":      "infer.generate",
    "LEARN":    "train.step",
    "BENCH":    "sys.info",
    "EVENT":    "sys.info",
    "CHAT":     "http.request",
    "DOWNLOAD": "download.start",
    "WORKFLOW": "workflow.start",
    "SYSTEM":   "sys.info",
}


def _derive_op(record: logging.LogRecord) -> str:
    """Derive the slo.log op field from record extras or legacy tag.

    Priority:
      1. record.op (explicit - callers that have migrated)
      2. record.tag -> lookup in _LEGACY_TAG_TO_OP
      3. Fallback: "sys.info"
    """
    explicit = getattr(record, "op", None)
    if explicit:
        return explicit
    tag = getattr(record, "tag", None)
    if tag:
        return _LEGACY_TAG_TO_OP.get(tag, "sys.info")
    return "sys.info"


# ── Domain payload extraction ─────────────────────────────────────────

_DOMAIN_KEYS = {
    "http":     {"method", "path", "status", "corr", "phase"},
    "train":    {"job_id", "epoch", "step", "total_steps", "loss", "lr"},
    "model":    {"id", "layers", "weights_count", "file_mb", "source", "quant_bits", "quant_mode"},
    "infer":    {"model_id", "tokens", "session_id", "prompt_len", "timeout_s"},
    "infra":    {"component", "worker_id", "model_id", "reason", "restart_count", "max_restarts"},
    "sys":      {"phase", "signal", "version"},
    "web":      {"event", "path"},
    "rag":      {"chunks", "chars", "top_k", "results", "verified", "confidence", "citations"},
    "download": {"resource", "elapsed_s", "url", "bytes", "speed"},
    "workflow": {"job_id", "kind", "status"},
}


def _collect_domain_payload(record: logging.LogRecord, domain: str) -> dict:
    """Extract domain-specific payload from record extras."""
    keys = _DOMAIN_KEYS.get(domain, set())
    if not keys:
        return {}
    payload = {}
    for key in keys:
        val = getattr(record, key, None)
        if val is not None:
            payload[key] = val
    return payload or {}


# ── Non-standard extras extraction ────────────────────────────────────

_KNOWN_KEYS = frozenset({
    "name", "levelno", "levelname", "pathname", "filename", "module",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "process", "processName", "args", "msg",
    "exc_info", "exc_text", "stack_info", "taskName", "message",
    "asctime", "tag", "request_id", "error_code",
    "op", "corr", "dur_ms", "ok", "err",
})


def _collect_extras(record: logging.LogRecord) -> dict:
    """Extract non-standard extra fields from a LogRecord."""
    ctx = {}
    for key, val in record.__dict__.items():
        if key in _KNOWN_KEYS or key.startswith("_"):
            continue
        if key in ("msg", "args", "levelname", "levelno", "pathname", "filename",
                    "module", "exc_info", "exc_text", "stack_info", "lineno",
                    "funcName", "created", "msecs", "relativeCreated", "thread",
                    "threadName", "processName", "process", "taskName", "message",
                    "name", "asctime"):
            continue
        if key in ("tag", "request_id", "error_code", "op", "corr", "dur_ms", "ok", "err"):
            continue
        ctx[key] = val
    return ctx


# ── Unified formatter ─────────────────────────────────────────────────

class SloFormatter(logging.Formatter):
    """Single unified formatter for console and file output.

    Usage:
        SloFormatter()              — human-readable with auto-detected colors
        SloFormatter(colors=False)  — human-readable without colors
        SloFormatter(fmt="json")    — slo.log v1 JSON (for file handler)

    Console (fmt="human"):
        HH:MM:SS LVL [OP] logger message key=val

    File (fmt="json"):
        {"v":1,"ts":"...","lvl":"INFO","op":"model.load","corr":"abc1",...}
    """

    def __init__(self, colors: bool = True, fmt: str = "human"):
        super().__init__()
        self._colors = colors
        self._fmt = fmt

    def format(self, record: logging.LogRecord) -> str:
        if self._fmt == "json":
            return self._format_json(record)
        return self._format_human(record)

    def _format_human(self, record: logging.LogRecord) -> str:
        parts = []
        c = self._colors

        # Timestamp
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        parts.append(f"{_A.GREY}{ts}{_A.RESET}" if c else ts)

        # Level badge
        color, abbrev = _LEVEL_STYLE.get(record.levelno, (_A.WHITE, "???"))
        parts.append(f"{color}{_A.BOLD}{abbrev:>3}{_A.RESET}" if c else abbrev.rjust(3))

        # Tag — prefer op, fall back to legacy tag
        op = getattr(record, "op", None)
        if op:
            domain = op.split(".")[0].upper() if "." in op else op.upper()
            tc, tt = _TAG_STYLE.get(domain, (_A.CYAN, domain))
            parts.append(f"{tc}{_A.BOLD}[{tt}]{_A.RESET}" if c else f"[{tt}]")
        else:
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

        # Request ID
        rid = getattr(record, "request_id", None)
        if rid:
            parts.append(f"{_A.DIM}req={rid}{_A.RESET}" if c else f"req={rid}")

        # Structured context
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

    def _format_json(self, record: logging.LogRecord) -> str:
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
            "msg": record.getMessage(),
            "corr": rid,
            "dur_ms": dur_ms,
            "ok": ok,
            "logger": record.name,
        }

        if err:
            entry["err"] = err

        # Domain payload — exactly one per line
        domain = op.split(".")[0] if "." in op else op
        domain_payload = _collect_domain_payload(record, domain)
        if domain_payload:
            entry[domain] = domain_payload

        # Legacy tag (backward compat for consumers still reading it)
        tag = getattr(record, "tag", None)
        if tag:
            entry["tag"] = tag

        # Non-standard extras
        ctx = _collect_extras(record)
        if ctx:
            entry["ctx"] = ctx

        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)

        raw = json.dumps(entry, default=str, ensure_ascii=False)

        if not self._colors:
            return raw

        return _colorize_json(raw)


def _colorize_json(raw: str) -> str:
    """Apply ANSI syntax highlighting to a JSON string.

    Semantic coloring:
      - Keys: cyan
      - Strings: green
      - true: green (ok=true), yellow (other)
      - false: red (ok=false), yellow (other)
      - null: grey
      - Numbers: yellow
      - lvl values: green(INFO), yellow(WRN), red(ERR/CRI)
    """
    out = []
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == '"':
            # Find end of string
            j = i + 1
            while j < n:
                if raw[j] == '\\':
                    j += 2
                    continue
                if raw[j] == '"':
                    j += 1
                    break
                j += 1
            s = raw[i:j]
            # Check if this is a key (followed by ':')
            k = j
            while k < n and raw[k] == ' ':
                k += 1
            if k < n and raw[k] == ':':
                out.append(f"{_A.CYAN}{s}{_A.RESET}")
            else:
                out.append(f"{_A.GREEN}{s}{_A.RESET}")
            i = j
        elif ch in ('{', '}', '[', ']'):
            out.append(f"{_A.GREY}{ch}{_A.RESET}")
            i += 1
        elif ch == ':':
            out.append(f"{_A.GREY}:{_A.RESET}")
            i += 1
        elif ch == ',':
            out.append(f"{_A.GREY},{_A.RESET}")
            i += 1
        elif ch == ' ':
            out.append(' ')
            i += 1
        elif raw[i:i+4] == 'true':
            # Check context: is this an "ok" value?
            preceding = raw[max(0, i-20):i]
            if '"ok"' in preceding:
                out.append(f"{_A.GREEN}true{_A.RESET}")
            else:
                out.append(f"{_A.YELLOW}true{_A.RESET}")
            i += 4
        elif raw[i:i+5] == 'false':
            preceding = raw[max(0, i-20):i]
            if '"ok"' in preceding:
                out.append(f"{_A.RED}false{_A.RESET}")
            else:
                out.append(f"{_A.YELLOW}false{_A.RESET}")
            i += 5
        elif raw[i:i+4] == 'null':
            out.append(f"{_A.GREY}null{_A.RESET}")
            i += 4
        elif ch == '-' or ch.isdigit():
            j = i + 1
            while j < n and (raw[j].isdigit() or raw[j] in ('.', 'e', 'E', '+', '-')):
                j += 1
            out.append(f"{_A.MAGENTA}{raw[i:j]}{_A.RESET}")
            i = j
        else:
            out.append(ch)
            i += 1
    return ''.join(out)


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
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.handlers.RotatingFileHandler:
    """Create a rotating file handler for slo.log v1 JSON output."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sloughgpt.log"

    handler = logging.handlers.RotatingFileHandler(
        str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(SloFormatter(fmt="json"))
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
    log_dir: Optional[str] = None,
    enable_file: Optional[bool] = None,
    enable_console: bool = True,
    enable_output_buffer: bool = True,
) -> dict[str, Any]:
    """Configure all logging with a single unified formatter.

    Console: colored human-readable (SloFormatter with colors).
    File: slo.log v1 JSON (SloFormatter without colors).

    Args:
        level:            Log level (default: SLO_LOG_LEVEL env or "INFO")
        log_dir:          Directory for log files (default: SLO_LOG_DIR env or "logs/")
        enable_file:      Enable file logging (default: SLO_LOG_NO_FILE != "1")
        enable_console:   Install console stderr handler (default: True)
        enable_output_buffer: Install OutputBuffer bridge (default: True)

    Returns:
        dict with setup info: {"level", "log_dir", "file_handler", "bridge"}
    """
    level_name = (level or os.environ.get("SLO_LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)
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

    for h in list(root.handlers):
        root.removeHandler(h)

    # Console handler — colored human-readable
    if enable_console:
        colors = _color_enabled(sys.stderr)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(SloFormatter(colors=colors))
        console_handler.addFilter(ClientExtensionFilter())
        root.addHandler(console_handler)

    # File handler — slo.log v1 JSON
    file_handler = None
    if use_file:
        try:
            file_handler = _create_file_handler(log_path, level=logging.DEBUG)
            root.addHandler(file_handler)
        except Exception as e:
            logging.getLogger(__name__).warning("Could not create log file handler: %s", e)

    # OutputBuffer bridge (for SSE streaming)
    bridge = None
    if enable_output_buffer:
        bridge = _install_output_buffer_bridge(root)

    # Dashboard event buffer filter
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
        "log_dir": str(log_path),
        "file_handler": file_handler,
        "bridge": bridge,
    }
