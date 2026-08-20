"""
RichLogger — my own Rich-powered logger.

Build your own, that's the point.

Usage::

    from domains.logging.rich_logger import RichLogger

    log = RichLogger("slo.api")
    log.info("server started", port=8000)
    log.success("model loaded")
    log.warning("cache miss")
    log.error("inference failed", exc=True)
    log.step("downloading weights")
"""

from __future__ import annotations

import sys
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, Optional


# ── Log Level ──────────────────────────────────────────────────────────

class Level(Enum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3


# ── ANSI Colors ────────────────────────────────────────────────────────

class _C:
    """ANSI escape codes — no Rich dependency for basic output."""
    RESET   = "\033[0m"
    DIM     = "\033[2m"
    BOLD    = "\033[1m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    BG_RED  = "\033[41m"


# ── Icons & Styles per Level ───────────────────────────────────────────

_LEVEL_CONFIG = {
    Level.DEBUG:  (_C.DIM + _C.CYAN,    "·", _C.DIM + _C.CYAN),
    Level.INFO:   (_C.GREEN,             "ℹ", _C.BLUE),
    Level.WARN:   (_C.BOLD + _C.YELLOW,  "!", _C.YELLOW),
    Level.ERROR:  (_C.BOLD + _C.RED,     "✗", _C.RED),
}


# ── Log Record ─────────────────────────────────────────────────────────

@dataclass
class Record:
    level: Level
    message: str
    logger: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    elapsed_ms: Optional[float] = None


# ── Rich Logger ────────────────────────────────────────────────────────

class RichLogger:
    """My own Rich-based logger. ANSI colors, icons, context, timing."""

    def __init__(
        self,
        name: str = "slo",
        level: Level = Level.DEBUG,
        stream=None,
    ) -> None:
        self._name = name
        self._level = level
        self._stream = stream or sys.stderr
        self._lock = threading.Lock()
        self._context: Dict[str, Any] = {}

    # ── Config ──────────────────────────────────────────────────────────

    @property
    def level(self) -> Level:
        return self._level

    @level.setter
    def level(self, value: Level) -> None:
        self._level = value

    def set_context(self, **kwargs: Any) -> None:
        self._context.update(kwargs)

    def clear_context(self) -> None:
        self._context.clear()

    def child(self, suffix: str) -> "RichLogger":
        return RichLogger(
            name=f"{self._name}.{suffix}",
            level=self._level,
            stream=self._stream,
        )

    # ── Emit ────────────────────────────────────────────────────────────

    def _emit(self, record: Record) -> None:
        if record.level.value < self._level.value:
            return

        style, icon, icon_style = _LEVEL_CONFIG.get(
            record.level, (_C.WHITE, "·", _C.WHITE)
        )

        parts = []

        # Icon
        parts.append(f"  {icon} ")

        # Level tag
        parts.append(f"[{record.level.name.lower()}] ")
        parts[-1] = _colorize(parts[-1], style)

        # Logger name
        parts.append(f"{record.logger} ")
        parts[-1] = _colorize(parts[-1], _C.DIM)

        # Context
        if record.context:
            ctx_str = " ".join(f"{k}={v}" for k, v in record.context.items())
            parts.append(f"{ctx_str} ")
            parts[-1] = _colorize(parts[-1], _C.DIM)

        # Elapsed
        if record.elapsed_ms is not None:
            parts.append(f"({record.elapsed_ms:.0f}ms) ")
            parts[-1] = _colorize(parts[-1], _C.DIM)

        # Message
        parts.append(record.message)

        # Exception
        if record.exception:
            parts.append(f" — {record.exception}")
            parts[-1] = _colorize(parts[-1], _C.RED)

        line = "".join(parts)

        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    # ── Convenience Methods ─────────────────────────────────────────────

    def debug(self, msg: str, **ctx: Any) -> None:
        self._emit(Record(Level.DEBUG, msg, self._name, context={**self._context, **ctx}))

    def info(self, msg: str, **ctx: Any) -> None:
        self._emit(Record(Level.INFO, msg, self._name, context={**self._context, **ctx}))

    def warn(self, msg: str, **ctx: Any) -> None:
        self._emit(Record(Level.WARN, msg, self._name, context={**self._context, **ctx}))

    def warning(self, msg: str, **ctx: Any) -> None:
        self.warn(msg, **ctx)

    def error(self, msg: str, exc: bool = False, **ctx: Any) -> None:
        exc_str = None
        if exc:
            import traceback
            exc_str = traceback.format_exc().strip().split("\n")[-1]
        self._emit(Record(Level.ERROR, msg, self._name, context={**self._context, **ctx}, exception=exc_str))

    def success(self, msg: str, **ctx: Any) -> None:
        """Success — green checkmark."""
        line = f"  ✓ {msg}"
        if ctx:
            ctx_str = " ".join(f"{k}={v}" for k, v in ctx.items())
            line += f" {_colorize(ctx_str, _C.DIM)}"
        with self._lock:
            self._stream.write(_colorize(line, _C.GREEN) + "\n")
            self._stream.flush()

    def step(self, msg: str, **ctx: Any) -> None:
        """Step/action — cyan arrow."""
        line = f"  → {msg}"
        if ctx:
            ctx_str = " ".join(f"{k}={v}" for k, v in ctx.items())
            line += f" {_colorize(ctx_str, _C.DIM)}"
        with self._lock:
            self._stream.write(_colorize(line, _C.CYAN) + "\n")
            self._stream.flush()

    def header(self, title: str, char: str = "=") -> None:
        """Bold header with separator."""
        width = 80
        with self._lock:
            self._stream.write(_colorize(title, _C.BOLD) + "\n")
            self._stream.write(_colorize(char * width, _C.DIM) + "\n")
            self._stream.flush()

    def section(self, title: str) -> None:
        """Section divider."""
        width = 80
        with self._lock:
            self._stream.write("\n")
            self._stream.write(_colorize(title, _C.BOLD) + "\n")
            self._stream.write(_colorize("-" * width, _C.DIM) + "\n")
            self._stream.flush()

    def blank(self, count: int = 1) -> None:
        with self._lock:
            for _ in range(count):
                self._stream.write("\n")
            self._stream.flush()

    def key_value(self, key: str, value: str, indent: int = 2) -> None:
        padding = " " * indent
        line = f"{padding}{_colorize(key + ':', _C.DIM)} {value}"
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def status(self, label: str, value: str, ok: bool = True) -> None:
        icon = "✓" if ok else "✗"
        color = _C.GREEN if ok else _C.RED
        line = f"  {_colorize(icon, color)} {label}: {value}"
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    # ── Timing Context Manager ──────────────────────────────────────────

    @contextmanager
    def timer(self, label: str = "elapsed") -> Generator[None, None, None]:
        """Context manager that logs elapsed time on exit."""
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            self.info(f"{label}", elapsed_ms=elapsed_ms)


# ── Helper ─────────────────────────────────────────────────────────────

def _colorize(text: str, color: str) -> str:
    return f"{color}{text}{_C.RESET}"


# ── Singleton ──────────────────────────────────────────────────────────

_default: Optional[RichLogger] = None
_lock = threading.Lock()


def get_logger(name: str = "slo", level: Level = Level.DEBUG) -> RichLogger:
    global _default
    with _lock:
        if _default is None:
            _default = RichLogger(name=name, level=level)
        return _default
