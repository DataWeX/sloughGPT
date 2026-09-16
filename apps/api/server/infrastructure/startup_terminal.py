"""Startup terminal visualization — prints boot progress to the console.

Usage:
    from infrastructure.startup_terminal import get_terminal_viz, terminal_hook

    viz = get_terminal_viz()
    viz.start()
    with terminal_hook("db_pool") as hook:
        await init_db_pool()
"""

import logging
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TextIO

logger = logging.getLogger(__name__)

# ANSI color codes
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "gray": "\033[90m",
}

STAGE_LABELS = {
    "INIT": "  Init",
    "CRITICAL": "Critical",
    "READY": "  Ready",
    "BACKGROUND": "  Bgnd ",
}

STATUS_SYMBOLS = {
    "pending": f"  {COLORS['gray']}·{COLORS['reset']}",
    "running": f"  {COLORS['cyan']}⟳{COLORS['reset']}",
    "ok": f"  {COLORS['green']}✓{COLORS['reset']}",
    "timeout": f"  {COLORS['yellow']}⏱{COLORS['reset']}",
    "error": f"  {COLORS['red']}✗{COLORS['reset']}",
}


@dataclass
class HookLine:
    """Terminal line for a hook."""

    name: str
    stage: str
    status: str = "pending"
    start_time: float = 0.0
    duration_ms: float = 0.0


class StartupTerminalViz:
    """Prints startup progress to the terminal."""

    def __init__(self, stream: TextIO | None = None, enabled: bool = True) -> None:
        self._stream = stream or sys.stderr
        self._enabled = enabled and self._stream.isatty()
        self._lines: list[HookLine] = []
        self._stage_name: str = "INIT"
        self._start_time: float = 0.0
        self._printed_count: int = 0

    def _write(self, text: str) -> None:
        if self._enabled:
            try:
                self._stream.write(text)
                self._stream.flush()
            except Exception:
                pass

    def _move_up(self, n: int) -> None:
        if n > 0 and self._enabled:
            self._write(f"\033[{n}A")

    def _clear_line(self) -> None:
        if self._enabled:
            self._write("\033[2K\r")

    def start(self) -> None:
        """Start the terminal visualization."""
        if not self._enabled:
            return

        self._start_time = time.monotonic()
        self._write(f"\n{COLORS['bold']}{COLORS['cyan']}⚡ sloughGPT Startup{COLORS['reset']}\n")
        self._write(f"{COLORS['dim']}{'─' * 40}{COLORS['reset']}\n")

    def set_stage(self, stage: str) -> None:
        """Update the current stage."""
        self._stage_name = stage
        label = STAGE_LABELS.get(stage, stage)
        elapsed = time.monotonic() - self._start_time if self._start_time else 0

        self._write(
            f"{COLORS['bold']}[{label}]{COLORS['reset']} "
            f"{COLORS['dim']}({elapsed:.1f}s){COLORS['reset']}\n"
        )
        self._printed_count += 1

    def add_hook(self, name: str, stage: str) -> None:
        """Register a hook for display."""
        line = HookLine(name=name, stage=stage)
        self._lines.append(line)

    def update_hook(self, name: str, status: str, duration_ms: float = 0) -> None:
        """Update a hook's status."""
        for line in self._lines:
            if line.name == name:
                line.status = status
                line.duration_ms = duration_ms
                break

        symbol = STATUS_SYMBOLS.get(status, "  ?")
        time.monotonic() - self._start_time if self._start_time else 0

        if status == "ok":
            color = COLORS["green"]
        elif status in ("timeout", "error"):
            color = COLORS["red"]
        elif status == "running":
            color = COLORS["cyan"]
        else:
            color = COLORS["gray"]

        duration_str = f" {duration_ms:.0f}ms" if duration_ms > 0 else ""
        self._write(
            f"  {symbol} {color}{name}{COLORS['reset']}"
            f"{COLORS['dim']}{duration_str}{COLORS['reset']}\n"
        )
        self._printed_count += 1

    def finish(self, success: bool = True) -> None:
        """Print the final summary."""
        if not self._enabled:
            return

        elapsed = time.monotonic() - self._start_time if self._start_time else 0

        self._write(f"{COLORS['dim']}{'─' * 40}{COLORS['reset']}\n")

        if success:
            ok_count = sum(1 for l in self._lines if l.status == "ok")
            fail_count = sum(1 for l in self._lines if l.status in ("error", "timeout"))
            self._write(
                f"{COLORS['green']}{COLORS['bold']}✓ Ready{COLORS['reset']}"
                f" {COLORS['dim']}in {elapsed:.1f}s{COLORS['reset']}"
                f" {COLORS['dim']}({ok_count} hooks"
                f"{f', {fail_count} failed' if fail_count else ''}){COLORS['reset']}\n"
            )
        else:
            self._write(
                f"{COLORS['red']}{COLORS['bold']}✗ Startup failed{COLORS['reset']}"
                f" {COLORS['dim']}after {elapsed:.1f}s{COLORS['reset']}\n"
            )


_global_viz: StartupTerminalViz | None = None


def get_terminal_viz() -> StartupTerminalViz:
    """Get the global terminal visualization."""
    global _global_viz
    if _global_viz is None:
        _global_viz = StartupTerminalViz()
    return _global_viz


@contextmanager
def terminal_hook(name: str) -> Generator[None, None, None]:
    """Context manager that updates terminal viz for a hook."""
    viz = get_terminal_viz()
    start = time.perf_counter()
    try:
        yield
        duration_ms = (time.perf_counter() - start) * 1000
        viz.update_hook(name, "ok", duration_ms)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        viz.update_hook(name, "error", duration_ms)
        raise
