"""
fmt — formatted terminal output that never corrupts the input line.

Provides structured output primitives (tables, boxes, separators, status
indicators) with an optional readline buffer save/restore so background
output never corrupts the user's in-progress input.

Usage:
  from .fmt import Fmt
  out = Fmt(io, has_readline=True)
  out.table(data, header=["A", "B"])
  out.box("Hello")
  out.sep()
  out.status("ok", "Task complete")
"""

from __future__ import annotations

import os
import shutil
from typing import Any


_COLOR_ENABLED = not os.environ.get("NO_COLOR")
if _COLOR_ENABLED:
    _C_CYAN = "\033[36m"
    _C_GREEN = "\033[32m"
    _C_YELLOW = "\033[33m"
    _C_RED = "\033[31m"
    _C_DIM = "\033[2m"
    _C_BOLD = "\033[1m"
    _C_RESET = "\033[0m"
else:
    _C_CYAN = _C_GREEN = _C_YELLOW = _C_RED = _C_DIM = _C_BOLD = _C_RESET = ""


def _color(text: str, code: str) -> str:
    return f"{code}{text}{_C_RESET}" if _COLOR_ENABLED and code else text


class Fmt:
    """Formatted terminal output with input-buffer preservation.

    Args:
        io: A ShellIO-compatible object (ConsoleIO, MemoryIO, etc.).
        has_readline: If True, saves/restores the readline buffer around
            output to avoid corrupting the user's in-progress input.
    """

    def __init__(self, io: Any, has_readline: bool = False) -> None:
        self._io = io
        self._has_readline = has_readline
        if has_readline:
            try:
                import readline  # noqa: F401
            except ImportError:
                self._has_readline = False

    # ── Input-safe write ─────────────────────────────────────────────

    def write(self, text: str, end: str = "\n") -> None:
        """Write text, preserving the readline input buffer if active."""
        if not self._has_readline or not text.strip():
            self._io.write(text, end=end)
            return

        try:
            import readline
            buf = readline.get_line_buffer()
        except (ImportError, RuntimeError):
            self._io.write(text, end=end)
            return

        if not buf:
            self._io.write(text, end=end)
            return

        lines_out = text.count("\n") + (1 if end == "\n" else 0)
        self._io.write("\033[s", end="")
        self._io.write(f"\033[{lines_out}B", end="")
        self._io.write(text, end=end)
        self._io.write("\033[u", end="")
        self._io.write(buf, end="")

    # ── Print raw text ───────────────────────────────────────────────

    def println(self, *args, **kwargs) -> None:
        """Print arguments separated by spaces."""
        end = kwargs.get("end", "\n")
        text = " ".join(str(a) for a in args)
        self.write(text, end=end)

    # ── Separator ────────────────────────────────────────────────────

    def sep(self, char: str = "─", color: str = _C_DIM) -> str:
        """Full-width horizontal separator. Returns the line."""
        cols = shutil.get_terminal_size().columns
        line = char * cols
        rendered = _color(line, color) if color else line
        self._io.write(rendered)
        return rendered

    # ── Section header ───────────────────────────────────────────────

    def section(self, title: str, char: str = "─", width: int | None = None) -> None:
        """Print a section header: '── title ──'."""
        cols = width or shutil.get_terminal_size().columns
        avail = cols - 2
        inner = f" {title} "
        side_len = max(0, (avail - len(inner)) // 2)
        side = char * side_len
        self._io.write(_color(f"{side}{inner}{side}", _C_DIM))

    # ── Box ──────────────────────────────────────────────────────────

    def box(self, text: str, width: int | None = None) -> None:
        """Draw a box around text using Unicode box-drawing characters."""
        cols = width or shutil.get_terminal_size().columns
        inner_w = cols - 4

        self._io.write("  " + _color("┌" + "─" * inner_w + "┐", _C_DIM))

        for line in text.split("\n"):
            padded = line.ljust(inner_w)
            self._io.write(f"  {_C_DIM}│{_C_RESET} {padded} {_C_DIM}│{_C_RESET}")

        self._io.write("  " + _color("└" + "─" * inner_w + "┘", _C_DIM))

    # ── Status indicators ────────────────────────────────────────────

    def status(self, kind: str, message: str, detail: str = "") -> None:
        """Colored status line. kind: ok|warn|error|info|step."""
        symbols = {
            "ok": _color("✓", _C_GREEN),
            "warn": _color("⚠", _C_YELLOW),
            "error": _color("✗", _C_RED),
            "info": _color("ℹ", _C_CYAN),
            "step": _color("→", _C_CYAN),
        }
        prefix = symbols.get(kind, kind)
        parts = [f"  {prefix} {message}"]
        if detail:
            parts.append(_color(f"({detail})", _C_DIM))
        self._io.write(" ".join(parts))

    # ── Table ────────────────────────────────────────────────────────

    def table(self, rows: list[list[str]], header: list[str] | None = None,
              sep_header: bool = True) -> None:
        """Print a formatted table with aligned columns."""
        if not rows:
            self._io.write("  (empty)")
            return

        cols = max(len(r) for r in rows)
        if header:
            cols = max(cols, len(header))

        widths = [0] * cols
        for row in rows:
            for i, cell in enumerate(row):
                if i < cols:
                    widths[i] = max(widths[i], len(str(cell)))
        if header:
            for i, cell in enumerate(header):
                if i < cols:
                    widths[i] = max(widths[i], len(str(cell)))

        fmt = "  ".join("{{:<{}}}".format(w) for w in widths)

        if header:
            self._io.write(f"  {fmt.format(*header)}")
            if sep_header:
                self._io.write(f"  {'  '.join('─' * w for w in widths)}")
        for row in rows:
            padded = list(row) + [""] * (cols - len(row))
            self._io.write(f"  {fmt.format(*padded)}")

    # ── Key-value list ───────────────────────────────────────────────

    def kv(self, items: list[tuple[str, str]], indent: int = 2) -> None:
        """Aligned key-value list."""
        if not items:
            return
        max_k = max(len(k) for k, _ in items)
        pad = " " * indent
        for k, v in items:
            self._io.write(f"{pad}{k:<{max_k}}  {v}")

    # ── Progress bar ─────────────────────────────────────────────────

    def progress(self, label: str, current: int, total: int,
                 bar_width: int = 20) -> None:
        """Overwritable progress bar with \\r."""
        frac = current / max(total, 1)
        filled = int(frac * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = f"{frac * 100:5.1f}%"
        self._io.write(f"\r  {label}: [{bar}] {pct} ({current}/{total})", end="")
        if current >= total:
            self._io.write("")

    # ── Error with hint ──────────────────────────────────────────────

    def err(self, message: str, hint: str = "") -> None:
        """Red error with optional dim hint below."""
        self._io.write(f"  {_color('Error:', _C_RED)} {message}")
        if hint:
            self._io.write(f"    {_color(hint, _C_DIM)}")

    # ── Success ──────────────────────────────────────────────────────

    def ok(self, message: str) -> None:
        """Green checkmark + message."""
        self._io.write(f"  {_color('✓', _C_GREEN)} {message}")

    # ── Info ─────────────────────────────────────────────────────────

    def info(self, message: str) -> None:
        """Cyan info prefix + message."""
        self._io.write(f"  {_color('ℹ', _C_CYAN)} {message}")
