"""
DisplayDevice — standalone display hardware.

Output operations with clean ioctl interface.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from .kernel_syscall import SyscallResult


class DisplayDevice:
    """Standalone display hardware — output operations.

    Has clean ioctl interface for assembly.
    Has function calls for direct use.
    """

    def __init__(self, name: str = "display"):
        self._name = name
        self._ops = {
            "PRINT": self._print,
            "PRINTLN": self._println,
            "CLEAR": self._clear,
            "MOVE": self._move,
            "COLOR": self._color,
            "STYLE": self._style,
            "FLUSH": self._flush,
            "WRITE": self._write,
            "INFO": self._info,
        }
        self._buffer: list[str] = []
        self._cursor_x: int = 0
        self._cursor_y: int = 0
        self._color: str = ""
        self._style: str = ""

    @property
    def name(self) -> str:
        return self._name

    def info(self) -> dict:
        return {
            "name": self._name,
            "type": "display",
            "buffer_size": len(self._buffer),
            "cursor": (self._cursor_x, self._cursor_y),
        }

    def call(self, method: str, *args: Any) -> Any:
        """VM Device interface — delegates to ioctl."""
        result = self.ioctl(method, *args)
        if result.success:
            return result.value
        raise Exception(result.error)

    # ── ioctl interface ───────────────────────────────────────────────────

    def ioctl(self, command: str, *args: Any) -> SyscallResult:
        """Clean ioctl interface — type-safe, documented."""
        try:
            fn = self._ops.get(command)
            if fn is None:
                return SyscallResult.fail(f"unknown command: {command}")
            result = fn(*args)
            return SyscallResult.ok(result)
        except Exception as e:
            return SyscallResult.fail(f"ioctl error: {e}")

    def list_commands(self) -> list[str]:
        """List all available commands."""
        return sorted(self._ops.keys())

    # ── Function calls (direct use) ───────────────────────────────────────

    def print_text(self, text: str) -> int:
        """Print text to display."""
        sys.stdout.write(text)
        sys.stdout.flush()
        return len(text)

    def println(self, text: str = "") -> int:
        """Print text with newline."""
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        return len(text) + 1

    def clear(self) -> bool:
        """Clear display."""
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        self._cursor_x = 0
        self._cursor_y = 0
        return True

    def move(self, x: int, y: int) -> bool:
        """Move cursor."""
        sys.stdout.write(f"\033[{y+1};{x+1}H")
        sys.stdout.flush()
        self._cursor_x = x
        self._cursor_y = y
        return True

    def color(self, fg: str = "", bg: str = "") -> bool:
        """Set color."""
        codes = {
            "black": "30", "red": "31", "green": "32", "yellow": "33",
            "blue": "34", "magenta": "35", "cyan": "36", "white": "37",
            "reset": "0",
        }
        if fg:
            sys.stdout.write(f"\033[{codes.get(fg, '37')}m")
        if bg:
            sys.stdout.write(f"\033[{codes.get(bg, '40')}m")
        sys.stdout.flush()
        return True

    def style(self, bold: bool = False, underline: bool = False) -> bool:
        """Set style."""
        if bold:
            sys.stdout.write("\033[1m")
        if underline:
            sys.stdout.write("\033[4m")
        sys.stdout.flush()
        return True

    def flush(self) -> bool:
        """Flush output."""
        sys.stdout.flush()
        return True

    def write(self, data: str) -> int:
        """Write raw data."""
        sys.stdout.write(data)
        return len(data)

    # ── Private methods (ioctl handlers) ──────────────────────────────────

    def _print(self, *args):
        return self.print_text(str(args[0]))

    def _println(self, *args):
        text = str(args[0]) if len(args) > 0 else ""
        return self.println(text)

    def _clear(self, *args):
        return self.clear()

    def _move(self, *args):
        return self.move(args[0], args[1])

    def _color(self, *args):
        fg = args[0] if len(args) > 0 else ""
        bg = args[1] if len(args) > 1 else ""
        return self.color(fg, bg)

    def _style(self, *args):
        bold = args[0] if len(args) > 0 else False
        underline = args[1] if len(args) > 1 else False
        return self.style(bold, underline)

    def _flush(self, *args):
        return self.flush()

    def _write(self, *args):
        return self.write(str(args[0]))

    def _info(self, *args):
        return self.info()
