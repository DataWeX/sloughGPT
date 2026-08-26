"""
ShellIO — compact standard I/O operations manager.

Provides a single abstraction over terminal read/write that the REPL,
CLI, and TUI share.  Swappable: ``ConsoleIO`` for CLI, ``TuiIO`` for
textual, ``MemoryIO`` for tests.

The manager is intentionally thin — it owns exactly three operations:
  - write(text, end)    → send output
  - read(prompt)        → get a line of input
  - capture()           → redirect writes into a string buffer

Everything else (parsing, expansion, command dispatch) stays in ShellREPL.
"""

from __future__ import annotations

import logging
import sys
from typing import Callable, Optional, Protocol

logger = logging.getLogger("slo.shell.io")


# ── Protocol ────────────────────────────────────────────────────────


class ShellIO(Protocol):
    """Abstract I/O interface — swap for TUI, tests, etc."""

    def write(self, text: str, end: str = "\n") -> None: ...
    def read(self, prompt: str = "") -> str: ...
    def flush(self) -> None: ...


# ── Console (OS-level terminal) ─────────────────────────────────────


class ConsoleIO:
    """OS-level terminal I/O via /dev/tty.

    Opens /dev/tty for direct terminal access when available.
    write() sends to /dev/tty (or stdout as fallback).
    read() reads from /dev/tty (or input() as fallback).

    Does NOT redirect sys.stdin/sys.stdout — subprocesses and
    command chains keep their original stdin/stdout intact.
    Only the REPL's own read/write go through the tty.
    """

    def __init__(self) -> None:
        self._tty = None
        self._has_readline = False
        self._uses_stdin = False

        try:
            self._tty = open("/dev/tty", "r+", buffering=1)
        except OSError:
            self._tty = None

        # Fallback: if stdin is a TTY, use stdin/stdout directly
        if self._tty is None and sys.stdin.isatty():
            self._uses_stdin = True

        try:
            import readline  # noqa: F401
            self._has_readline = True
        except ImportError:
            pass

    @property
    def _is_tty(self) -> bool:
        return self._tty is not None or self._uses_stdin

    def write(self, text: str, end: str = "\n") -> None:
        if self._tty:
            self._tty.write(text + end)
            self._tty.flush()
        else:
            sys.stdout.write(text + end)
            sys.stdout.flush()

    def read(self, prompt: str = "") -> str:
        if self._tty:
            self._tty.write(prompt)
            self._tty.flush()
            return self._tty.readline().strip()
        return input(prompt).strip()

    def flush(self) -> None:
        if self._tty:
            self._tty.flush()
        else:
            sys.stdout.flush()

    def setup_completion(self, completer: Callable[[str, int], Optional[str]]) -> None:
        """Wire readline tab completion."""
        if not self._has_readline:
            return
        try:
            import readline
            readline.set_completer(completer)
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind('"\\C-r": reverse-search-history')
            readline.parse_and_bind('"\\C-s": forward-search-history')
        except Exception as e:
            logger.debug("readline setup failed: %s", e)

    def save_history(self, path: str) -> None:
        if not self._has_readline:
            return
        try:
            import readline
            readline.write_history_file(path)
        except Exception as e:
            logger.debug("readline history save failed: %s", e)

    def load_history(self, path: str) -> None:
        if not self._has_readline:
            return
        try:
            import readline
            readline.read_history_file(path)
            readline.set_history_length(500)
        except FileNotFoundError:
            pass

    def __del__(self) -> None:
        """Close the TTY handle if not already closed."""
        self.close()

    def __enter__(self) -> "ConsoleIO":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._tty is not None:
            try:
                self._tty.close()
            except Exception as e:
                logger.debug("tty close failed: %s", e)
            self._tty = None


# ── Memory (tests / TUI) ───────────────────────────────────────────


class MemoryIO:
    """In-memory I/O for testing and programmatic use."""

    def __init__(self) -> None:
        self._output: list[str] = []
        self._inputs: list[str] = []
        self._input_idx = 0
        self._is_tty: bool = False

    def write(self, text: str, end: str = "\n") -> None:
        self._output.append(text + end if text else end)

    def read(self, prompt: str = "") -> str:
        if self._input_idx < len(self._inputs):
            line = self._inputs[self._input_idx]
            self._input_idx += 1
            return line.strip()
        raise EOFError

    def flush(self) -> None:
        pass

    def get_output(self) -> str:
        return "".join(self._output)

    def clear(self) -> None:
        self._output.clear()

    def feed(self, *lines: str) -> None:
        """Pre-load input lines for scripted execution."""
        self._inputs.extend(lines)
        self._input_idx = 0


# ── Capture (output redirection) ───────────────────────────────────


class _Capture:
    """Context manager: redirect all writes to a string buffer."""

    def __init__(self, io_shell: ShellIO):
        self._io = io_shell
        self._buf: list[str] = []
        self._old_write: Callable | None = None

    def __enter__(self) -> "_Capture":
        self._buf.clear()
        self._old_write = self._io.write
        self._io.write = self._write_to_buf  # type: ignore
        return self

    def __exit__(self, *exc):
        if self._old_write:
            self._io.write = self._old_write

    def _write_to_buf(self, text: str, end: str = "\n") -> None:
        self._buf.append(text + end if text else end)

    def getvalue(self) -> str:
        return "".join(self._buf)


# ── Convenience ─────────────────────────────────────────────────────


def capture_output(io: ShellIO) -> _Capture:
    """Shorthand: ``with capture_output(io) as cap: ...``"""
    return _Capture(io)


def capture_cmd(repl, method, *args) -> str:
    """Call a command method, capture output via MemoryIO, return the string."""
    mem = MemoryIO()
    old_io = repl.io
    old_console_io = repl.console._io
    old_interactive_io = repl.console._interactive._io
    old_interactive_tty = repl.console._interactive._is_tty
    repl.io = mem
    repl.console._io = mem
    repl.console._interactive._io = mem
    repl.console._interactive._is_tty = False
    try:
        method(*args)
    finally:
        repl.io = old_io
        repl.console._io = old_console_io
        repl.console._interactive._io = old_interactive_io
        repl.console._interactive._is_tty = old_interactive_tty
    return mem.get_output()
