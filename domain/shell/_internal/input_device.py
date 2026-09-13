"""
InputDevice — standalone input hardware.

User input operations with clean ioctl interface.
"""

from __future__ import annotations

import sys
import select
import termios
import tty
from typing import Any

from .kernel_syscall import SyscallResult


class InputDevice:
    """Standalone input hardware — user input operations.

    Has clean ioctl interface for assembly.
    Has function calls for direct use.
    """

    def __init__(self, name: str = "input"):
        self._name = name
        self._ops = {
            "READLINE": self._readline,
            "READCHAR": self._readchar,
            "READKEY": self._readkey,
            "READLINE_ECHO": self._readline_echo,
            "POLL": self._poll,
            "FLUSH": self._flush,
            "INFO": self._info,
        }

    @property
    def name(self) -> str:
        return self._name

    def info(self) -> dict:
        return {
            "name": self._name,
            "type": "input",
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

    def readline(self) -> str:
        """Read line from stdin."""
        return sys.stdin.readline().rstrip("\n")

    def readchar(self) -> str:
        """Read single character."""
        return sys.stdin.read(1)

    def readkey(self) -> str:
        """Read single key (with escape sequences)."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    return f"\x1b[{ch3}"
                return ch + ch2
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def readline_echo(self) -> str:
        """Read line with echo."""
        line = ""
        while True:
            ch = self.readchar()
            if ch == "\n" or ch == "\r":
                print()
                return line
            elif ch == "\x7f" or ch == "\x08":
                if line:
                    line = line[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif ch == "\x03":
                raise KeyboardInterrupt
            else:
                line += ch
                sys.stdout.write(ch)
                sys.stdout.flush()

    def poll(self, timeout: float = 0.0) -> bool:
        """Check if input is available."""
        if select.select([sys.stdin], [], [], timeout)[0]:
            return True
        return False

    def flush(self) -> bool:
        """Flush input buffer."""
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        return True

    # ── Private methods (ioctl handlers) ──────────────────────────────────

    def _readline(self, *args):
        return self.readline()

    def _readchar(self, *args):
        return self.readchar()

    def _readkey(self, *args):
        return self.readkey()

    def _readline_echo(self, *args):
        return self.readline_echo()

    def _poll(self, *args):
        timeout = args[0] if len(args) > 0 else 0.0
        return self.poll(timeout)

    def _flush(self, *args):
        return self.flush()

    def _info(self, *args):
        return self.info()
