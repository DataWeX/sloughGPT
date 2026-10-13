"""Screen-level control. — InteractivePrompt mixin.

Whole-screen helpers with no prompt semantics of their own: banner, clear, status, notify, and the progress/spinner wrappers. Last in the MRO because they are the thinnest.
"""

from __future__ import annotations

import sys

from domain.shell._internal.interactive_base import InteractiveBase
from domain.shell._internal.interactive_spinner import (
    _InteractiveSpinner,
    _SpinnerPrompts,
)
from domain.shell._internal.interactive_term import (
    _BOLD,
    _CYAN,
    _DIM,
    _ERASE_LINE,
    _GREEN,
    _RED,
    _RESET,
)


class _ScreenPrompts(_SpinnerPrompts, InteractiveBase):
    """Screen-level control..

    Split out of InteractivePrompt by .splitwork/split_interactive.py.
    """

    def banner(self, text: str, style: str = "double") -> None:
        """Display a styled banner with box-drawing characters.

        Args:
            text: the banner text
            style: border style - "single", "double", "thick", "dashed"
        """
        width = max(len(text) + 4, 20)
        styles = {
            "single": ("\u250c", "\u2500", "\u2510", "\u2502", "\u2514", "\u2518"),
            "double": ("\u2554", "\u2550", "\u2557", "\u2551", "\u255a", "\u255d"),
            "thick": ("\u250f", "\u2501", "\u2513", "\u2503", "\u2517", "\u251b"),
            "dashed": ("\u250c", "\u2504", "\u2510", "\u2502", "\u2514", "\u2518"),
        }
        tl, h, tr, v, bl, br = styles.get(style, styles["double"])
        inner = h * (width - 2)
        pad = " " * max(0, width - len(text) - 3)
        self._io.write(f"  {_BOLD}{_CYAN}{tl}{inner}{tr}{_RESET}")
        self._io.write(
            f"  {_BOLD}{_CYAN}{v}{_RESET} {_BOLD}{text}{_RESET}{pad}{_BOLD}{_CYAN}{v}{_RESET}"
        )
        self._io.write(f"  {_BOLD}{_CYAN}{bl}{inner}{br}{_RESET}")

    def notify(self, title: str, message: str, level: str = "info") -> None:
        """Display a styled notification banner.

        Args:
            title: notification title
            message: notification message
            level: "info", "success", "warn", "error"
        """
        icons = {
            "info": "\u2139",
            "success": "\u2714",
            "warn": "\u26a0",
            "error": "\u2716",
        }
        colors = {"info": _CYAN, "success": _GREEN, "warn": _RED, "error": _RED}
        icon = icons.get(level, "\u2139")
        color = colors.get(level, _CYAN)
        self._io.write(f"  {color}{_BOLD}{icon} {title}{_RESET}")
        if message:
            self._io.write(f"  {_DIM}  {message}{_RESET}")

    def progress_multi(self, items: list[tuple[str, int, int]]) -> None:
        """Display multiple progress bars stacked.

        Args:
            items: list of (label, current, total) tuples
        """
        for label, current, total in items:
            frac = current / max(total, 1)
            bar_w = 20
            filled = int(frac * bar_w)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            pct = f"{frac * 100:5.1f}%"
            self._io.write(f"  {_CYAN}{label}{_RESET} {_DIM}{bar}{_RESET} {pct}")

    def clear(self) -> None:
        """Clear the terminal screen."""
        if self._is_tty:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
        else:
            self._io.write("\033[2J\033[H", end="")

    def status(self, kind: str, message: str) -> None:
        """Print a colored status line.

        Args:
            kind: one of 'ok', 'warn', 'error', 'info', 'step'
            message: status text
        """
        symbols = {
            "ok": f"{_GREEN}\u2713{_RESET}",
            "warn": f"\033[33m\u26a0{_RESET}",
            "error": f"{_RED}\u2717{_RESET}",
            "info": f"{_CYAN}\u2139{_RESET}",
            "step": f"{_CYAN}\u2192{_RESET}",
        }
        symbol = symbols.get(kind, kind)
        line = f"  {symbol} {message}"
        if self._is_tty:
            sys.stdout.write(f"{_ERASE_LINE}{line}\n")
            sys.stdout.flush()
        else:
            self._io.write(line)

    def progress(self, label: str, current: int, total: int, bar_width: int = 20) -> None:
        """Show a progress bar. Overwrites the current line.

        Args:
            label: text label before the bar
            current: current progress value
            total: total value (100%)
            bar_width: width of the bar in characters
        """
        frac = current / max(total, 1)
        filled = int(frac * bar_width)
        bar = f"{_GREEN}{'█' * filled}{_DIM}{'░' * (bar_width - filled)}{_RESET}"
        pct = f"{frac * 100:5.1f}%"
        line = f"\r  {label}: [{bar}] {pct} ({current}/{total})"
        sys.stdout.write(f"{_ERASE_LINE}{line}")
        sys.stdout.flush()
        if current >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def spinner(self, message: str = "", rate: float = 0.1) -> _InteractiveSpinner:
        """Return a context manager that shows a spinner while a task runs.

        Usage::

            with prompt.spinner("Loading") as s:
                do_work()
            s.ok("done")
            s.fail("error")
        """
        return _InteractiveSpinner(self, message, rate)
