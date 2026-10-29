"""The progress_* family. — InteractivePrompt mixin.

Every progress_* variant: the plain bar plus its coloured, ETA, striped, gradient, segmented, multi-segment, animated, indeterminate and step forms, plus the progress_* entry points.
"""

from __future__ import annotations

import sys
import time

from domain.shell._internal.interactive_base import InteractiveBase
from domain.shell._internal.interactive_table import _TablePrompts
from domain.shell._internal.interactive_term import (
    _BOLD,
    _CYAN,
    _DIM,
    _GREEN,
    _HIDE_CURSOR,
    _RED,
    _RESET,
    _SHOW_CURSOR,
    _YELLOW,
)


class _ProgressPrompts(_TablePrompts, InteractiveBase):
    """The progress_* family..

    Split out of InteractivePrompt by .splitwork/split_interactive.py.
    """

    def progress_step(self, steps: list[str], current: int, done: bool = False) -> None:
        """Display a step-by-step progress indicator."""
        for i, step in enumerate(steps):
            if done or i < current:
                icon = f"{_GREEN}\u2714{_RESET}"
            elif i == current:
                icon = f"{_CYAN}\u25cf{_RESET}"
            else:
                icon = f"{_DIM}\u25cb{_RESET}"
            self._io.write(f"  {icon} {step}")

    def progress_eta(self, label: str, current: int, total: int, elapsed: float = 0) -> None:
        """Display a progress bar with estimated time remaining."""
        frac = current / max(total, 1)
        bar_w = 25
        filled = int(frac * bar_w)
        bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
        pct = f"{frac * 100:5.1f}%"
        if current > 0 and elapsed > 0:
            remaining = (total - current) / (current / elapsed)
            if remaining >= 3600:
                eta = f"{int(remaining // 3600)}h{int((remaining % 3600) // 60)}m"
            elif remaining >= 60:
                eta = f"{int(remaining // 60)}m{int(remaining % 60)}s"
            else:
                eta = f"{remaining:.0f}s"
            eta_str = f"  {_DIM}ETA {eta}{_RESET}"
        else:
            eta_str = ""
        self._io.write(f"  {_CYAN}{label}{_RESET} {_DIM}{bar}{_RESET} {pct}{eta_str}")

    def progress_ring(self, label: str, current: int, total: int) -> None:
        """Display a circular progress indicator with Unicode blocks.

        Args:
            label: label text
            current: current progress value
            total: total target value
        """
        frac = current / max(total, 1)
        blocks = [
            "\u2581",
            "\u2582",
            "\u2583",
            "\u2584",
            "\u2585",
            "\u2586",
            "\u2587",
            "\u2588",
        ]
        idx = int(frac * (len(blocks) - 1))
        ring = blocks[idx] * 8
        pct = f"{frac * 100:.0f}%"
        self._io.write(f"  {_CYAN}{label}{_RESET} {_BOLD}{ring}{_RESET} {pct}")

    def progress_bar(self, label: str, current: int, total: int, width: int = 30) -> None:
        """Display a progress bar with percentage.

        Args:
            label: label text
            current: current progress value
            total: total target value
            width: bar width in characters
        """
        frac = current / max(total, 1)
        filled = int(frac * width)
        empty = width - filled
        bar = "\u2588" * filled + "\u2591" * empty
        pct = f"{frac * 100:.0f}%"
        self._io.write(f"  {_CYAN}{label}{_RESET} {_BOLD}{bar}{_RESET} {pct}")

    def progress_bar_colored(self, label: str, current: int, total: int, width: int = 30) -> None:
        """Display a colored progress bar (green→yellow→red based on progress).

        Args:
            label: label text
            current: current progress value
            total: total target value
            width: bar width in characters
        """
        frac = current / max(total, 1)
        filled = int(frac * width)
        empty = width - filled
        if frac < 0.33:
            color = _RED
        elif frac < 0.66:
            color = _YELLOW
        else:
            color = _GREEN
        bar = f"{color}{'█' * filled}{_DIM}{'░' * empty}{_RESET}"
        pct = f"{frac * 100:.0f}%"
        self._io.write(f"  {_CYAN}{label}{_RESET} {bar} {pct}")

    def progress_bar_eta(
        self, label: str, current: int, total: int, elapsed: float, width: int = 30
    ) -> None:
        """Display a progress bar with percentage and ETA.

        Args:
            label: label text
            current: current progress value
            total: total target value
            elapsed: elapsed time in seconds
            width: bar width in characters
        """
        frac = current / max(total, 1)
        filled = int(frac * width)
        empty = width - filled
        bar = "\u2588" * filled + "\u2591" * empty
        pct = f"{frac * 100:.0f}%"
        if current > 0 and elapsed > 0:
            rate = current / elapsed
            remaining = (total - current) / rate
            if remaining < 60:
                eta = f"{remaining:.0f}s"
            elif remaining < 3600:
                eta = f"{remaining / 60:.0f}m"
            else:
                eta = f"{remaining / 3600:.1f}h"
        else:
            eta = "?"
        self._io.write(
            f"  {_CYAN}{label}{_RESET} {_BOLD}{bar}{_RESET} {pct} {_DIM}ETA: {eta}{_RESET}"
        )

    def progress_bar_with_status(
        self, label: str, current: int, total: int, status: str = "", width: int = 30
    ) -> None:
        """Display a progress bar with percentage and status message.

        Args:
            label: label text
            current: current progress value
            total: total target value
            status: status text to display after bar
            width: bar width in characters
        """
        frac = current / max(total, 1)
        filled = int(frac * width)
        empty = width - filled
        bar = "\u2588" * filled + "\u2591" * empty
        pct = f"{frac * 100:.0f}%"
        status_str = f"  {_DIM}{status}{_RESET}" if status else ""
        self._io.write(f"  {_CYAN}{label}{_RESET} {_BOLD}{bar}{_RESET} {pct}{status_str}")

    def progress_bar_indeterminate(self, label: str, status: str = "") -> None:
        """Display an indeterminate progress bar (no known total).

        Args:
            label: label text
            status: status text to display
        """
        if not self._is_tty:
            self._io.write(f"  {label}... {status}")
            return
        self._progress_bar_indeterminate_raw(label, status)

    def _progress_bar_indeterminate_raw(self, label: str, status: str) -> None:
        frames = [
            "\u2581",
            "\u2582",
            "\u2583",
            "\u2584",
            "\u2585",
            "\u2586",
            "\u2587",
            "\u2588",
            "\u2587",
            "\u2586",
            "\u2585",
            "\u2584",
            "\u2583",
            "\u2582",
        ]
        pos = 0
        for _ in range(20):
            bar = " " * pos + frames[_ % len(frames)] + " " * (20 - pos)
            status_str = f"  {_DIM}{status}{_RESET}" if status else ""
            sys.stdout.write(
                f"\r  {_CYAN}{label}{_RESET} {_BOLD}{bar}{_RESET}{status_str} {_HIDE_CURSOR}"
            )
            sys.stdout.flush()
            time.sleep(0.1)
            pos = (pos + 1) % 21
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    def progress_bar_stripe(self, label: str, current: int, total: int, width: int = 30) -> None:
        """Display a striped progress bar.

        Args:
            label: label text
            current: current progress value
            total: total target value
            width: bar width in characters
        """
        frac = current / max(total, 1)
        filled = int(frac * width)
        empty = width - filled
        stripes = ["\u2592", "\u2593"]
        bar = ""
        for i in range(filled):
            bar += stripes[i % 2]
        bar += "\u2591" * empty
        pct = f"{frac * 100:.0f}%"
        self._io.write(f"  {_CYAN}{label}{_RESET} {_BOLD}{bar}{_RESET} {pct}")

    def progress_bar_gradient(self, label: str, current: int, total: int, width: int = 30) -> None:
        """Display a gradient-colored progress bar (red→yellow→green).

        Args:
            label: label text
            current: current progress value
            total: total target value
            width: bar width in characters
        """
        frac = current / max(total, 1)
        filled = int(frac * width)
        empty = width - filled
        gradient_blocks = [
            "\u2581",
            "\u2582",
            "\u2583",
            "\u2584",
            "\u2585",
            "\u2586",
            "\u2587",
            "\u2588",
        ]
        bar = ""
        for i in range(filled):
            block_frac = i / max(width, 1)
            if block_frac < 0.33:
                bar += f"{_RED}{gradient_blocks[min(int(block_frac * 24), 7)]}"
            elif block_frac < 0.66:
                bar += f"{_YELLOW}{gradient_blocks[min(int(block_frac * 12), 7)]}"
            else:
                bar += f"{_GREEN}{gradient_blocks[min(int(block_frac * 8), 7)]}"
        bar += f"{_RESET}{'░' * empty}"
        pct = f"{frac * 100:.0f}%"
        self._io.write(f"  {_CYAN}{label}{_RESET} {bar} {pct}")

    def progress_bar_segmented(
        self, label: str, segments: list[tuple[str, int]], width: int = 30
    ) -> None:
        """Display a segmented progress bar with different colors per segment.

        Args:
            label: label text
            segments: list of (color_name, value) tuples
                      color_name: "green", "yellow", "red", "cyan"
            width: bar width in characters
        """
        total = sum(v for _, v in segments)
        colors = {
            "green": _GREEN,
            "yellow": _YELLOW,
            "red": _RED,
            "cyan": _CYAN,
            "magenta": "\033[35m",
        }
        bar = ""
        for name, val in segments:
            frac = val / max(total, 1)
            filled = int(frac * width)
            color = colors.get(name, _CYAN)
            bar += f"{color}{'█' * filled}"
        remaining = width - sum(int(v / max(total, 1) * width) for _, v in segments)
        bar += f"{_DIM}{'░' * max(remaining, 0)}{_RESET}"
        self._io.write(f"  {_CYAN}{label}{_RESET} {bar}")

    def progress_bar_multi_segment(
        self, label: str, segments: list[tuple[str, int, str]], width: int = 30
    ) -> None:
        """Display a multi-segment progress bar with labels."""
        total = sum(v for _, v, _ in segments)
        colors_map = {
            "green": _GREEN,
            "yellow": _YELLOW,
            "red": _RED,
            "cyan": _CYAN,
            "magenta": "\033[35m",
            "blue": "\033[34m",
        }
        bar = ""
        for _name, val, color in segments:
            frac = val / max(total, 1)
            filled = int(frac * width)
            c = colors_map.get(color, _CYAN)
            bar += f"{c}{'█' * filled}"
        remaining = width - sum(int(v / max(total, 1) * width) for _, v, _ in segments)
        bar += f"{_DIM}{'░' * max(remaining, 0)}{_RESET}"
        labels = " ".join(f"{colors_map.get(c, _CYAN)}{n}{_RESET}: {v}" for n, v, c in segments)
        self._io.write(f"  {_CYAN}{label}{_RESET} {bar}\n  {_DIM}{labels}{_RESET}")

    def progress_bar_animated(self, label: str, current: int, total: int, width: int = 30) -> None:
        """Display an animated shimmer progress bar."""
        frac = current / max(total, 1)
        filled = int(frac * width)
        empty = width - filled
        shimmer = ["\u2591", "\u2592", "\u2593"]
        bar = ""
        for i in range(filled):
            bar += f"{_CYAN}{shimmer[i % 3]}"
        bar += f"{_DIM}{'░' * empty}{_RESET}"
        pct = f"{frac * 100:.0f}%"
        self._io.write(f"  {_CYAN}{label}{_RESET} {bar} {pct}")

    def progress_bar_with_status_and_eta(
        self,
        label: str,
        current: int,
        total: int,
        status: str,
        width: int = 30,
        elapsed: float = 0.0,
    ) -> None:
        """Display a progress bar with status text and ETA.

        Args:
            label: progress label
            current: current progress value
            total: total progress value
            status: status text to display
            width: bar width in characters
            elapsed: elapsed time in seconds (for ETA calculation)
        """
        frac = current / max(total, 1)
        filled = int(frac * width)
        empty = width - filled
        bar = f"{_CYAN}{'█' * filled}{_DIM}{'░' * empty}{_RESET}"
        pct = f"{frac * 100:.0f}%"
        eta_display = ""
        if current > 0 and elapsed > 0:
            eta_sec = elapsed / current * (total - current)
            if eta_sec > 60:
                eta_str = f"{int(eta_sec // 60)}m{int(eta_sec % 60)}s"
            else:
                eta_str = f"{int(eta_sec)}s"
            eta_display = f"  {_DIM}ETA {eta_str}{_RESET}"
        status_display = f"  {_DIM}{status}{_RESET}" if status else ""
        self._io.write(f"  {_CYAN}{label}{_RESET} {bar} {pct}{status_display}{eta_display}")

    def progress_bar_with_eta_and_status(
        self,
        label: str,
        current: int,
        total: int,
        status: str,
        elapsed: float = 0.0,
        width: int = 30,
    ) -> None:
        """Display a progress bar with ETA and status text."""
        frac = current / max(total, 1)
        filled = int(frac * width)
        empty = width - filled
        bar = f"{_CYAN}{'█' * filled}{_DIM}{'░' * empty}{_RESET}"
        pct = f"{frac * 100:.0f}%"
        eta_display = ""
        if current > 0 and elapsed > 0:
            eta_sec = elapsed / current * (total - current)
            if eta_sec > 3600:
                eta_str = f"{int(eta_sec // 3600)}h{int((eta_sec % 3600) // 60)}m"
            elif eta_sec > 60:
                eta_str = f"{int(eta_sec // 60)}m{int(eta_sec % 60)}s"
            else:
                eta_str = f"{int(eta_sec)}s"
            eta_display = f"  {_DIM}ETA {eta_str}{_RESET}"
        status_display = f"  {_YELLOW}{status}{_RESET}" if status else ""
        self._io.write(f"  {_CYAN}{label}{_RESET} {bar} {pct}{status_display}{eta_display}")

    def progress_bar_with_steps(
        self, label: str, steps: list[str], current_step: int, width: int = 30
    ) -> None:
        """Display a multi-step progress bar."""
        total = len(steps)
        frac = (current_step + 1) / max(total, 1)
        filled = int(frac * width)
        empty = width - filled
        bar = f"{_CYAN}{'█' * filled}{_DIM}{'░' * empty}{_RESET}"
        step_text = steps[current_step] if current_step < total else ""
        step_display = f"  {_YELLOW}{step_text}{_RESET}" if step_text else ""
        completed = f"  {_DIM}{current_step + 1}/{total}{_RESET}"
        self._io.write(f"  {_CYAN}{label}{_RESET} {bar}{completed}{step_display}")
