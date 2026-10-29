"""The spinner_* family. — InteractivePrompt mixin.

Every spinner_* variant and _InteractiveSpinner, the one piece of shared mutable machinery the class has.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.shell._internal.interactive_base import InteractiveBase

if TYPE_CHECKING:
    import threading

    from domain.shell._internal.interactive_prompt import InteractivePrompt


import sys
import time

from domain.shell._internal.interactive_progress import _ProgressPrompts
from domain.shell._internal.interactive_term import (
    _CYAN,
    _DIM,
    _ERASE_LINE,
    _GREEN,
    _HIDE_CURSOR,
    _RED,
    _RESET,
    _SHOW_CURSOR,
)


class _SpinnerPrompts(_ProgressPrompts, InteractiveBase):
    """The spinner_* family..

    Split out of InteractivePrompt by .splitwork/split_interactive.py.
    """

    def spinner_with_status(self, message: str, status: str) -> None:
        """Display a spinner with a status message.

        Args:
            message: main message
            status: status text to show
        """
        if not self._is_tty:
            self._io.write(f"  {message}... {status}")
            return
        self._spinner_with_status_raw(message, status)

    def _spinner_with_status_raw(self, message: str, status: str) -> None:
        frames = ["\u250f", "\u2513", "\u251b", "\u2517"]
        for frame in frames:
            sys.stdout.write(
                f"\r  {_CYAN}{frame}{_RESET} {message}  {_DIM}{status}{_RESET} {_HIDE_CURSOR}"
            )
            sys.stdout.flush()
            time.sleep(0.15)
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    def spinner_with_progress(self, message: str, current: int, total: int) -> None:
        """Display a spinner with progress percentage.

        Args:
            message: main message
            current: current progress value
            total: total target value
        """
        if not self._is_tty:
            self._io.write(f"  {message}... {current}/{total}")
            return
        self._spinner_with_progress_raw(message, current, total)

    def _spinner_with_progress_raw(self, message: str, current: int, total: int) -> None:
        frames = ["\u250f", "\u2513", "\u251b", "\u2517"]
        frac = current / max(total, 1)
        pct = f"{frac * 100:.0f}%"
        for frame in frames:
            sys.stdout.write(
                f"\r  {_CYAN}{frame}{_RESET} {message}  {_DIM}{pct}{_RESET} {_HIDE_CURSOR}"
            )
            sys.stdout.flush()
            time.sleep(0.15)
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    def spinner_with_dots(self, message: str) -> None:
        """Display a spinner with dots animation.

        Args:
            message: main message
        """
        if not self._is_tty:
            self._io.write(f"  {message}...")
            return
        self._spinner_with_dots_raw(message)

    def _spinner_with_dots_raw(self, message: str) -> None:
        dots = ["", ".", "..", "..."]
        for dot in dots:
            sys.stdout.write(f"\r  {_CYAN}\u25f7{_RESET} {message}{dot}  {_HIDE_CURSOR}")
            sys.stdout.flush()
            time.sleep(0.25)
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    def spinner_with_eta(self, message: str, elapsed: float, progress: float = 0) -> None:
        """Display a spinner with elapsed time and optional ETA.

        Args:
            message: main message
            elapsed: elapsed time in seconds
            progress: current progress (0-1), used for ETA calculation
        """
        if not self._is_tty:
            self._io.write(f"  {message}... {elapsed:.1f}s")
            return
        self._spinner_with_eta_raw(message, elapsed, progress)

    def _spinner_with_eta_raw(self, message: str, elapsed: float, progress: float) -> None:
        frames = ["\u250f", "\u2513", "\u251b", "\u2517"]
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            time_str = f"{elapsed / 60:.1f}m"
        else:
            time_str = f"{elapsed / 3600:.1f}h"
        if progress > 0 and elapsed > 0:
            rate = progress / elapsed
            remaining = (1 - progress) / rate
            if remaining < 60:
                eta = f"{remaining:.0f}s"
            elif remaining < 3600:
                eta = f"{remaining / 60:.0f}m"
            else:
                eta = f"{remaining / 3600:.1f}h"
            eta_str = f" {_DIM}(ETA: {eta}){_RESET}"
        else:
            eta_str = ""
        for frame in frames:
            sys.stdout.write(
                f"\r  {_CYAN}{frame}{_RESET} {message}  {_DIM}{time_str}{_RESET}{eta_str} {_HIDE_CURSOR}"
            )
            sys.stdout.flush()
            time.sleep(0.15)
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    def spinner_with_dots_eta(self, message: str, elapsed: float, progress: float = 0) -> None:
        """Display a spinner with dots animation and ETA.

        Args:
            message: main message
            elapsed: elapsed time in seconds
            progress: current progress (0-1), used for ETA calculation
        """
        if not self._is_tty:
            self._io.write(f"  {message}... {elapsed:.1f}s")
            return
        self._spinner_with_dots_eta_raw(message, elapsed, progress)

    def _spinner_with_dots_eta_raw(self, message: str, elapsed: float, progress: float) -> None:
        frames = ["\u250f", "\u2513", "\u251b", "\u2517"]
        dots = ["", ".", "..", "..."]
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            time_str = f"{elapsed / 60:.1f}m"
        else:
            time_str = f"{elapsed / 3600:.1f}h"
        if progress > 0 and elapsed > 0:
            rate = progress / elapsed
            remaining = (1 - progress) / rate
            if remaining < 60:
                eta = f"{remaining:.0f}s"
            elif remaining < 3600:
                eta = f"{remaining / 60:.0f}m"
            else:
                eta = f"{remaining / 3600:.1f}h"
            eta_str = f" {_DIM}(ETA: {eta}){_RESET}"
        else:
            eta_str = ""
        for i in range(12):
            frame = frames[i % len(frames)]
            dot = dots[i % len(dots)]
            sys.stdout.write(
                f"\r  {_CYAN}{frame}{_RESET} {message}{dot}  {_DIM}{time_str}{_RESET}{eta_str} {_HIDE_CURSOR}"
            )
            sys.stdout.flush()
            time.sleep(0.2)
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    def spinner_pulse(self, message: str, duration: float = 2.0) -> None:
        """Display a pulsing dot spinner.

        Args:
            message: main message
            duration: how long to spin in seconds
        """
        if not self._is_tty:
            self._io.write(f"  {message}...")
            return
        self._spinner_pulse_raw(message, duration)

    def _spinner_pulse_raw(self, message: str, duration: float) -> None:
        dots = ["\u25cf", "\u25cb", "\u25cc", "\u25cd"]
        start = time.time()
        i = 0
        while time.time() - start < duration:
            sys.stdout.write(f"\r  {_CYAN}{dots[i % len(dots)]}{_RESET} {message}  {_HIDE_CURSOR}")
            sys.stdout.flush()
            time.sleep(0.3)
            i += 1
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    def spinner_wave(self, message: str, duration: float = 2.0) -> None:
        """Display a wave animation spinner.

        Args:
            message: main message
            duration: how long to spin in seconds
        """
        if not self._is_tty:
            self._io.write(f"  {message}...")
            return
        self._spinner_wave_raw(message, duration)

    def _spinner_wave_raw(self, message: str, duration: float) -> None:
        waves = ["\u2591", "\u2592", "\u2593", "\u2593", "\u2592", "\u2591"]
        start = time.time()
        i = 0
        while time.time() - start < duration:
            wave_str = ""
            for j in range(6):
                idx = (i + j) % len(waves)
                wave_str += waves[idx]
            sys.stdout.write(f"\r  {_CYAN}{wave_str}{_RESET} {message}  {_HIDE_CURSOR}")
            sys.stdout.flush()
            time.sleep(0.15)
            i += 1
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    def spinner_bounce(self, message: str, duration: float = 2.0) -> None:
        """Display a bouncing animation spinner."""
        if not self._is_tty:
            self._io.write(f"  {message}...")
            return
        self._spinner_bounce_raw(message, duration)

    def _spinner_bounce_raw(self, message: str, duration: float) -> None:
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
        start = time.time()
        i = 0
        while time.time() - start < duration:
            pos = i % len(frames)
            bar_left = " " * pos
            bar_right = " " * (len(frames) - pos - 1)
            sys.stdout.write(
                f"\r  {_CYAN}{bar_left}{frames[pos]}{bar_right}{_RESET} {message}  {_HIDE_CURSOR}"
            )
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    def spinner_clock(self, message: str, duration: float = 2.0) -> None:
        """Display a clock animation spinner."""
        if not self._is_tty:
            self._io.write(f"  {message}...")
            return
        self._spinner_clock_raw(message, duration)

    def _spinner_clock_raw(self, message: str, duration: float) -> None:
        clocks = ["\u25f4", "\u25f5", "\u25f6", "\u25f7"]
        start = time.time()
        i = 0
        while time.time() - start < duration:
            sys.stdout.write(
                f"\r  {_CYAN}{clocks[i % len(clocks)]}{_RESET} {message}  {_HIDE_CURSOR}"
            )
            sys.stdout.flush()
            time.sleep(0.25)
            i += 1
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    def spinner_with_messages(
        self, message: str, messages: list[str], duration: float = 3.0
    ) -> None:
        """Display a spinner that cycles through different messages."""
        if not self._is_tty:
            self._io.write(f"  {message}...")
            return
        self._spinner_with_messages_raw(message, messages, duration)

    def _spinner_with_messages_raw(
        self, message: str, messages: list[str], duration: float
    ) -> None:
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        start = time.time()
        i = 0
        while time.time() - start < duration:
            msg_idx = int((time.time() - start) / 2) % len(messages)
            sys.stdout.write(
                f"\r  {_CYAN}{frames[i % len(frames)]}{_RESET} {message}  {_DIM}{messages[msg_idx]}{_RESET}  {_HIDE_CURSOR}"
            )
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    def spinner_with_dots_and_status(
        self, message: str, status: str, duration: float = 2.0
    ) -> None:
        """Display a spinner with dots and status text."""
        if not self._is_tty:
            self._io.write(f"  {message}...")
            return
        self._spinner_with_dots_and_status_raw(message, status, duration)

    def _spinner_with_dots_and_status_raw(self, message: str, status: str, duration: float) -> None:
        dot_sets = [
            ("\u25f4", "\u25f5", "\u25f6", "\u25f7"),
            (".", "..", "...", "...."),
            ("\u2581", "\u2582", "\u2583", "\u2584"),
        ]
        dots = dot_sets[0]
        start = time.time()
        i = 0
        while time.time() - start < duration:
            sys.stdout.write(
                f"\r  {_CYAN}{dots[i % len(dots)]}{_RESET} {message}  {_DIM}{status}{_RESET}  {_HIDE_CURSOR}"
            )
            sys.stdout.flush()
            time.sleep(0.2)
            i += 1
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    def spinner_with_eta_message(self, message: str, total: int, duration: float = 3.0) -> None:
        """Display a spinner with ETA message."""
        if not self._is_tty:
            self._io.write(f"  {message}...")
            return
        self._spinner_with_eta_message_raw(message, total, duration)

    def _spinner_with_eta_message_raw(self, message: str, total: int, duration: float) -> None:
        frames = ["\u25f4", "\u25f5", "\u25f6", "\u25f7"]
        start = time.time()
        i = 0
        while time.time() - start < duration:
            elapsed = time.time() - start
            if elapsed > 0 and i > 0:
                eta = elapsed / i * (total - i)
                if eta > 60:
                    eta_str = f"{int(eta // 60)}m{int(eta % 60)}s"
                else:
                    eta_str = f"{int(eta)}s"
                eta_display = f" {_DIM}(ETA {eta_str}){_RESET}"
            else:
                eta_display = ""
            sys.stdout.write(
                f"\r  {_CYAN}{frames[i % len(frames)]}{_RESET} {message}{eta_display}  {_HIDE_CURSOR}"
            )
            sys.stdout.flush()
            time.sleep(0.25)
            i += 1
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()


class _InteractiveSpinner:
    """Context manager that drives a spinner animation for InteractivePrompt."""

    _FRAMES = [
        "\u280b",
        "\u2819",
        "\u2839",
        "\u2838",
        "\u283c",
        "\u2834",
        "\u2826",
        "\u2827",
        "\u2807",
        "\u280f",
    ]

    def __init__(self, prompt: InteractivePrompt, message: str, rate: float) -> None:
        self._prompt = prompt
        self._message = message
        self._rate = rate
        self._stop = False
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _InteractiveSpinner:
        import threading

        self._stop = False

        def _spin() -> None:
            i = 0
            while not self._stop:
                frame = self._FRAMES[i % len(self._FRAMES)]
                sys.stdout.write(f"\r  {_CYAN}{frame}{_RESET} {self._message}\r")
                sys.stdout.flush()
                i += 1
                time.sleep(self._rate)

        self._thread = threading.Thread(target=_spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop = True
        if self._thread:
            self._thread.join()
        sys.stdout.write(f"\r{_ERASE_LINE}")
        sys.stdout.flush()

    def ok(self, message: str = "") -> None:
        """Replace spinner with a success line."""
        self.__exit__()
        sys.stdout.write(f"\r  {_GREEN}\u2713{_RESET} {message or self._message}\n")
        sys.stdout.flush()

    def fail(self, message: str = "") -> None:
        """Replace spinner with a failure line."""
        self.__exit__()
        sys.stdout.write(f"\r  {_RED}\u2717{_RESET} {message or self._message}\n")
        sys.stdout.flush()
