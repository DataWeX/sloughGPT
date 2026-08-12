"""
Progress bars and spinners for CLI operations.

Provides visual feedback for long-running operations.
Uses Rich for terminal-formatted progress bars with ETA, speed, and color.
"""
import os
import sys
import time
import threading
from typing import Optional, Callable


def _is_terminal() -> bool:
    """Check if stdout is a real terminal (not piped/redirected)."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class ProgressBar:
    """Terminal progress bar with dotted fill, ETA, and speed.

    Uses space-padding to overwrite previous output instead of ANSI escape
    sequences, which avoids scroll/flicker in shells and captured output.
    """

    FILLED = "█"
    EMPTY = "░"
    HALF = "▓"

    def __init__(
        self,
        total: int,
        desc: str = "",
        width: int = 40,
        show_eta: bool = True,
        show_speed: bool = False,
    ):
        self.total = total
        self.desc = desc
        self.width = width
        self.show_eta = show_eta
        self.show_speed = show_speed
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0.0
        self._update_interval = 0.1
        self._last_rendered = ""
        self._last_pct = -1
        self._last_desc = ""
        self._is_tty = _is_terminal()

    def update(self, n: int = 1):
        self.current = min(self.current + n, self.total)
        now = time.time()
        if now - self.last_update >= self._update_interval:
            self._render()
            self.last_update = now

    def set_progress(self, current: int):
        self.current = min(current, self.total)
        self._render()

    def finish(self):
        self.current = self.total
        self._render()
        if self._is_tty:
            # Clear the progress bar line and move to next line
            sys.stdout.write(f"\r\033[2K{self._last_rendered}\n")
            sys.stdout.flush()
        self._last_rendered = ""

    def _render(self):
        if self.total == 0:
            return

        pct = self.current / self.total
        pct_int = int(pct * 100)

        # Dedup: skip if nothing changed (same pct + same desc)
        if pct_int == self._last_pct and self.desc == self._last_desc:
            return
        self._last_pct = pct_int
        self._last_desc = self.desc

        # Build bar
        filled_width = int(self.width * pct)
        has_half = (self.width * pct) - filled_width >= 0.5
        bar = self.FILLED * filled_width
        if has_half and filled_width < self.width:
            bar += self.HALF
            bar += self.EMPTY * (self.width - filled_width - 1)
        else:
            bar += self.EMPTY * (self.width - filled_width)

        elapsed = time.time() - self.start_time
        parts = []

        if self.desc:
            parts.append(self.desc)

        parts.append(f"[{bar}] {pct_int:3d}%")

        if self.current > 0 and self.total > 0:
            parts.append(f"{self.current}/{self.total}")

        if self.show_speed and elapsed > 0:
            speed = self.current / elapsed
            parts.append(f"{speed:.1f}/s")

        if self.show_eta and self.current > 0 and self.total > 0 and elapsed > 0:
            eta = (self.total - self.current) / (self.current / elapsed)
            parts.append(f"eta {self._format_time(eta)}")

        parts.append(f"({self._format_time(elapsed)})")

        line = " ".join(parts)

        if self._is_tty:
            # TTY: use ANSI escape to clear line, then print
            # \033[2K clears the entire line
            # \r moves cursor to start of line
            sys.stdout.write(f"\r\033[2K{line}")
            sys.stdout.flush()
        else:
            # Non-TTY (piped/redirected): just print the line
            print(line)

        self._last_rendered = line

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m {s:02d}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m:02d}m"


class Spinner:
    """Animated spinner for indeterminate operations."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, text: str = "", interval: float = 0.08):
        self.text = text
        self.interval = interval
        self._running = False
        self._thread = None
        self._last_rendered = ""

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self, message: str = "Done"):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        # Clear spinner line with space padding
        if self._last_rendered:
            pad = " " * len(self._last_rendered)
            sys.stdout.write(f"\r{pad}\r")
            sys.stdout.flush()
        if message:
            print(message)

    def _animate(self):
        i = 0
        while self._running:
            frame = self.FRAMES[i % len(self.FRAMES)]
            line = f"{frame} {self.text}" if self.text else frame
            # Pad to overwrite previous longer line
            pad_len = max(0, len(self._last_rendered) - len(line))
            sys.stdout.write(f"\r{line}{' ' * pad_len}\r")
            sys.stdout.flush()
            self._last_rendered = line
            time.sleep(self.interval)
            i += 1


def progress_iter(
    iterable,
    total: Optional[int] = None,
    desc: str = "",
    width: int = 40,
) -> ProgressBar:
    """Wrap an iterable with a progress bar.

    Args:
        iterable: Iterable to wrap
        total: Total count (auto-detected if iterable has __len__)
        desc: Description text
        width: Bar width

    Yields:
        Items from iterable with progress updates
    """
    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)

    bar = ProgressBar(total=total or 0, desc=desc, width=width)

    try:
        for item in iterable:
            yield item
            bar.update()
    finally:
        bar.finish()
