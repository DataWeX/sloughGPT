"""
Progress bars and spinners for CLI operations.

Provides visual feedback for long-running operations.
Uses Rich for terminal-formatted progress bars with ETA, speed, and color.
"""
import sys
import time
import threading
from typing import Optional, Callable


class ProgressBar:
    """Rich-backed progress bar with dotted fill, ETA, and speed."""

    # Dotted fill characters (popular in modern CLIs like pip, npm, cargo)
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
        self._last_pct = -1

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
        print()

    def _render(self):
        if self.total == 0:
            return

        pct = self.current / self.total
        pct_int = int(pct * 100)

        # Only re-render if percentage changed (avoids flicker)
        if pct_int == self._last_pct:
            return
        self._last_pct = pct_int

        # Build bar with half-block for sub-character precision
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
            parts.append(f"  {self.desc}")

        parts.append(f"[{bar}] {pct_int:3d}%")

        if self.current > 0 and self.total > 0:
            parts.append(f"{self.current}/{self.total}")

        if self.show_speed and elapsed > 0:
            speed = self.current / elapsed
            parts.append(f"{speed:.1f}/s")

        if self.show_eta and self.current > 0 and self.total > 0 and elapsed > 0:
            eta = (self.total - self.current) / (self.current / elapsed)
            parts.append(f"eta {self._format_time(eta)}")

        parts.append(f"({self._format_time(elapsed)} elapsed)")

        line = " ".join(parts)
        sys.stdout.write(f"\r\033[K{line}")
        sys.stdout.flush()

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
        """Initialize spinner.

        Args:
            text: Text to display alongside spinner
            interval: Update interval in seconds
        """
        self.text = text
        self.interval = interval
        self._running = False
        self._thread = None

    def start(self):
        """Start spinner animation."""
        self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self, message: str = "Done"):
        """Stop spinner and print final message."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        # Clear line and print message
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()
        if message:
            print(message)

    def _animate(self):
        """Animate spinner frames."""
        i = 0
        while self._running:
            frame = self.FRAMES[i % len(self.FRAMES)]
            if self.text:
                sys.stdout.write(f"\r{frame} {self.text}")
            else:
                sys.stdout.write(f"\r{frame}")
            sys.stdout.flush()
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
