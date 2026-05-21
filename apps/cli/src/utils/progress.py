"""
Progress bars and spinners for CLI operations.

Provides visual feedback for long-running operations.
"""
import sys
import time
import threading
from typing import Optional, Callable


class ProgressBar:
    """Simple text-based progress bar."""

    def __init__(
        self,
        total: int,
        desc: str = "",
        width: int = 40,
        show_eta: bool = True,
        show_speed: bool = False,
    ):
        """Initialize progress bar.

        Args:
            total: Total number of items
            desc: Description text
            width: Bar width in characters
            show_eta: Show estimated time remaining
            show_speed: Show items per second
        """
        self.total = total
        self.desc = desc
        self.width = width
        self.show_eta = show_eta
        self.show_speed = show_speed
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0.0
        self._update_interval = 0.1  # Update every 100ms

    def update(self, n: int = 1):
        """Increment progress by n."""
        self.current = min(self.current + n, self.total)
        now = time.time()
        if now - self.last_update >= self._update_interval:
            self._render()
            self.last_update = now

    def set_progress(self, current: int):
        """Set current progress directly."""
        self.current = min(current, self.total)
        self._render()

    def finish(self):
        """Complete progress bar."""
        self.current = self.total
        self._render()
        print()  # New line after completion

    def _render(self):
        """Render progress bar to terminal."""
        if self.total == 0:
            return

        pct = self.current / self.total
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)

        parts = [f"\r{self.desc}"] if self.desc else ["\r"]
        parts.append(f"|{bar}| {pct * 100:5.1f}%")
        parts.append(f"{self.current}/{self.total}")

        elapsed = time.time() - self.start_time

        if self.show_speed and elapsed > 0:
            speed = self.current / elapsed
            parts.append(f"[{speed:.1f}/s]")

        if self.show_eta and self.current > 0 and elapsed > 0:
            eta = (self.total - self.current) / (self.current / elapsed)
            parts.append(f"[{self._format_time(eta)}]")

        # Pad to clear previous longer output
        line = " ".join(parts)
        sys.stdout.write(line + " " * (10))
        sys.stdout.flush()

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.0f}m {seconds % 60:.0f}s"
        else:
            return f"{seconds / 3600:.1f}h"


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
