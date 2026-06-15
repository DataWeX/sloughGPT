"""
Standard I/O abstraction for shell — one writer, one reader, no raw escapes.

Provides a clean interface over the terminal so that the REPL, CLI, and TUI
share the same output model. All ANSI escapes, cursor movement, pagination,
and layout are encapsulated here.
"""

import os
import sys
import shutil
import signal
import threading
from contextlib import contextmanager
from typing import IO, Callable, Optional


# ── Colour & Style helpers ──────────────────────────────────────────────

_NO_COLOR = os.environ.get("NO_COLOR", "").strip() == "1"


def _detect_color() -> bool:
    if _NO_COLOR:
        return False
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "").lower()
    if "256" in term or "truecolor" in term or term in ("xterm", "xterm-256color", "screen-256color"):
        return True
    return bool(shutil.get_terminal_size().columns >= 80)


_COLOR_CAPS = _detect_color()


class Ansi:
    """ANSI escape builder — all methods return escape strings."""

    @staticmethod
    def reset() -> str:
        return "\033[0m" if _COLOR_CAPS else ""

    @staticmethod
    def bold() -> str:
        return "\033[1m" if _COLOR_CAPS else ""

    @staticmethod
    def dim() -> str:
        return "\033[2m" if _COLOR_CAPS else ""

    @staticmethod
    def italic() -> str:
        return "\033[3m" if _COLOR_CAPS else ""

    @staticmethod
    def fg(code: int) -> str:
        return f"\033[38;5;{code}m" if _COLOR_CAPS else ""

    @staticmethod
    def bg(code: int) -> str:
        return f"\033[48;5;{code}m" if _COLOR_CAPS else ""

    @staticmethod
    def rgb(r: int, g: int, b: int) -> str:
        return f"\033[38;2;{r};{g};{b}m" if _COLOR_CAPS else ""

    @staticmethod
    def black() -> str:     return "\033[30m" if _COLOR_CAPS else ""
    @staticmethod
    def red() -> str:       return "\033[31m" if _COLOR_CAPS else ""
    @staticmethod
    def green() -> str:     return "\033[32m" if _COLOR_CAPS else ""
    @staticmethod
    def yellow() -> str:    return "\033[33m" if _COLOR_CAPS else ""
    @staticmethod
    def blue() -> str:      return "\033[34m" if _COLOR_CAPS else ""
    @staticmethod
    def magenta() -> str:   return "\033[35m" if _COLOR_CAPS else ""
    @staticmethod
    def cyan() -> str:      return "\033[36m" if _COLOR_CAPS else ""
    @staticmethod
    def white() -> str:     return "\033[37m" if _COLOR_CAPS else ""

    @staticmethod
    def grey() -> str:      return "\033[90m" if _COLOR_CAPS else ""

    @classmethod
    def ok(cls) -> str:         return cls.green()
    @classmethod
    def warn(cls) -> str:       return cls.yellow()
    @classmethod
    def err(cls) -> str:        return cls.red()
    @classmethod
    def info(cls) -> str:       return cls.cyan()
    @classmethod
    def muted(cls) -> str:      return cls.grey()
    @classmethod
    def highlight(cls) -> str:  return cls.bold() + cls.cyan()

    # -- Cursor & Screen --
    @staticmethod
    def cursor_up(n: int = 1) -> str:       return f"\033[{n}A"
    @staticmethod
    def cursor_down(n: int = 1) -> str:     return f"\033[{n}B"
    @staticmethod
    def cursor_forward(n: int = 1) -> str:  return f"\033[{n}C"
    @staticmethod
    def cursor_back(n: int = 1) -> str:     return f"\033[{n}D"
    @staticmethod
    def cursor_save() -> str:               return "\033[s"
    @staticmethod
    def cursor_restore() -> str:            return "\033[u"
    @staticmethod
    def cursor_hide() -> str:               return "\033[?25l"
    @staticmethod
    def cursor_show() -> str:               return "\033[?25h"
    @staticmethod
    def cursor_col(col: int) -> str:        return f"\033[{col}G"
    @staticmethod
    def cursor_pos(row: int, col: int) -> str: return f"\033[{row};{col}H"

    @staticmethod
    def erase_line() -> str:                return "\033[2K\r"
    @staticmethod
    def erase_down() -> str:                return "\033[J"
    @staticmethod
    def erase_screen() -> str:              return "\033[2J\033[H"

    @staticmethod
    def scroll_up(n: int = 1) -> str:       return f"\033[{n}S"
    @staticmethod
    def scroll_down(n: int = 1) -> str:     return f"\033[{n}T"


# ── Terminal Info ───────────────────────────────────────────────────────

class TerminalInfo:
    """Snapshot of terminal dimensions and capabilities."""

    def __init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        size = shutil.get_terminal_size()
        self.width: int = size.columns
        self.height: int = size.lines
        self.color: bool = _COLOR_CAPS
        self.is_tty: bool = sys.stdout.isatty()


# ── Output Types ────────────────────────────────────────────────────────

class OutputLine:
    """A single line of output with style metadata."""

    __slots__ = ("text", "style", "indent")

    def __init__(self, text: str = "", style: str = "", indent: int = 0) -> None:
        self.text = text
        self.style = style       # ANSI prefix
        self.indent = indent

    def render(self, width: int, color: bool = True) -> str:
        prefix = " " * self.indent
        line = prefix + self.text
        if color and self.style:
            return self.style + line + Ansi.reset()
        return line

    def __repr__(self) -> str:
        return f"OutputLine({self.text[:30]!r})"


class OutputBuffer:
    """Scrollable buffer of OutputLines with viewport tracking."""

    def __init__(self, max_lines: int = 5000) -> None:
        self._lines: list[OutputLine] = []
        self._max = max_lines
        self._view_top = 0        # first visible line index
        self._view_height = 0     # number of visible lines
        self._lock = threading.Lock()

    def append(self, line: OutputLine) -> None:
        with self._lock:
            self._lines.append(line)
            if len(self._lines) > self._max:
                excess = len(self._lines) - self._max
                self._lines = self._lines[excess:]
                self._view_top = max(0, self._view_top - excess)

    def append_text(self, text: str, style: str = "", indent: int = 0) -> None:
        self.append(OutputLine(text, style, indent))

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
            self._view_top = 0

    @property
    def lines(self) -> list[OutputLine]:
        return self._lines

    @property
    def count(self) -> int:
        return len(self._lines)

    def scroll(self, delta: int) -> None:
        with self._lock:
            max_top = max(0, len(self._lines) - self._view_height)
            self._view_top = max(0, min(self._view_top + delta, max_top))

    def scroll_to_bottom(self) -> None:
        with self._lock:
            self._view_top = max(0, len(self._lines) - self._view_height)

    @property
    def visible_lines(self) -> list[OutputLine]:
        with self._lock:
            return self._lines[self._view_top:self._view_top + self._view_height]

    def set_viewport(self, height: int) -> None:
        self._view_height = max(1, height)
        with self._lock:
            max_top = max(0, len(self._lines) - self._view_height)
            self._view_top = max(0, min(self._view_top, max_top))


# ── StdioWriter ─────────────────────────────────────────────────────────

class StdioWriter:
    """Unified terminal writer with buffering, pagination, and overlays.

    Handles:
    - Plain text output (auto-newline)
    - Raw escape sequences (for progress bars, inline updates)
    - Paged output (buffered, scrollable viewport)
    - Overlays (status bar, prompts) that don't scroll the buffer

    Note: ``stream`` is stored as a fixed reference at init. If you
    need temp capture (e.g. for tests), set ``_stream`` directly or
    use ``captured_output()``.
    """

    def __init__(self, stream: Optional[IO] = None) -> None:
        self._original_stdout = sys.stdout
        self._stream: IO = stream or sys.stdout
        self._stream_id = id(sys.stdout)  # detect if sys.stdout gets swapped
        self._term = TerminalInfo()
        self._buffer = OutputBuffer()
        self._pager_active = False
        self._overlay_lines: list[str] = []   # lines drawn outside buffer (status bar, etc.)
        self._last_line_count = 0
        self._lock = threading.Lock()

    @property
    def _output_stream(self) -> IO:
        """Resolve output stream — follows sys.stdout if it was swapped.

        Only auto-follows sys.stdout if the stored stream was the original
        sys.stdout. If someone explicitly set ``_stream`` to a different
        object (e.g. ``captured_output``), respect that choice.
        """
        if self._stream is self._original_stdout and id(sys.stdout) != self._stream_id:
            self._stream_id = id(sys.stdout)
            self._stream = sys.stdout
        return self._stream

    # ── Public API ──

    def print(self, text: str = "", style: str = "", indent: int = 0, end: str = "\n") -> None:
        """Write a line of text to the terminal (and buffer for replay)."""
        with self._lock:
            line = OutputLine(text, style, indent)
            self._buffer.append(line)
            rendered = line.render(self._term.width) + end
            self._output_stream.write(rendered)
            self._output_stream.flush()

    def raw(self, text: str) -> None:
        """Write raw text without buffering (progress bars, cursor moves)."""
        with self._lock:
            self._output_stream.write(text)
            self._output_stream.flush()

    def overlay(self, lines: list[str]) -> None:
        """Draw overlay lines that sit above normal output (status bar, etc.).
        Call ``clear_overlay()`` before next normal print to avoid ghosting.
        """
        with self._lock:
            self._clear_overlay_locked()
            for line in lines:
                self._output_stream.write(Ansi.erase_line() + line + "\n")
            self._output_stream.flush()
            self._overlay_lines = lines

    def clear_overlay(self) -> None:
        """Remove overlay lines from terminal."""
        with self._lock:
            self._clear_overlay_locked()

    def _clear_overlay_locked(self) -> None:
        if self._overlay_lines:
            n = len(self._overlay_lines)
            self._output_stream.write(Ansi.cursor_up(n))
            for _ in range(n):
                self._output_stream.write(Ansi.erase_line())
                self._output_stream.write(Ansi.cursor_down(1))
            self._output_stream.write(Ansi.cursor_up(n))
            self._output_stream.flush()
            self._overlay_lines = []

    def progress(self, text: str, done: bool = False) -> None:
        """Render an in-place progress line (overwrites current line)."""
        with self._lock:
            end = "\n" if done else ""
            self._output_stream.write(Ansi.erase_line() + text + end)
            self._output_stream.flush()

    def clear_screen(self) -> None:
        """Clear terminal and reset cursor."""
        with self._lock:
            self._output_stream.write(Ansi.erase_screen())
            self._output_stream.flush()

    def bell(self) -> None:
        """Terminal bell (^G)."""
        self._output_stream.write("\a")
        self._output_stream.flush()

    # ── Pager ──

    def paged_print(self, lines: list[OutputLine], page_size: Optional[int] = None) -> None:
        """Print lines with paging — pauses every *page_size* lines.

        This is a blocking call that reads from stdin.
        """
        self._term.refresh()
        if page_size is None:
            page_size = self._term.height - 3

        pos = 0
        total = len(lines)
        while pos < total:
            chunk = lines[pos:pos + page_size]
            for out_line in chunk:
                self.print(out_line.text, out_line.style, out_line.indent)
            pos += page_size

            if pos < total and self._term.is_tty:
                self.raw(
                    Ansi.muted()
                    + f"  ── more ({pos}/{total} lines, space/Enter=next, q=quit) ──"
                    + Ansi.reset()
                )
                try:
                    key = self._read_one()
                except (EOFError, KeyboardInterrupt):
                    break
                self.raw(Ansi.erase_line() + Ansi.cursor_up(1) + Ansi.erase_line())
                if key in ("q", "Q"):
                    break
                if key == "\x1b":  # Escape
                    break

    # ── Info ──

    @property
    def width(self) -> int:
        self._term.refresh()
        return self._term.width

    @property
    def height(self) -> int:
        self._term.refresh()
        return self._term.height

    @property
    def buffer(self) -> OutputBuffer:
        return self._buffer

    def refresh_term(self) -> None:
        self._term.refresh()

    # ── Helpers ──

    @staticmethod
    def _read_one() -> str:
        """Read a single keypress from stdin (blocking)."""
        fd = sys.stdin.fileno()
        old = None
        try:
            import termios
            old = termios.tcgetattr(fd)
            new = termios.tcgetattr(fd)
            new[3] &= ~(termios.ECHO | termios.ICANON)
            termios.tcsetattr(fd, termios.TCSADRAIN, new)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = ch + sys.stdin.read(2) if select_stdin(0.05) else ch
                return seq
            return ch
        except (ImportError, termios.error):
            return sys.stdin.readline(1)
        finally:
            if old is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                except Exception:
                    pass


# ── StdioReader ─────────────────────────────────────────────────────────

class StdioReader:
    """Keyboard input reader with readline-style editing, history, completion.

    Wraps the builtin ``input()`` with readline integration when available,
    and falls back to plain ``input()`` otherwise.
    """

    def __init__(self, history_file: str = "") -> None:
        self._history_file = history_file
        self._completer: Optional[Callable[[str], list[str]]] = None
        self._setup_readline()

    def _setup_readline(self) -> None:
        try:
            import readline  # type: ignore[import-untyped]
            if self._history_file:
                try:
                    readline.read_history_file(self._history_file)
                except FileNotFoundError:
                    pass
                readline.set_history_length(500)
            if self._completer:
                readline.set_completer(self._completer)
                readline.parse_and_bind("tab: complete")
                readline.parse_and_bind('"\\C-r": reverse-search-history')
                readline.parse_and_bind('"\\C-s": forward-search-history')
        except ImportError:
            pass

    def set_completer(self, completer: Callable[[str, int], Optional[str]]) -> None:
        """Set readline completer (``readline.set_completer`` signature)."""
        self._completer = completer
        try:
            import readline  # type: ignore[import-untyped]
            readline.set_completer(completer)
            readline.parse_and_bind("tab: complete")
        except ImportError:
            pass

    def read(self, prompt: str = "") -> str:
        """Read a line of input, return stripped result."""
        try:
            line = input(prompt)
            return line.strip()
        except (EOFError, KeyboardInterrupt):
            raise

    def read_multiline(self, prompt: str, cont_prompt: str = "> ") -> str:
        """Read a potentially multi-line input (continuation on backslash)."""
        lines: list[str] = []
        while True:
            raw = self.read(prompt if not lines else cont_prompt)
            if raw.endswith("\\"):
                lines.append(raw[:-1].rstrip())
                continue
            lines.append(raw)
            break
        return " ".join(lines)

    def save_history(self) -> None:
        """Persist readline history to disk."""
        if not self._history_file:
            return
        try:
            import readline  # type: ignore[import-untyped]
            readline.write_history_file(self._history_file)
        except Exception:
            pass

    def wait_key(self, prompt: str = "") -> str:
        """Blocking single-key read — returns the key string."""
        sys.stdout.write(prompt)
        sys.stdout.flush()
        try:
            return StdioWriter._read_one()
        except (EOFError, KeyboardInterrupt):
            return ""


# ── Context managers ────────────────────────────────────────────────────

@contextmanager
def captured_output(writer: StdioWriter) -> Callable[[], str]:
    """Capture all ``print()`` calls into a string, restoring the writer.

    Usage::
        with captured_output(writer) as get:
            some_function_that_uses_print()
            output = get()
    """
    import io
    buf = io.StringIO()
    old = writer._stream
    writer._stream = buf
    try:
        def get() -> str:
            return buf.getvalue()
        yield get
    finally:
        writer._stream = old


@contextmanager
def hidden_cursor(stream: IO = sys.stdout) -> None:
    """Temporarily hide the cursor."""
    stream.write(Ansi.cursor_hide())
    stream.flush()
    try:
        yield
    finally:
        stream.write(Ansi.cursor_show())
        stream.flush()


# ── Helpers ──

def select_stdin(timeout: float = 0) -> bool:
    """Check if stdin has data available to read."""
    import select
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return bool(r)


# ── Format helpers ──

def format_table(
    rows: list[list[str]],
    headers: Optional[list[str]] = None,
    separator: str = "  ",
) -> list[OutputLine]:
    """Format tabular data as a list of OutputLines with aligned columns."""
    if not rows and not headers:
        return []
    col_count = max(len(r) for r in rows) if rows else len(headers)
    col_widths = [0] * col_count
    all_rows: list[list[str]] = []
    if headers:
        all_rows.append(headers)
    all_rows.extend(rows)
    for row in all_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    out: list[OutputLine] = []
    for row in all_rows:
        padded = []
        for i, cell in enumerate(row):
            w = col_widths[i]
            padded.append(str(cell).ljust(w))
        out.append(OutputLine(separator.join(padded).rstrip()))
    return out


def truncate(text: str, max_len: int = 80, ellipsis: str = "…") -> str:
    """Truncate text to *max_len* with ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[:max_len - len(ellipsis)] + ellipsis
