"""
InteractivePrompt — arrow-key navigation and type-to-filter for line-mode.

Provides gh-auth-login-style interactive selection in the default
(readline-based) shell.  Falls back to a numbered menu when the terminal
does not support raw input (e.g. piped stdin, tests with MemoryIO).

Usage:
    from domains.shell.interactive import InteractivePrompt
    prompt = InteractivePrompt(io)
    choice = prompt.select("Pick an account", ["GitHub.com", "GitHub Enterprise"])
"""

from __future__ import annotations

import os
import sys
import termios
import time
import tty


# ── Raw key reading ──────────────────────────────────────────────────────────

_KEY_UP = "up"
_KEY_DOWN = "down"
_KEY_LEFT = "left"
_KEY_RIGHT = "right"
_KEY_ENTER = "enter"
_KEY_ESC = "esc"
_KEY_BACKSPACE = "backspace"
_KEY_DELETE = "delete"
_KEY_HOME = "home"
_KEY_END = "end"
_KEY_CTRL_C = "ctrl_c"
_KEY_CTRL_A = "ctrl_a"
_KEY_CTRL_E = "ctrl_e"
_KEY_PAGE_UP = "page_up"
_KEY_PAGE_DOWN = "page_down"
_KEY_CHAR = "char"


def _terminal_width() -> int:
    """Get terminal width in columns."""
    try:
        return os.get_terminal_size().columns
    except (AttributeError, ValueError, OSError):
        return 80


def _terminal_height() -> int:
    """Get terminal height in rows."""
    try:
        return os.get_terminal_size().lines
    except (AttributeError, ValueError, OSError):
        return 24


class _RawKey:
    """A decoded key press from the terminal."""

    def __init__(self, kind: str, char: str = "") -> None:
        self.kind = kind
        self.char = char

    def __repr__(self) -> str:
        if self.char:
            return f"RawKey({self.kind!r}, {self.char!r})"
        return f"RawKey({self.kind!r})"


def _read_raw_key(fd: int) -> _RawKey:
    """Read a single key press from a raw-mode file descriptor.

    Decodes escape sequences for arrow keys, and returns plain characters
    for printable input.  Blocks until a key is pressed.

    Args:
        fd: file descriptor for the terminal (e.g. sys.stdin.fileno())

    Returns:
        _RawKey with kind and optional char.
    """
    ch = os.read(fd, 1)
    if not ch:
        return _RawKey(_KEY_ESC)

    byte = ch[0]

    # Ctrl+C
    if byte == 3:
        return _RawKey(_KEY_CTRL_C)

    # Ctrl+A — Home
    if byte == 1:
        return _RawKey(_KEY_HOME)

    # Ctrl+E — End
    if byte == 5:
        return _RawKey(_KEY_END)

    # Enter
    if byte in (10, 13):
        return _RawKey(_KEY_ENTER)

    # Escape — read rest of sequence
    if byte == 27:
        seq = os.read(fd, 1)
        if not seq:
            return _RawKey(_KEY_ESC)
        if seq[0] == ord("["):
            code = os.read(fd, 1)
            if not code:
                return _RawKey(_KEY_ESC)
            c = code[0]
            if c == 65:  # A — Up
                return _RawKey(_KEY_UP)
            if c == 66:  # B — Down
                return _RawKey(_KEY_DOWN)
            if c == 67:  # C — Right
                return _RawKey(_KEY_RIGHT)
            if c == 68:  # D — Left
                return _RawKey(_KEY_LEFT)
            if c == 72 or c == 55:  # H or 7 — Home
                return _RawKey(_KEY_HOME)
            if c == 70 or c == 56:  # F or 8 — End
                return _RawKey(_KEY_END)
            if c == 51:  # 3 — could be Delete
                trailer = os.read(fd, 1)
                if trailer and trailer[0] == 126:
                    return _RawKey(_KEY_DELETE)
                return _RawKey(_KEY_ESC)
            if c == 49:  # 1 — could be Home (some terminals)
                trailer = os.read(fd, 1)
                if trailer and trailer[0] == 126:
                    return _RawKey(_KEY_HOME)
                return _RawKey(_KEY_ESC)
            if c == 53:  # 5 — Page Up
                trailer = os.read(fd, 1)
                if trailer and trailer[0] == 126:
                    return _RawKey(_KEY_PAGE_UP)
                return _RawKey(_KEY_ESC)
            if c == 54:  # 6 — Page Down
                trailer = os.read(fd, 1)
                if trailer and trailer[0] == 126:
                    return _RawKey(_KEY_PAGE_DOWN)
                return _RawKey(_KEY_ESC)
            if c == 52:  # 4 — could be End (some terminals)
                trailer = os.read(fd, 1)
                if trailer and trailer[0] == 126:
                    return _RawKey(_KEY_END)
                return _RawKey(_KEY_ESC)
            return _RawKey(_KEY_ESC)
        if seq[0] == ord("O"):
            code = os.read(fd, 1)
            if not code:
                return _RawKey(_KEY_ESC)
            if code[0] == 72:  # OH — Home
                return _RawKey(_KEY_HOME)
            if code[0] == 70:  # OF — End
                return _RawKey(_KEY_END)
            if code[0] == 77:  # M — mouse event, ignore
                os.read(fd, 1)
                os.read(fd, 1)
                return _RawKey(_KEY_ESC)
            return _RawKey(_KEY_ESC)
        return _RawKey(_KEY_ESC)

    # Backspace / Delete
    if byte in (127, 8):
        return _RawKey(_KEY_BACKSPACE)

    # Printable ASCII
    if 32 <= byte < 127:
        return _RawKey(_KEY_CHAR, chr(byte))

    # Non-ASCII (utf-8 multi-byte) — read remaining bytes
    if byte >= 128:
        remaining = []
        if byte < 224:
            count = 1
        elif byte < 240:
            count = 2
        elif byte < 248:
            count = 3
        else:
            count = 4
        for _ in range(count):
            more = os.read(fd, 1)
            if more:
                remaining.append(more[0])
        try:
            char = bytes([byte] + remaining).decode("utf-8", errors="replace")
        except Exception:
            char = ""
        return _RawKey(_KEY_CHAR, char)

    return _RawKey(_KEY_ESC)


# ── Terminal helpers ──────────────────────────────────────────────────────────

class _RawTerminal:
    """Context manager that puts the terminal in raw mode and restores it."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._old_attrs: list | None = None

    def __enter__(self) -> "_RawTerminal":
        try:
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setraw(self._fd)
        except termios.error:
            self._old_attrs = None
        return self

    def __exit__(self, *exc) -> None:
        if self._old_attrs is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            except termios.error:
                pass


# ── ANSI helpers ──────────────────────────────────────────────────────────────

_ERASE_LINE = "\033[2K"
_CURSOR_UP = "\033[1A"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_REVERSE = "\033[7m"
_RESET_REVERSE = "\033[27m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


# ── InteractivePrompt ─────────────────────────────────────────────────────────

class InteractivePrompt:
    """Interactive arrow-key prompts for line-mode shell.

    Wraps a ShellIO instance and provides select/confirm/ask methods that
    use raw terminal input when available, falling back to numbered menus
    for non-TTY environments (tests, piped input).
    """

    def __init__(self, io: object) -> None:
        self._io = io
        self._is_tty = self._detect_tty()

    def _detect_tty(self) -> bool:
        """Check if we can use raw terminal input."""
        if hasattr(self._io, "_is_tty"):
            return self._io._is_tty
        try:
            fd = sys.stdin.fileno()
            return os.isatty(fd)
        except (AttributeError, ValueError, OSError):
            return False

    def _get_fd(self) -> int:
        """Get the file descriptor for raw input."""
        if hasattr(self._io, "_tty") and self._io._tty is not None:
            return self._io._tty.fileno()
        return sys.stdin.fileno()

    # ── Select (arrow-key menu) ───────────────────────────────────────────

    def select(self, title: str, options: list[str]) -> str:
        """Show an interactive arrow-key selection menu.

        Args:
            title: prompt text displayed above the options
            options: list of strings to choose from

        Returns:
            the selected option string, or the first option on cancel/error
        """
        if not options:
            return ""
        if len(options) == 1:
            return options[0]
        if not self._is_tty:
            return self._select_fallback(title, options)
        return self._select_raw(title, options)

    def _select_raw(self, title: str, options: list[str]) -> str:
        """Interactive select using raw terminal input."""
        fd = self._get_fd()
        query = ""
        cursor = 0
        scroll = 0
        max_visible = min(len(options), 20)
        width = _terminal_width()

        def _get_filtered() -> list[str]:
            if query:
                return [o for o in options if query.lower() in o.lower()]
            return list(options)

        def _render() -> list[str]:
            filtered = _get_filtered()
            lines: list[str] = []
            lines.append(f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}")
            if query:
                lines.append(f"{_ERASE_LINE}\r  Filter: {_DIM}{query}{_RESET}{_HIDE_CURSOR}")
            else:
                lines.append("")

            if not filtered:
                lines.append(f"{_ERASE_LINE}\r  {_DIM}No matching options{_RESET}")
            else:
                for i in range(min(max_visible, len(filtered))):
                    idx = scroll + i
                    if idx >= len(filtered):
                        break
                    text = _truncate(filtered[idx], width - 5)
                    if idx == cursor:
                        lines.append(f"{_ERASE_LINE}\r{_REVERSE} > {text}{_RESET_REVERSE}{_RESET}")
                    else:
                        lines.append(f"{_ERASE_LINE}\r   {text}")

            if len(filtered) > max_visible:
                lines.append(f"{_ERASE_LINE}\r   {_DIM}({len(filtered)} items, {cursor + 1}/{len(filtered)}){_RESET}")

            lines.append(f"{_ERASE_LINE}\r  {_DIM}Enter: select  Esc: cancel  Type to filter{_RESET}{_HIDE_CURSOR}")
            return lines

        def _write_lines(lines: list[str]) -> None:
            sys.stdout.write(_HIDE_CURSOR)
            for line in lines:
                sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        def _clear(line_count: int) -> None:
            for _ in range(line_count):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                prev_count = 0
                while True:
                    _clear(prev_count)
                    filtered = _get_filtered()
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)

                    key = _read_raw_key(fd)

                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count)
                        return options[0]

                    if key.kind == _KEY_ENTER:
                        _clear(prev_count)
                        if 0 <= cursor < len(filtered):
                            return filtered[cursor]
                        return options[0]

                    if key.kind == _KEY_UP:
                        cursor = max(0, cursor - 1)
                        if cursor < scroll:
                            scroll = cursor

                    elif key.kind == _KEY_DOWN:
                        cursor = min(len(filtered) - 1, cursor + 1)
                        if cursor >= scroll + max_visible:
                            scroll = cursor - max_visible + 1

                    elif key.kind == _KEY_BACKSPACE:
                        if query:
                            query = query[:-1]
                            cursor = 0
                            scroll = 0

                    elif key.kind == _KEY_CHAR:
                        query += key.char
                        cursor = 0
                        scroll = 0

        except (termios.error, OSError):
            return self._select_fallback(title, options)

    def _select_fallback(self, title: str, options: list[str]) -> str:
        """Numbered-menu fallback for non-TTY environments."""
        self._io.write(f"  {title}")
        for i, opt in enumerate(options, 1):
            self._io.write(f"    {i}. {opt}")
        while True:
            self._io.write("  Enter number: ", end="")
            try:
                raw = self._io.read("").strip()
            except (EOFError, KeyboardInterrupt):
                return options[0] if options else ""
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return options[idx]

    # ── Multi-select (checkbox menu) ─────────────────────────────────────

    def select_multi(self, title: str, options: list[str]) -> list[str]:
        """Show an interactive multi-select menu with checkboxes.

        Use Space to toggle items, arrows to move, Enter to confirm.

        Args:
            title: prompt text displayed above the options
            options: list of strings to choose from

        Returns:
            list of selected option strings
        """
        if not options:
            return []
        if not self._is_tty:
            return self._select_multi_fallback(title, options)
        return self._select_multi_raw(title, options)

    def _select_multi_raw(self, title: str, options: list[str]) -> list[str]:
        """Interactive multi-select using raw terminal input."""
        fd = self._get_fd()
        query = ""
        cursor = 0
        scroll = 0
        checked: set[int] = set()
        max_visible = min(len(options), 20)
        width = _terminal_width()

        def _get_filtered() -> list[tuple[int, str]]:
            if query:
                return [(i, o) for i, o in enumerate(options) if query.lower() in o.lower()]
            return [(i, o) for i, o in enumerate(options)]

        def _render() -> list[str]:
            filtered = _get_filtered()
            lines: list[str] = []
            lines.append(f"{_ERASE_LINE}\r  {_truncate(title, width - 4)}")
            if query:
                lines.append(f"{_ERASE_LINE}\r  Filter: {query}{_HIDE_CURSOR}")
            else:
                lines.append("")

            for i in range(max_visible):
                idx = scroll + i
                if idx >= len(filtered):
                    break
                orig_idx, text = filtered[idx]
                check = f"{_GREEN}\u25c9{_RESET}" if orig_idx in checked else "\u25cb"
                prefix = f"{_REVERSE} > {_RESET_REVERSE}" if idx == cursor else "   "
                display = _truncate(text, width - len(prefix) - 5)
                lines.append(f"{_ERASE_LINE}\r{prefix} {check} {display}")

            if len(filtered) > max_visible:
                lines.append(f"{_ERASE_LINE}\r   ({len(filtered)} items, {cursor + 1}/{len(filtered)})")

            n_checked = len(checked)
            lines.append(f"{_ERASE_LINE}\r  Space: toggle  Enter: confirm ({n_checked} selected)  Esc: cancel{_HIDE_CURSOR}")
            return lines

        def _write_lines(lines: list[str]) -> None:
            sys.stdout.write(_HIDE_CURSOR)
            for line in lines:
                sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        def _clear(line_count: int) -> None:
            for _ in range(line_count):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                prev_count = 0
                while True:
                    _clear(prev_count)
                    filtered = _get_filtered()
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)

                    key = _read_raw_key(fd)

                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count)
                        return []

                    if key.kind == _KEY_ENTER:
                        _clear(prev_count)
                        return [options[i] for i in sorted(checked)]

                    if key.kind == _KEY_UP:
                        cursor = max(0, cursor - 1)
                        if cursor < scroll:
                            scroll = cursor

                    elif key.kind == _KEY_DOWN:
                        cursor = min(len(filtered) - 1, cursor + 1)
                        if cursor >= scroll + max_visible:
                            scroll = cursor - max_visible + 1

                    elif key.kind == _KEY_CHAR and key.char == " ":
                        if 0 <= cursor < len(filtered):
                            orig_idx = filtered[cursor][0]
                            if orig_idx in checked:
                                checked.discard(orig_idx)
                            else:
                                checked.add(orig_idx)

                    elif key.kind == _KEY_BACKSPACE:
                        if query:
                            query = query[:-1]
                            cursor = 0
                            scroll = 0

                    elif key.kind == _KEY_CHAR:
                        query += key.char
                        cursor = 0
                        scroll = 0

        except (termios.error, OSError):
            return self._select_multi_fallback(title, options)

    def _select_multi_fallback(self, title: str, options: list[str]) -> list[str]:
        """Numbered-menu multi-select fallback for non-TTY environments."""
        self._io.write(f"  {title}")
        for i, opt in enumerate(options, 1):
            self._io.write(f"    {i}. {opt}")
        self._io.write("  Enter numbers (comma-separated): ", end="")
        try:
            raw = self._io.read("").strip()
        except (EOFError, KeyboardInterrupt):
            return []
        if not raw:
            return []
        result = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(options):
                    result.append(options[idx])
        return result

    # ── Confirm (y/n with arrow toggle) ───────────────────────────────────

    def confirm(self, message: str, default: bool = False) -> bool:
        """Interactive yes/no prompt with arrow-key toggle.

        Args:
            message: prompt text
            default: default answer if Enter is pressed

        Returns:
            True for yes, False for no
        """
        if not self._is_tty:
            return self._confirm_fallback(message, default)
        return self._confirm_raw(message, default)

    def _confirm_raw(self, message: str, default: bool) -> bool:
        """Interactive confirm using raw terminal input."""
        fd = self._get_fd()
        hint = "Y/n" if default else "y/N"
        selected = default

        def _render() -> str:
            if selected:
                yes_text = f"{_REVERSE}{_GREEN} Yes {_RESET_REVERSE}{_RESET}"
                no_text = f" No "
            else:
                yes_text = f" Yes "
                no_text = f"{_REVERSE}{_RED} No {_RESET_REVERSE}{_RESET}"
            return f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET} [{hint}]  {yes_text}  {no_text}{_HIDE_CURSOR}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()

                    key = _read_raw_key(fd)

                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return not default

                    if key.kind == _KEY_ENTER:
                        _clear()
                        return selected

                    if key.kind in (_KEY_UP, _KEY_DOWN):
                        selected = not selected

                    if key.kind == _KEY_CHAR:
                        c = key.char.lower()
                        if c in ("y", "1"):
                            _clear()
                            return True
                        if c in ("n", "0"):
                            _clear()
                            return False

        except (termios.error, OSError):
            return self._confirm_fallback(message, default)

    def _confirm_fallback(self, message: str, default: bool) -> bool:
        """Line-mode fallback for non-TTY."""
        hint = "Y/n" if default else "y/N"
        self._io.write(f"  {message} [{hint}] ", end="")
        try:
            raw = self._io.read("").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return default
        if not raw:
            return default
        return raw in ("y", "yes", "ye", "true", "1")

    # ── Ask (free-form with cursor) ───────────────────────────────────────

    def ask(self, message: str, default: str = "") -> str:
        """Interactive text input with cursor movement.

        Args:
            message: prompt text
            default: default value (shown in brackets, used if empty)

        Returns:
            the entered string, or the default if empty/cancelled
        """
        if not self._is_tty:
            return self._ask_fallback(message, default)
        return self._ask_raw(message, default)

    def _ask_raw(self, message: str, default: str) -> str:
        """Interactive ask using raw terminal input."""
        fd = self._get_fd()
        suffix = f" [{_DIM}{default}{_RESET}]" if default else ""
        buf: list[str] = list(default) if default else []
        cursor = 0

        def _render() -> str:
            display = "".join(buf)
            before = display[:cursor]
            after = display[cursor:]
            return f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}{suffix}: {before}{_REVERSE}|{_RESET_REVERSE}{after}{_HIDE_CURSOR}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()

                    key = _read_raw_key(fd)

                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return default

                    if key.kind == _KEY_ENTER:
                        _clear()
                        text = "".join(buf)
                        return text if text else default

                    if key.kind == _KEY_BACKSPACE:
                        if cursor > 0:
                            buf.pop(cursor - 1)
                            cursor -= 1

                    elif key.kind == _KEY_DELETE:
                        if cursor < len(buf):
                            buf.pop(cursor)

                    elif key.kind == _KEY_LEFT:
                        cursor = max(0, cursor - 1)

                    elif key.kind == _KEY_RIGHT:
                        cursor = min(len(buf), cursor + 1)

                    elif key.kind == _KEY_HOME:
                        cursor = 0

                    elif key.kind == _KEY_END:
                        cursor = len(buf)

                    elif key.kind in (_KEY_UP, _KEY_DOWN):
                        pass  # no multi-line in ask

                    elif key.kind == _KEY_CHAR:
                        buf.insert(cursor, key.char)
                        cursor += 1

        except (termios.error, OSError):
            return self._ask_fallback(message, default)

    def _ask_fallback(self, message: str, default: str) -> str:
        """Line-mode fallback for non-TTY."""
        suffix = f" [{default}]" if default else ""
        self._io.write(f"  {message}{suffix}: ", end="")
        try:
            raw = self._io.read("").strip()
        except (EOFError, KeyboardInterrupt):
            return default
        return raw if raw else default

    # ── Edit (validated text input) ────────────────────────────────────

    def edit(self, message: str, default: str = "", validator: "Callable[[str], str | None] | None" = None) -> str:
        """Interactive text input with inline validation.

        Args:
            message: prompt text
            default: default value
            validator: optional function that returns an error message if
                       invalid, or None if valid

        Returns:
            the entered string, or the default if empty/cancelled
        """
        if not self._is_tty:
            return self._ask_fallback(message, default)
        return self._edit_raw(message, default, validator)

    # ── Password input ─────────────────────────────────────────────────

    def password(self, message: str) -> str:
        """Interactive password input with masked characters."""
        if not self._is_tty:
            return self._ask_fallback(message, "")
        return self._password_raw(message)

    def _password_raw(self, message: str) -> str:
        fd = self._get_fd()
        buf: list[str] = []
        cursor = 0

        def _render() -> str:
            masked = "*" * len(buf)
            before = masked[:cursor]
            after = masked[cursor:]
            return f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}: {before}{_REVERSE}|{_RESET_REVERSE}{after}{_HIDE_CURSOR}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return ""
                    if key.kind == _KEY_ENTER:
                        _clear(); return "".join(buf)
                    if key.kind == _KEY_BACKSPACE:
                        if cursor > 0: buf.pop(cursor - 1); cursor -= 1
                    elif key.kind == _KEY_DELETE:
                        if cursor < len(buf): buf.pop(cursor)
                    elif key.kind == _KEY_LEFT:
                        cursor = max(0, cursor - 1)
                    elif key.kind == _KEY_RIGHT:
                        cursor = min(len(buf), cursor + 1)
                    elif key.kind == _KEY_HOME:
                        cursor = 0
                    elif key.kind == _KEY_END:
                        cursor = len(buf)
                    elif key.kind == _KEY_CHAR:
                        buf.insert(cursor, key.char); cursor += 1
        except (termios.error, OSError):
            return self._ask_fallback(message, "")

    def pager(self, content: str, title: str = "Output") -> None:
        """Display long content in a scrollable pager view.

        Controls: ↑/↓ or j/k to scroll, Page Up/Down, q to quit.

        Args:
            content: the text content to display
            title: title shown at top
        """
        if not self._is_tty:
            self._io.write(content)
            return
        self._pager_raw(content, title)

    def _pager_raw(self, content: str, title: str) -> None:
        """Scrollable pager using raw terminal input."""
        fd = self._get_fd()
        lines = content.split("\n")
        total = len(lines)
        height = _terminal_height() - 3  # title + footer
        offset = 0

        def _render() -> None:
            sys.stdout.write("\033[2J\033[H")  # clear + home
            sys.stdout.write(f"{_BOLD}{_CYAN}{title}{_RESET}  ")
            sys.stdout.write(f"{_DIM}(↑/↓/j/k scroll, q quit){_RESET}\n")
            visible = lines[offset:offset + height]
            for line in visible:
                sys.stdout.write(f"  {line}\n")
            pos = f"Line {offset + 1}-{min(offset + height, total)} of {total}"
            sys.stdout.write(f"{_DIM}{pos}{_RESET}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                _render()
                while True:
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC, _KEY_CHAR) and key.char == "q":
                        break
                    elif key.kind == _KEY_UP or (key.kind == _KEY_CHAR and key.char == "k"):
                        offset = max(0, offset - 1)
                    elif key.kind == _KEY_DOWN or (key.kind == _KEY_CHAR and key.char == "j"):
                        offset = min(total - height, offset + 1)
                    elif key.kind == _KEY_PAGE_UP:
                        offset = max(0, offset - height)
                    elif key.kind == _KEY_PAGE_DOWN:
                        offset = min(total - height, offset + height)
                    elif key.kind == _KEY_HOME:
                        offset = 0
                    elif key.kind == _KEY_END:
                        offset = max(0, total - height)
                    if offset < 0:
                        offset = 0
                    _render()
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.flush()
        except (termios.error, OSError):
            self._io.write(content)

    # ── Confirm action ─────────────────────────────────────────────────

    def confirm_action(self, action: str, details: str = "",
                       danger: bool = False) -> bool:
        """Confirm an action with a descriptive prompt.

        Args:
            action: the action description (e.g. "Delete file")
            details: optional additional details
            danger: if True, shows a warning color

        Returns:
            True if confirmed, False otherwise
        """
        if not self._is_tty:
            return self._confirm_fallback(f"{action}? (y/N)", False)
        return self._confirm_action_raw(action, details, danger)

    def _confirm_action_raw(self, action: str, details: str,
                            danger: bool) -> bool:
        fd = self._get_fd()
        selected = False

        def _render() -> str:
            if selected:
                yes_text = f"{_REVERSE}{_GREEN} Yes {_RESET_REVERSE}{_RESET}"
                no_text = " No "
            else:
                yes_text = " Yes "
                no_text = f"{_REVERSE}{_RED} No {_RESET_REVERSE}{_RESET}"
            color = _RED if danger else _CYAN
            detail = f"\n  {_DIM}{details}{_RESET}" if details else ""
            return f"{_ERASE_LINE}\r  {_BOLD}{color}{action}{_RESET}?{detail}  {yes_text}  {no_text}{_HIDE_CURSOR}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            if details:
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return False
                    if key.kind == _KEY_ENTER:
                        _clear(); return selected
                    if key.kind in (_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear(); return True
                        if key.char in ("n", "N"):
                            _clear(); return False
        except (termios.error, OSError):
            return self._confirm_fallback(f"{action}? (y/N)", False)

    # ── Countdown ──────────────────────────────────────────────────────

    def countdown(self, seconds: int, message: str = "Starting in") -> bool:
        """Show a visual countdown timer. Returns True if completed."""
        if not self._is_tty:
            for i in range(seconds, 0, -1):
                self._io.write(f"  {message} {i}...")
            return True
        return self._countdown_raw(seconds, message)

    def _countdown_raw(self, seconds: int, message: str) -> bool:
        fd = self._get_fd()
        remaining = seconds
        try:
            with _RawTerminal(fd):
                while remaining > 0:
                    bar_w = 30
                    filled = int((remaining / seconds) * bar_w)
                    bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
                    pct = remaining / seconds * 100
                    sys.stdout.write(
                        f"\r{_ERASE_LINE}\r  {_CYAN}{message}{_RESET} "
                        f"{_BOLD}{remaining:>2}s{_RESET} "
                        f"{_DIM}{bar}{_RESET} {pct:.0f}%{_HIDE_CURSOR}"
                    )
                    sys.stdout.flush()
                    time.sleep(1)
                    remaining -= 1
                sys.stdout.write(
                    f"\r{_ERASE_LINE}\r  {_GREEN}{_BOLD}\u2714 Done{_RESET}{_SHOW_CURSOR}\n"
                )
                sys.stdout.flush()
                return True
        except (termios.error, OSError):
            for i in range(seconds, 0, -1):
                self._io.write(f"  {message} {i}...")
            return True

    # ── Banner ─────────────────────────────────────────────────────────

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
            "thick":  ("\u250f", "\u2501", "\u2513", "\u2503", "\u2517", "\u251b"),
            "dashed": ("\u250c", "\u2504", "\u2510", "\u2502", "\u2514", "\u2518"),
        }
        tl, h, tr, v, bl, br = styles.get(style, styles["double"])
        inner = h * (width - 2)
        pad = " " * max(0, width - len(text) - 3)
        self._io.write(f"  {_BOLD}{_CYAN}{tl}{inner}{tr}{_RESET}")
        self._io.write(f"  {_BOLD}{_CYAN}{v}{_RESET} {_BOLD}{text}{_RESET}{pad}{_BOLD}{_CYAN}{v}{_RESET}")
        self._io.write(f"  {_BOLD}{_CYAN}{bl}{inner}{br}{_RESET}")

    # ── Slider ─────────────────────────────────────────────────────────

    def slider(self, message: str, min_val: int = 0, max_val: int = 100,
               default: int = 50, step: int = 1) -> int:
        """Interactive numeric slider with visual bar.

        Args:
            message: prompt text
            min_val: minimum value
            max_val: maximum value
            default: starting value
            step: increment per arrow press

        Returns:
            the selected integer value
        """
        if not self._is_tty:
            raw = self._ask_fallback(message, str(default))
            try:
                return int(raw) if raw else default
            except ValueError:
                return default
        return self._slider_raw(message, min_val, max_val, default, step)

    def _slider_raw(self, message: str, min_val: int, max_val: int,
                    default: int, step: int) -> int:
        fd = self._get_fd()
        value = default
        bar_w = 30

        def _render() -> str:
            ratio = (value - min_val) / max(max_val - min_val, 1)
            filled = int(ratio * bar_w)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET} "
                f"{_BOLD}{value:>4}{_RESET} "
                f"{_DIM}[{min_val}]{_RESET} {bar} {_DIM}[{max_val}]{_RESET}"
                f"{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return default
                    if key.kind == _KEY_ENTER:
                        _clear(); return value
                    if key.kind == _KEY_LEFT:
                        value = max(min_val, value - step)
                    elif key.kind == _KEY_RIGHT:
                        value = min(max_val, value + step)
                    elif key.kind == _KEY_HOME:
                        value = min_val
                    elif key.kind == _KEY_END:
                        value = max_val
        except (termios.error, OSError):
            return default

    # ── Toggle ─────────────────────────────────────────────────────────

    def toggle(self, message: str, default: bool = False) -> bool:
        """Interactive on/off toggle switch.

        Args:
            message: prompt text
            default: starting state

        Returns:
            the toggled boolean value
        """
        if not self._is_tty:
            return self._confirm_fallback(message, default)
        return self._toggle_raw(message, default)

    def _toggle_raw(self, message: str, default: bool) -> bool:
        fd = self._get_fd()
        on = default

        def _render() -> str:
            if on:
                on_text = f"{_REVERSE}{_GREEN} ON {_RESET_REVERSE}{_RESET}"
                off_text = f"OFF "
            else:
                on_text = f" ON "
                off_text = f"{_REVERSE}{_RED}OFF{_RESET_REVERSE}{_RESET}"
            return f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {on_text}  {off_text}{_HIDE_CURSOR}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return default
                    if key.kind == _KEY_ENTER:
                        _clear(); return on
                    if key.kind in (_KEY_LEFT, _KEY_RIGHT):
                        on = not on
                    if key.kind == _KEY_CHAR:
                        if key.char in ("1", "t", "T"):
                            on = True
                        elif key.char in ("0", "f", "F"):
                            on = False
        except (termios.error, OSError):
            return default

    # ── Tag input ──────────────────────────────────────────────────────

    def tag_input(self, message: str, defaults: list[str] | None = None,
                  placeholder: str = "Add tag...") -> list[str]:
        """Interactive tag input. Type and press Enter to add, Backspace to remove.

        Args:
            message: prompt text
            defaults: initial tags
            placeholder: placeholder when empty

        Returns:
            list of entered tags
        """
        if not self._is_tty:
            raw = self._ask_fallback(message, "")
            return [t.strip() for t in raw.split(",") if t.strip()] if raw else (defaults or [])
        return self._tag_input_raw(message, defaults or [], placeholder)

    def _tag_input_raw(self, message: str, defaults: list[str],
                       placeholder: str) -> list[str]:
        fd = self._get_fd()
        tags: list[str] = list(defaults)
        buf: list[str] = []

        def _render() -> str:
            tag_str = ""
            for t in tags:
                tag_str += f" {_GREEN}\u25cf {t}{_RESET}\u2500"
            if not tags and not buf:
                tag_str = f" {_DIM}{placeholder}{_RESET}"
            input_part = "".join(buf)
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}"
                f"{tag_str} {input_part}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return tags
                    if key.kind == _KEY_ENTER:
                        word = "".join(buf).strip()
                        if word:
                            tags.append(word)
                            buf.clear()
                    elif key.kind == _KEY_BACKSPACE:
                        if buf:
                            buf.pop()
                        elif tags:
                            tags.pop()
                    elif key.kind == _KEY_DELETE:
                        if buf:
                            buf.clear()
                    elif key.kind == _KEY_CHAR:
                        if key.char == ",":
                            word = "".join(buf).strip()
                            if word:
                                tags.append(word)
                                buf.clear()
                        else:
                            buf.append(key.char)
                    if key.kind == _KEY_ENTER and not "".join(buf).strip():
                        _clear(); return tags
        except (termios.error, OSError):
            return tags

    # ── Select tree ────────────────────────────────────────────────────

    def select_tree(self, title: str, tree: dict[str, list[str] | dict],
                    expanded: set[str] | None = None) -> str | None:
        """Interactive tree selector with expand/collapse.

        Args:
            title: prompt text
            tree: nested dict structure
            expanded: set of initially expanded node names

        Returns:
            selected leaf label or None if cancelled
        """
        if not self._is_tty:
            return self._select_fallback(title, self._tree_leaves(tree)) or None
        return self._select_tree_raw(title, tree, expanded or set())

    def _tree_leaves(self, tree: dict) -> list[str]:
        leaves: list[str] = []
        for k, v in tree.items():
            if isinstance(v, dict):
                leaves.extend(self._tree_leaves(v))
            elif isinstance(v, list):
                leaves.extend(v)
        return leaves

    def _select_tree_raw(self, title: str, tree: dict[str, list[str] | dict],
                         expanded: set[str]) -> str | None:
        fd = self._get_fd()
        flat: list[tuple[str, int, str]] = []

        def _flatten(d: dict, depth: int = 0) -> None:
            for k, v in d.items():
                is_branch = isinstance(v, dict)
                kind = "branch" if is_branch else "leaf"
                flat.append((k, depth, kind))
                if is_branch and k in expanded:
                    _flatten(v, depth + 1)
                elif isinstance(v, list):
                    for item in v:
                        flat.append((item, depth + 1, "leaf"))

        def _rebuild() -> None:
            flat.clear()
            _flatten(tree)

        _rebuild()
        cursor = 0
        query = ""
        max_visible = min(len(flat), 15)

        def _get_filtered() -> list[tuple[str, int, str]]:
            if query:
                return [(n, d, k) for n, d, k in flat if query.lower() in n.lower()]
            return list(flat)

        def _render() -> list[str]:
            filtered = _get_filtered()
            lines: list[str] = []
            lines.append(f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}")
            if query:
                lines.append(f"{_ERASE_LINE}\r  Filter: {_DIM}{query}{_RESET}{_HIDE_CURSOR}")
            else:
                lines.append("")

            if not filtered:
                lines.append(f"{_ERASE_LINE}\r  {_DIM}No matching items{_RESET}")
            else:
                for i in range(min(max_visible, len(filtered))):
                    name, depth, kind = filtered[i]
                    prefix = "  " * depth
                    if kind == "branch":
                        icon = "\u25bc" if name in expanded else "\u25b6"
                        text = f"{_BOLD}{name}{_RESET}"
                    else:
                        icon = "\u25cf"
                        text = f"{_DIM}{name}{_RESET}"
                    display_text = f"{prefix}{icon} {text}"
                    if flat.index(filtered[i]) == cursor:
                        lines.append(f"{_ERASE_LINE}\r{_REVERSE} {display_text}{_RESET_REVERSE}{_RESET}")
                    else:
                        lines.append(f"{_ERASE_LINE}\r  {display_text}")

            lines.append(f"{_ERASE_LINE}\r  {_DIM}Enter: select/expand  Esc: cancel  Type to filter{_RESET}{_HIDE_CURSOR}")
            return lines

        def _write_lines(lines: list[str]) -> None:
            sys.stdout.write(_HIDE_CURSOR)
            for line in lines:
                sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        def _clear(line_count: int) -> None:
            for _ in range(line_count):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                prev_count = 0
                while True:
                    _clear(prev_count)
                    filtered = _get_filtered()
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count); return None
                    if key.kind == _KEY_ENTER:
                        if 0 <= cursor < len(filtered):
                            name, _, kind = filtered[cursor]
                            if kind == "branch":
                                if name in expanded:
                                    expanded.discard(name)
                                else:
                                    expanded.add(name)
                                _rebuild()
                                cursor = min(cursor, len(flat) - 1)
                            else:
                                _clear(prev_count)
                                return name
                    if key.kind == _KEY_UP:
                        cursor = max(0, cursor - 1)
                    elif key.kind == _KEY_DOWN:
                        cursor = min(len(filtered) - 1, cursor + 1)
                    elif key.kind == _KEY_BACKSPACE:
                        if query:
                            query = query[:-1]
                            cursor = 0
                    elif key.kind == _KEY_CHAR:
                        query += key.char
                        cursor = 0
        except (termios.error, OSError):
            leaves = self._tree_leaves(tree)
            return leaves[0] if leaves else None

    # ── Spin wait ──────────────────────────────────────────────────────

    def spin_wait(self, message: str, check_fn: "Callable[[], bool]",
                  interval: float = 0.1, timeout: float = 0) -> bool:
        """Wait for a condition with a spinner. Returns True when ready.

        Args:
            message: message to display
            check_fn: function that returns True when done
            interval: check interval in seconds
            timeout: max seconds to wait (0 = forever)

        Returns:
            True if condition met, False if timed out
        """
        if not self._is_tty:
            import time as _t
            start = _t.monotonic()
            while not check_fn():
                _t.sleep(interval)
                if timeout and _t.monotonic() - start >= timeout:
                    return False
            return True
        return self._spin_wait_raw(message, check_fn, interval, timeout)

    def _spin_wait_raw(self, message: str, check_fn: "Callable[[], bool]",
                       interval: float, timeout: float) -> bool:
        import time as _t
        fd = self._get_fd()
        frames = ["\u25cf", "\u25cf\u25cf", "\u25cf\u25cf\u25cf", "\u25cf"]
        idx = 0
        start = _t.monotonic()
        try:
            with _RawTerminal(fd):
                while True:
                    frame = frames[idx % len(frames)]
                    elapsed = _t.monotonic() - start
                    sys.stdout.write(
                        f"\r{_ERASE_LINE}\r  {_CYAN}{frame}{_RESET} {_BOLD}{message}{_RESET} "
                        f"{_DIM}{elapsed:.1f}s{_RESET}{_HIDE_CURSOR}"
                    )
                    sys.stdout.flush()
                    if check_fn():
                        sys.stdout.write(
                            f"\r{_ERASE_LINE}\r  {_GREEN}\u2714{_RESET} {_BOLD}{message}{_RESET} "
                            f"{_GREEN}done{_RESET}{_SHOW_CURSOR}\n"
                        )
                        sys.stdout.flush()
                        return True
                    if timeout and elapsed >= timeout:
                        sys.stdout.write(
                            f"\r{_ERASE_LINE}\r  {_RED}\u2716{_RESET} {_BOLD}{message}{_RESET} "
                            f"{_RED}timeout{_RESET}{_SHOW_CURSOR}\n"
                        )
                        sys.stdout.flush()
                        return False
                    _t.sleep(interval)
                    idx += 1
        except (termios.error, OSError):
            while not check_fn():
                _t.sleep(interval)
                if timeout and _t.monotonic() - start >= timeout:
                    return False
            return True

    # ── Confirm dangerous (typing) ─────────────────────────────────────

    def confirm_dangerous(self, action: str, phrase: str = "yes, I am sure") -> bool:
        """Confirm a dangerous action by typing a phrase."""
        if not self._is_tty:
            raw = self._ask_fallback(f"{action}. Type '{phrase}' to confirm", "")
            return raw.strip().lower() == phrase.lower()
        return self._confirm_dangerous_raw(action, phrase)

    def _confirm_dangerous_raw(self, action: str, phrase: str) -> bool:
        fd = self._get_fd()
        buf: list[str] = []

        def _render() -> str:
            typed = "".join(buf)
            match = typed.lower() == phrase.lower()
            color = _GREEN if match else _RED
            return (
                f"{_ERASE_LINE}\r  {_RED}\u26a0 {_BOLD}{action}{_RESET}\n"
                f"  {_DIM}Type {_BOLD}{phrase}{_DIM} to confirm{_RESET}: "
                f"{color}{typed}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return False
                    if key.kind == _KEY_ENTER:
                        typed = "".join(buf).strip().lower()
                        _clear(); return typed == phrase.lower()
                    if key.kind == _KEY_BACKSPACE:
                        if buf: buf.pop()
                    elif key.kind == _KEY_CHAR:
                        buf.append(key.char)
        except (termios.error, OSError):
            raw = self._ask_fallback(f"{action}. Type '{phrase}' to confirm", "")
            return raw.strip().lower() == phrase.lower()

    # ── File browser ───────────────────────────────────────────────────

    def file_browser(self, title: str, start_dir: str = ".",
                     pattern: str = "*") -> str | None:
        """Interactive file browser with directory navigation."""
        import os as _os
        import glob as _glob
        if not self._is_tty:
            return self._select_fallback(title, ["(no TTY)"])

        def _list_dir(path: str) -> list[str]:
            try:
                entries: list[str] = []
                for e in sorted(_os.listdir(path)):
                    full = _os.path.join(path, e)
                    if _os.path.isdir(full):
                        entries.append(f"\U0001f4c1 {e}/")
                    elif _glob.fnmatch.fnmatch(e, pattern):
                        entries.append(f"\U0001f4c4 {e}")
                return entries
            except PermissionError:
                return [f"{_RED}(permission denied){_RESET}"]
        return self._file_browser_raw(title, _os.path.abspath(start_dir), _list_dir)

    def _file_browser_raw(self, title: str, start: str,
                          list_fn: "Callable[[str], list[str]]") -> str | None:
        import os as _os
        fd = self._get_fd()
        current = start
        cursor = 0
        query = ""
        max_visible = 15

        def _get_filtered() -> list[str]:
            items = list_fn(current)
            if query:
                return [i for i in items if query.lower() in i.lower()]
            return items

        def _render() -> list[str]:
            filtered = _get_filtered()
            lines: list[str] = []
            lines.append(f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}")
            lines.append(f"{_ERASE_LINE}\r  {_DIM}{current}{_RESET}")
            if query:
                lines.append(f"{_ERASE_LINE}\r  Filter: {_DIM}{query}{_RESET}{_HIDE_CURSOR}")
            else:
                lines.append("")
            if not filtered:
                lines.append(f"{_ERASE_LINE}\r  {_DIM}(empty){_RESET}")
            else:
                for i in range(min(max_visible, len(filtered))):
                    text = filtered[i][:60]
                    if i == cursor:
                        lines.append(f"{_ERASE_LINE}\r{_REVERSE} > {text}{_RESET_REVERSE}{_RESET}")
                    else:
                        lines.append(f"{_ERASE_LINE}\r   {text}")
            lines.append(f"{_ERASE_LINE}\r  {_DIM}Enter: select  Esc: cancel  Backspace: parent{_RESET}{_HIDE_CURSOR}")
            return lines

        def _write_lines(lines: list[str]) -> None:
            sys.stdout.write(_HIDE_CURSOR)
            for line in lines:
                sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        def _clear(line_count: int) -> None:
            for _ in range(line_count):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                prev_count = 0
                while True:
                    _clear(prev_count)
                    filtered = _get_filtered()
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count); return None
                    if key.kind == _KEY_ENTER:
                        if 0 <= cursor < len(filtered):
                            selected = filtered[cursor]
                            if selected.startswith("\U0001f4c1"):
                                dirname = selected[2:].rstrip("/")
                                current = _os.path.abspath(_os.path.join(current, dirname))
                                cursor = 0; query = ""
                            else:
                                _clear(prev_count)
                                return _os.path.join(current, selected[2:])
                    if key.kind == _KEY_UP:
                        cursor = max(0, cursor - 1)
                    elif key.kind == _KEY_DOWN:
                        cursor = min(len(filtered) - 1, cursor + 1)
                    elif key.kind == _KEY_BACKSPACE:
                        if query:
                            query = query[:-1]; cursor = 0
                        elif current != "/":
                            current = _os.path.dirname(current); cursor = 0
                    elif key.kind == _KEY_CHAR:
                        query += key.char; cursor = 0
        except (termios.error, OSError):
            return None

    # ── Progress step ──────────────────────────────────────────────────

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

    # ── Multi choice ───────────────────────────────────────────────────

    def multi_choice(self, title: str, options: list[str],
                     defaults: list[int] | None = None) -> list[str]:
        """Select multiple options with Space to toggle, Enter to confirm."""
        if not options:
            return []
        if not self._is_tty:
            return self._select_multi_fallback(title, options)
        return self._multi_choice_raw(title, options, defaults or [])

    def _multi_choice_raw(self, title: str, options: list[str],
                          defaults: list[int]) -> list[str]:
        fd = self._get_fd()
        selected: set[int] = set(defaults)
        cursor = 0
        query = ""
        max_visible = min(len(options), 15)

        def _get_filtered() -> list[tuple[int, str]]:
            if query:
                return [(i, o) for i, o in enumerate(options) if query.lower() in o.lower()]
            return [(i, o) for i, o in enumerate(options)]

        def _render() -> list[str]:
            filtered = _get_filtered()
            lines: list[str] = []
            lines.append(f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}  {_DIM}({len(selected)} selected){_RESET}")
            if query:
                lines.append(f"{_ERASE_LINE}\r  Filter: {_DIM}{query}{_RESET}{_HIDE_CURSOR}")
            else:
                lines.append("")
            if not filtered:
                lines.append(f"{_ERASE_LINE}\r  {_DIM}No matching options{_RESET}")
            else:
                for i in range(min(max_visible, len(filtered))):
                    orig_idx, text = filtered[i]
                    check = f"{_GREEN}\u2611{_RESET}" if orig_idx in selected else f"{_DIM}\u2610{_RESET}"
                    if orig_idx == cursor:
                        lines.append(f"{_ERASE_LINE}\r{_REVERSE} {check} {text}{_RESET_REVERSE}{_RESET}")
                    else:
                        lines.append(f"{_ERASE_LINE}\r  {check} {text}")
            lines.append(f"{_ERASE_LINE}\r  {_DIM}Space: toggle  Enter: confirm  Type to filter{_RESET}{_HIDE_CURSOR}")
            return lines

        def _write_lines(lines: list[str]) -> None:
            sys.stdout.write(_HIDE_CURSOR)
            for line in lines:
                sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        def _clear(line_count: int) -> None:
            for _ in range(line_count):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                prev_count = 0
                while True:
                    _clear(prev_count)
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count); return []
                    if key.kind == _KEY_ENTER:
                        _clear(prev_count)
                        filtered = _get_filtered()
                        return [o for i, o in filtered if i in selected]
                    if key.kind == _KEY_UP:
                        cursor = max(0, cursor - 1)
                    elif key.kind == _KEY_DOWN:
                        cursor = min(len(options) - 1, cursor + 1)
                    elif key.kind == _KEY_SPACE:
                        if cursor in selected:
                            selected.discard(cursor)
                        else:
                            selected.add(cursor)
                    elif key.kind == _KEY_BACKSPACE:
                        if query:
                            query = query[:-1]
                            cursor = min(cursor, len(options) - 1)
                    elif key.kind == _KEY_CHAR:
                        query += key.char; cursor = 0
        except (termios.error, OSError):
            return self._select_multi_fallback(title, options)

    # ── Date picker ────────────────────────────────────────────────────

    def date_picker(self, message: str, default: str = "") -> str:
        """Interactive date picker with year/month/day navigation.

        Args:
            message: prompt text
            default: default date as "YYYY-MM-DD" (empty = today)

        Returns:
            selected date as "YYYY-MM-DD"
        """
        import datetime as _dt
        if not self._is_tty:
            return self._ask_fallback(message, default or _dt.date.today().isoformat())
        return self._date_picker_raw(message, default)

    def _date_picker_raw(self, message: str, default: str) -> str:
        import datetime as _dt
        fd = self._get_fd()
        if default:
            try:
                parts = default.split("-")
                cur = _dt.date(int(parts[0]), int(parts[1]), int(parts[2]))
            except (ValueError, IndexError):
                cur = _dt.date.today()
        else:
            cur = _dt.date.today()
        focus = 0  # 0=year, 1=month, 2=day

        labels = ["Year", "Month", "Day"]

        def _render() -> str:
            y_color = _CYAN if focus == 0 else ""
            m_color = _CYAN if focus == 1 else ""
            d_color = _CYAN if focus == 2 else ""
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  "
                f"{y_color}{_BOLD}{cur.year:>4}{_RESET}-"
                f"{m_color}{_BOLD}{cur.month:>02}{_RESET}-"
                f"{d_color}{_BOLD}{cur.day:>02}{_RESET}  "
                f"{_DIM}{labels[focus]}{_RESET}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return default or _dt.date.today().isoformat()
                    if key.kind == _KEY_ENTER:
                        _clear(); return cur.isoformat()
                    if key.kind == _KEY_LEFT:
                        focus = max(0, focus - 1)
                    elif key.kind == _KEY_RIGHT:
                        focus = min(2, focus + 1)
                    elif key.kind == _KEY_UP:
                        if focus == 0:
                            cur = cur.replace(year=cur.year + 1)
                        elif focus == 1:
                            m = cur.month % 12 + 1
                            cur = cur.replace(month=m)
                        else:
                            try:
                                cur = cur.replace(day=cur.day + 1)
                            except ValueError:
                                cur = cur.replace(day=1)
                    elif key.kind == _KEY_DOWN:
                        if focus == 0:
                            cur = cur.replace(year=cur.year - 1)
                        elif focus == 1:
                            m = (cur.month - 2) % 12 + 1
                            cur = cur.replace(month=m)
                        else:
                            try:
                                cur = cur.replace(day=cur.day - 1)
                            except ValueError:
                                import calendar
                                cur = cur.replace(day=calendar.monthrange(cur.year, cur.month)[1])
        except (termios.error, OSError):
            return default or _dt.date.today().isoformat()

    # ── History search ────────────────────────────────────────────────

    def history_search(self, history: list[str], message: str = "History:") -> str | None:
        """Interactive history search with type-to-filter (Ctrl+R style)."""
        if not self._is_tty:
            return self._select_fallback(message, history[-10:]) if history else None
        return self._history_search_raw(history, message)

    def _history_search_raw(self, history: list[str], message: str) -> str | None:
        fd = self._get_fd()
        query = ""
        idx = 0

        def _filtered() -> list[str]:
            if not query:
                return list(reversed(history[-50:]))
            q = query.lower()
            return [h for h in reversed(history) if q in h.lower()][:50]

        def _render() -> str:
            flt = _filtered()
            lines = []
            for i, cmd in enumerate(flt):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                display = cmd[:60] + "..." if len(cmd) > 60 else cmd
                lines.append(f"  {prefix}{color}{display}{_RESET}")
            if not lines:
                lines = [f"  {_DIM}(no matches){_RESET}"]
            query_display = f"  {_BOLD}Search:{_RESET} {query}{_CYAN}\u2502{_RESET}" if query else f"  {_DIM}Type to search history{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows, type=search, enter=select){_RESET}\n"
                + query_display
                + "\n" + "\n".join(lines)
                + f"{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(2 + len(history[-50:])):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    flt = _filtered()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return None
                    if key.kind == _KEY_ENTER:
                        _clear(); return flt[idx] if flt else None
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % max(len(flt), 1)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % max(len(flt), 1)
                    elif key.kind == _KEY_CHAR:
                        if key.char == "\x7f":
                            query = query[:-1]
                        else:
                            query += key.char
                        idx = 0
                    elif key.kind == _KEY_BACKSPACE:
                        query = query[:-1]
                        idx = 0
        except (termios.error, OSError):
            flt = _filtered()
            return flt[0] if flt else None

    # ── Process manager ───────────────────────────────────────────────

    def process_manager(self, processes: list[dict[str, str]],
                        message: str = "Processes:") -> dict[str, str] | None:
        """Interactive process manager with live status.

        Args:
            processes: list of dicts with at least 'name' and 'status' keys
            message: prompt text

        Returns:
            selected process dict or None
        """
        if not self._is_tty:
            return processes[0] if processes else None
        return self._process_manager_raw(processes, message)

    def _process_manager_raw(self, processes: list[dict[str, str]],
                             message: str) -> dict[str, str] | None:
        fd = self._get_fd()
        idx = 0

        def _render() -> str:
            lines = []
            for i, proc in enumerate(processes):
                name = proc.get("name", "?")
                status = proc.get("status", "?")
                status_color = _GREEN if status == "running" else _YELLOW if status == "pending" else _RED
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(
                    f"  {prefix}{color}{name:<30} {status_color}{status}{_RESET}{reset}"
                )
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows, enter=select){_RESET}\n"
                + "\n".join(lines)
                + f"{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(2 + len(processes)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return None
                    if key.kind == _KEY_ENTER:
                        _clear(); return processes[idx]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % len(processes)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % len(processes)
        except (termios.error, OSError):
            return processes[idx] if processes else None

    # ── Log viewer ────────────────────────────────────────────────────

    def log_viewer(self, logs: list[str], message: str = "Logs:") -> str | None:
        """Interactive log viewer with scroll and filter."""
        if not self._is_tty:
            return logs[-1] if logs else None
        return self._log_viewer_raw(logs, message)

    def _log_viewer_raw(self, logs: list[str], message: str) -> str | None:
        fd = self._get_fd()
        query = ""
        scroll = 0
        max_visible = 20

        def _filtered() -> list[str]:
            if not query:
                return logs
            q = query.lower()
            return [l for l in logs if q in l.lower()]

        def _render() -> str:
            flt = _filtered()
            visible = flt[scroll:scroll + max_visible]
            lines = []
            for line in visible:
                if "error" in line.lower():
                    color = _RED
                elif "warn" in line.lower():
                    color = _YELLOW
                elif "info" in line.lower():
                    color = _CYAN
                else:
                    color = _DIM
                display = line[:70] + "..." if len(line) > 70 else line
                lines.append(f"  {color}{display}{_RESET}")
            if not lines:
                lines = [f"  {_DIM}(no logs){_RESET}"]
            pos = f"  {_DIM}{scroll + 1}-{min(scroll + max_visible, len(flt))}/{len(flt)}{_RESET}"
            query_display = f"  {_BOLD}Filter:{_RESET} {query}{_CYAN}\u2502{_RESET}" if query else f"  {_DIM}Type to filter{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows=scroll, type=filter){_RESET}\n"
                + query_display
                + "\n" + "\n".join(lines)
                + f"\n{pos}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + max_visible + 1):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    flt = _filtered()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return None
                    if key.kind == _KEY_ENTER:
                        _clear(); return flt[scroll] if flt else None
                    if key.kind == _KEY_UP:
                        scroll = max(0, scroll - 1)
                    elif key.kind == _KEY_DOWN:
                        scroll = min(max(0, len(flt) - max_visible), scroll + 1)
                    elif key.kind == _KEY_PAGE_UP:
                        scroll = max(0, scroll - max_visible)
                    elif key.kind == _KEY_PAGE_DOWN:
                        scroll = min(max(0, len(flt) - max_visible), scroll + max_visible)
                    elif key.kind == _KEY_CHAR:
                        if key.char == "\x7f":
                            query = query[:-1]
                        else:
                            query += key.char
                        scroll = 0
                    elif key.kind == _KEY_BACKSPACE:
                        query = query[:-1]
                        scroll = 0
        except (termios.error, OSError):
            return logs[-1] if logs else None

    # ── Config editor ─────────────────────────────────────────────────

    def config_editor(self, config: dict[str, str | int | float | bool],
                      message: str = "Config:") -> dict[str, str | int | float | bool]:
        """Interactive config editor for key-value pairs."""
        if not self._is_tty:
            return config
        return self._config_editor_raw(config, message)

    def _config_editor_raw(self, config: dict[str, str | int | float | bool],
                           message: str) -> dict[str, str | int | float | bool]:
        fd = self._get_fd()
        keys = list(config.keys())
        idx = 0
        editing = False
        edit_buf = ""

        def _render() -> str:
            lines = []
            for i, key in enumerate(keys):
                val = config[key]
                val_str = str(val)
                if editing and i == idx:
                    lines.append(f"  {_REVERSE}{_CYAN}>> {key} = {edit_buf}\u2502{_RESET_REVERSE}{_RESET}")
                else:
                    prefix = ">> " if i == idx else "   "
                    color = _CYAN if i == idx else ""
                    reset = _RESET if i == idx else ""
                    lines.append(f"  {prefix}{color}{key} = {val_str}{_RESET}")
            help_text = f"  {_DIM}(arrows=navigate, enter=edit, esc=save){_RESET}" if not editing else f"  {_DIM}(type=value, enter=confirm, esc=cancel){_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                + "\n".join(lines)
                + f"\n{help_text}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + len(keys)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if editing:
                        if key.kind == _KEY_ENTER:
                            val = config[keys[idx]]
                            if isinstance(val, bool):
                                config[keys[idx]] = edit_buf.lower() in ("true", "1", "yes")
                            elif isinstance(val, int):
                                try:
                                    config[keys[idx]] = int(edit_buf)
                                except ValueError:
                                    config[keys[idx]] = val
                            elif isinstance(val, float):
                                try:
                                    config[keys[idx]] = float(edit_buf)
                                except ValueError:
                                    config[keys[idx]] = val
                            else:
                                config[keys[idx]] = edit_buf
                            editing = False
                        elif key.kind == _KEY_ESC:
                            editing = False
                        elif key.kind == _KEY_CHAR:
                            if key.char == "\x7f":
                                edit_buf = edit_buf[:-1]
                            else:
                                edit_buf += key.char
                        elif key.kind == _KEY_BACKSPACE:
                            edit_buf = edit_buf[:-1]
                    else:
                        if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                            _clear(); return config
                        if key.kind == _KEY_ENTER:
                            editing = True
                            edit_buf = str(config[keys[idx]])
                        if key.kind == _KEY_UP:
                            idx = (idx - 1) % len(keys)
                        elif key.kind == _KEY_DOWN:
                            idx = (idx + 1) % len(keys)
        except (termios.error, OSError):
            return config

    # ── Diff viewer ───────────────────────────────────────────────────

    def diff_viewer(self, old: str, new: str, message: str = "Diff:") -> str:
        """Interactive side-by-side diff viewer."""
        if not self._is_tty:
            return new
        return self._diff_viewer_raw(old, new, message)

    def _diff_viewer_raw(self, old: str, new: str, message: str) -> str:
        fd = self._get_fd()
        old_lines = old.splitlines()
        new_lines = new.splitlines()
        scroll = 0
        max_visible = 20

        def _render() -> str:
            visible_old = old_lines[scroll:scroll + max_visible]
            visible_new = new_lines[scroll:scroll + max_visible]
            lines = []
            max_len = max(len(visible_old), len(visible_new))
            for i in range(max_len):
                old_line = visible_old[i] if i < len(visible_old) else ""
                new_line = visible_new[i] if i < len(visible_new) else ""
                old_display = old_line[:35] + "..." if len(old_line) > 35 else old_line
                new_display = new_line[:35] + "..." if len(new_line) > 35 else new_line
                if old_line != new_line:
                    lines.append(
                        f"  {_RED}{old_display:<38}{_RESET} {_GREEN}{new_display}{_RESET}"
                    )
                else:
                    lines.append(
                        f"  {_DIM}{old_display:<38}{_RESET} {_DIM}{new_display}{_RESET}"
                    )
            if not lines:
                lines = [f"  {_DIM}(no differences){_RESET}"]
            pos = f"  {_DIM}{scroll + 1}-{min(scroll + max_visible, max(len(old_lines), len(new_lines)))}/{max(len(old_lines), len(new_lines))}{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows=scroll){_RESET}\n"
                + "\n".join(lines)
                + f"\n{pos}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + max_visible + 1):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC, _KEY_ENTER):
                        _clear(); return new
                    if key.kind == _KEY_UP:
                        scroll = max(0, scroll - 1)
                    elif key.kind == _KEY_DOWN:
                        max_scroll = max(0, max(len(old_lines), len(new_lines)) - max_visible)
                        scroll = min(max_scroll, scroll + 1)
                    elif key.kind == _KEY_PAGE_UP:
                        scroll = max(0, scroll - max_visible)
                    elif key.kind == _KEY_PAGE_DOWN:
                        max_scroll = max(0, max(len(old_lines), len(new_lines)) - max_visible)
                        scroll = min(max_scroll, scroll + max_visible)
        except (termios.error, OSError):
            return new

    # ── Interactive search ────────────────────────────────────────────

    def interactive_search(self, items: list[str], preview_fn: Callable[[str], str] | None = None,
                           message: str = "Search:") -> str | None:
        """Interactive search with type-to-filter and optional preview."""
        if not self._is_tty:
            return self._select_fallback(message, items[:10]) if items else None
        return self._interactive_search_raw(items, preview_fn, message)

    def _interactive_search_raw(self, items: list[str],
                                preview_fn: Callable[[str], str] | None,
                                message: str) -> str | None:
        fd = self._get_fd()
        query = ""
        idx = 0

        def _filtered() -> list[str]:
            if not query:
                return items
            q = query.lower()
            return [i for i in items if q in i.lower()]

        def _render() -> str:
            flt = _filtered()
            preview_str = ""
            if preview_fn and flt:
                preview_text = preview_fn(flt[idx])
                preview_lines = preview_text.split("\n")[:5]
                preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            lines = []
            for i, item in enumerate(flt):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                display = item[:50] + "..." if len(item) > 50 else item
                lines.append(f"  {prefix}{color}{display}{_RESET}")
            if not lines:
                lines = [f"  {_DIM}(no matches){_RESET}"]
            count = f"  {_DIM}{len(flt)}/{len(items)} results{_RESET}"
            query_display = f"  {_BOLD}Search:{_RESET} {query}{_CYAN}\u2502{_RESET}" if query else f"  {_DIM}Type to search{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows, type=search){_RESET}\n"
                + query_display
                + "\n" + "\n".join(lines)
                + f"\n{count}\n{preview_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(4 + len(items[:50])):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    flt = _filtered()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return None
                    if key.kind == _KEY_ENTER:
                        _clear(); return flt[idx] if flt else None
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % max(len(flt), 1)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % max(len(flt), 1)
                    elif key.kind == _KEY_CHAR:
                        if key.char == "\x7f":
                            query = query[:-1]
                        else:
                            query += key.char
                        idx = 0
                    elif key.kind == _KEY_BACKSPACE:
                        query = query[:-1]
                        idx = 0
        except (termios.error, OSError):
            flt = _filtered()
            return flt[0] if flt else None

    # ── Wizard builder ────────────────────────────────────────────────

    def wizard(self, steps: list[dict[str, str | list[str] | None]],
               message: str = "Wizard") -> dict[str, str]:
        """Multi-step wizard with labeled steps.

        Each step is a dict with:
            'label': step label
            'type': 'input' | 'select' | 'confirm'
            'options': list of options (for select type)
            'default': default value
        """
        if not self._is_tty:
            return {s.get("label", f"step{i}"): str(s.get("default", ""))
                    for i, s in enumerate(steps)}
        return self._wizard_raw(steps, message)

    def _wizard_raw(self, steps: list[dict[str, str | list[str] | None]],
                    message: str) -> dict[str, str]:
        fd = self._get_fd()
        results: dict[str, str] = {}
        current = 0

        def _render() -> str:
            lines = []
            for i, step in enumerate(steps):
                label = str(step.get("label", f"Step {i + 1}"))
                step_type = str(step.get("type", "input"))
                if i < current:
                    lines.append(f"  {_GREEN}\u2713{_RESET} {label}")
                elif i == current:
                    lines.append(f"  {_CYAN}\u25b6{_RESET} {_BOLD}{label}{_RESET}  {_DIM}({step_type}){_RESET}")
                else:
                    lines.append(f"  {_DIM}\u25fb{label}{_RESET}")
            step = steps[current]
            label = str(step.get("label", f"Step {current + 1}"))
            step_type = str(step.get("type", "input"))
            help_text = f"  {_DIM}Step {current + 1}/{len(steps)} — {step_type}{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                + "\n".join(lines)
                + f"\n{help_text}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + len(steps)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while current < len(steps):
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    step = steps[current]
                    label = str(step.get("label", f"step{current}"))
                    step_type = str(step.get("type", "input"))
                    default = str(step.get("default", ""))
                    if step_type == "confirm":
                        result = self._confirm_fallback(f"  {label}?", default.lower() in ("true", "1", "yes"))
                        results[label] = str(result)
                        current += 1
                    elif step_type == "select":
                        options = step.get("options", []) or []
                        if options:
                            result = self._select_fallback(f"  {label}", [str(o) for o in options])
                            results[label] = result
                            current += 1
                        else:
                            current += 1
                    else:
                        result = self._ask_fallback(f"  {label}", default)
                        results[label] = result
                        current += 1
                _clear()
                return results
        except (termios.error, OSError):
            _clear()
            return results

    # ── Spreadsheet editor ────────────────────────────────────────────

    def spreadsheet_editor(self, headers: list[str], rows: list[list[str]],
                           message: str = "Spreadsheet:") -> list[list[str]]:
        """Interactive spreadsheet editor with cell navigation."""
        if not self._is_tty:
            return rows
        return self._spreadsheet_editor_raw(headers, rows, message)

    def _spreadsheet_editor_raw(self, headers: list[str], rows: list[list[str]],
                                message: str) -> list[list[str]]:
        fd = self._get_fd()
        col = 0
        row = 0
        editing = False
        edit_buf = ""
        col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
                     for i, h in enumerate(headers)]

        def _render() -> str:
            lines = []
            header = "  " + "  ".join(f"{_BOLD}{h:<{col_widths[i]}}{_RESET}"
                                      for i, h in enumerate(headers))
            lines.append(header)
            for r_idx, r in enumerate(rows):
                cells = []
                for c_idx, c in enumerate(r):
                    display = c[:col_widths[c_idx]]
                    if r_idx == row and c_idx == col and editing:
                        cells.append(f"{_REVERSE}{_CYAN}{edit_buf}\u2502{_RESET_REVERSE}")
                    elif r_idx == row and c_idx == col:
                        cells.append(f"{_REVERSE}{display}{_RESET_REVERSE}")
                    elif r_idx == row:
                        cells.append(f"{_CYAN}{display}{_RESET}")
                    elif c_idx == col:
                        cells.append(f"{_YELLOW}{display}{_RESET}")
                    else:
                        cells.append(display)
                lines.append("  " + "  ".join(cells))
            pos = f"  {_DIM}row {row + 1}/{len(rows)}, col {col + 1}/{len(headers)}{_RESET}"
            help_text = f"  {_DIM}(arrows=move, enter=edit, tab=new row, esc=done){_RESET}" if not editing else f"  {_DIM}(type=value, enter=confirm, esc=cancel){_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                + "\n".join(lines)
                + f"\n{pos}\n{help_text}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(4 + len(rows) + 2):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if editing:
                        if key.kind == _KEY_ENTER:
                            rows[row][col] = edit_buf
                            editing = False
                        elif key.kind == _KEY_ESC:
                            editing = False
                        elif key.kind == _KEY_CHAR:
                            if key.char == "\x7f":
                                edit_buf = edit_buf[:-1]
                            else:
                                edit_buf += key.char
                        elif key.kind == _KEY_BACKSPACE:
                            edit_buf = edit_buf[:-1]
                    else:
                        if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                            _clear(); return rows
                        if key.kind == _KEY_ENTER:
                            editing = True
                            edit_buf = rows[row][col]
                        if key.kind == _KEY_UP:
                            row = (row - 1) % len(rows)
                        elif key.kind == _KEY_DOWN:
                            row = (row + 1) % len(rows)
                        elif key.kind == _KEY_LEFT:
                            col = (col - 1) % len(headers)
                        elif key.kind == _KEY_RIGHT:
                            col = (col + 1) % len(headers)
                        elif key.kind == _KEY_TAB:
                            new_row = [""] * len(headers)
                            rows.insert(row + 1, new_row)
                            row += 1
                            col = 0
                        elif key.kind == _KEY_DELETE:
                            if len(rows) > 1:
                                rows.pop(row)
                                row = row % len(rows)
        except (termios.error, OSError):
            return rows

    # ── Hierarchical menu ─────────────────────────────────────────────

    def hierarchical_menu(self, menu: dict[str, str | list[str] | dict],
                          message: str = "Menu:") -> str | None:
        """Navigate a hierarchical menu with nested dicts.

        Args:
            menu: nested dict structure. Leaf values are action strings.
                  Nested dicts become submenus.
            message: prompt text

        Returns:
            selected action string or None
        """
        if not self._is_tty:
            return None
        return self._hierarchical_menu_raw(menu, message)

    def _hierarchical_menu_raw(self, menu: dict[str, str | list[str] | dict],
                               message: str) -> str | None:
        fd = self._get_fd()
        path: list[dict] = [menu]
        idx = 0

        def _current() -> dict:
            return path[-1]

        def _items() -> list[str]:
            return list(_current().keys())

        def _render() -> str:
            items = _items()
            breadcrumb = " > ".join(str(k) for k in path[1:])
            breadcrumb_str = f"  {_DIM}{breadcrumb}{_RESET}" if breadcrumb else ""
            lines = []
            for i, item in enumerate(items):
                val = _current()[item]
                is_dir = isinstance(val, dict)
                icon = "\U0001f4c1" if is_dir else "\u25b6"
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{color}{icon} {item}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows, enter=open, backspace=back){_RESET}\n"
                + breadcrumb_str
                + "\n" + "\n".join(lines)
                + f"{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + len(_items())):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    items = _items()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return None
                    if key.kind == _KEY_ENTER:
                        val = _current()[items[idx]]
                        if isinstance(val, dict):
                            path.append(val)
                            idx = 0
                        else:
                            _clear(); return str(val)
                    if key.kind == _KEY_BACKSPACE:
                        if len(path) > 1:
                            path.pop()
                            idx = 0
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % len(items)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % len(items)
        except (termios.error, OSError):
            return None

    # ── Form builder ──────────────────────────────────────────────────

    def form(self, fields: list[dict[str, str | list[str] | None | bool]],
             message: str = "Form") -> dict[str, str]:
        """Multi-field form with validation.

        Each field is a dict with:
            'label': field label
            'type': 'text' | 'select' | 'toggle' | 'password'
            'options': list of options (for select type)
            'default': default value
            'required': bool
        """
        if not self._is_tty:
            return {f.get("label", f"field{i}"): str(f.get("default", ""))
                    for i, f in enumerate(fields)}
        return self._form_raw(fields, message)

    def _form_raw(self, fields: list[dict[str, str | list[str] | None | bool]],
                  message: str) -> dict[str, str]:
        fd = self._get_fd()
        results: dict[str, str] = {}
        idx = 0
        editing = False
        edit_buf = ""

        def _render() -> str:
            lines = []
            for i, field in enumerate(fields):
                label = str(field.get("label", f"Field {i + 1}"))
                field_type = str(field.get("type", "text"))
                default = str(field.get("default", ""))
                required = field.get("required", False)
                value = results.get(label, default)
                req_str = f" {_RED}*{_RESET}" if required else ""
                if i == idx and editing:
                    if field_type == "password":
                        display = "*" * len(edit_buf)
                    else:
                        display = edit_buf
                    lines.append(f"  {_REVERSE}{_CYAN}{label}{req_str}: {display}\u2502{_RESET_REVERSE}{_RESET}")
                elif i == idx:
                    lines.append(f"  {_REVERSE}{_CYAN}{label}{req_str}: {value}{_RESET_REVERSE}{_RESET}")
                else:
                    lines.append(f"  {label}{req_str}: {value}")
            help_text = f"  {_DIM}(arrows=navigate, enter=edit, esc=submit){_RESET}" if not editing else f"  {_DIM}(type=value, enter=confirm, esc=cancel){_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                + "\n".join(lines)
                + f"\n{help_text}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + len(fields) + 1):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    field = fields[idx]
                    label = str(field.get("label", f"Field {idx + 1}"))
                    field_type = str(field.get("type", "text"))
                    default = str(field.get("default", ""))
                    if editing:
                        if key.kind == _KEY_ENTER:
                            if field_type == "toggle":
                                current = results.get(label, default)
                                results[label] = "false" if current == "true" else "true"
                            else:
                                results[label] = edit_buf
                            editing = False
                        elif key.kind == _KEY_ESC:
                            editing = False
                        elif key.kind == _KEY_CHAR:
                            if key.char == "\x7f":
                                edit_buf = edit_buf[:-1]
                            else:
                                edit_buf += key.char
                        elif key.kind == _KEY_BACKSPACE:
                            edit_buf = edit_buf[:-1]
                    else:
                        if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                            _clear(); return results
                        if key.kind == _KEY_ENTER:
                            editing = True
                            edit_buf = results.get(label, default)
                        if key.kind == _KEY_UP:
                            idx = (idx - 1) % len(fields)
                        elif key.kind == _KEY_DOWN:
                            idx = (idx + 1) % len(fields)
        except (termios.error, OSError):
            return results

    # ── Playlist manager ──────────────────────────────────────────────

    def playlist_manager(self, items: list[str], message: str = "Playlist:") -> list[str]:
        """Manage ordered list with move up/down and delete."""
        if not self._is_tty:
            return items
        return self._playlist_manager_raw(items, message)

    def _playlist_manager_raw(self, items: list[str], message: str) -> list[str]:
        fd = self._get_fd()
        idx = 0

        def _render() -> str:
            lines = []
            for i, item in enumerate(items):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                pos = f"{i + 1:>2}. "
                lines.append(f"  {prefix}{color}{pos}{item}{_RESET}")
            if not lines:
                lines = [f"  {_DIM}(empty playlist){_RESET}"]
            help_text = f"  {_DIM}(arrows=move, d=delete, u=up, n=down, enter=done){_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                + "\n".join(lines)
                + f"\n{help_text}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + len(items) + 1):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC, _KEY_ENTER):
                        _clear(); return items
                    if key.kind == _KEY_UP or (key.kind == _KEY_CHAR and key.char == "k"):
                        idx = (idx - 1) % max(len(items), 1)
                    elif key.kind == _KEY_DOWN or (key.kind == _KEY_CHAR and key.char == "j"):
                        idx = (idx + 1) % max(len(items), 1)
                    elif key.kind == _KEY_CHAR and key.char == "u":
                        if idx > 0:
                            items[idx], items[idx - 1] = items[idx - 1], items[idx]
                            idx -= 1
                    elif key.kind == _KEY_CHAR and key.char == "n":
                        if idx < len(items) - 1:
                            items[idx], items[idx + 1] = items[idx + 1], items[idx]
                            idx += 1
                    elif key.kind == _KEY_CHAR and key.char == "d":
                        if items:
                            items.pop(idx)
                            idx = idx % max(len(items), 1)
                    elif key.kind == _KEY_DELETE:
                        if items:
                            items.pop(idx)
                            idx = idx % max(len(items), 1)
        except (termios.error, OSError):
            return items

    # ── Kanban board ──────────────────────────────────────────────────

    def kanban_board(self, columns: dict[str, list[str]],
                     message: str = "Kanban") -> dict[str, list[str]]:
        """Interactive kanban board with move between columns."""
        if not self._is_tty:
            return columns
        return self._kanban_board_raw(columns, message)

    def _kanban_board_raw(self, columns: dict[str, list[str]],
                          message: str) -> dict[str, list[str]]:
        fd = self._get_fd()
        col_idx = 0
        item_idx = 0
        col_names = list(columns.keys())

        def _render() -> str:
            lines = []
            for c, col_name in enumerate(col_names):
                items = columns[col_name]
                col_header = f"  {_BOLD}{col_name}{_RESET}"
                lines.append(col_header)
                for i, item in enumerate(items):
                    prefix = ">>" if c == col_idx and i == item_idx else "  "
                    color = _CYAN if c == col_idx and i == item_idx else ""
                    reset = _RESET if c == col_idx and i == item_idx else ""
                    lines.append(f"    {prefix} {color}{item}{_RESET}")
                if not items:
                    lines.append(f"    {_DIM}(empty){_RESET}")
                lines.append("")
            help_text = f"  {_DIM}(h/l=column, j/k=item, left/right=move, enter=done){_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                + "\n".join(lines)
                + f"\n{help_text}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + len(col_names) * 4):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC, _KEY_ENTER):
                        _clear(); return columns
                    col_items = columns[col_names[col_idx]]
                    if key.kind == _KEY_LEFT or (key.kind == _KEY_CHAR and key.char == "h"):
                        col_idx = (col_idx - 1) % len(col_names)
                        item_idx = min(item_idx, len(columns[col_names[col_idx]]) - 1)
                        item_idx = max(0, item_idx)
                    elif key.kind == _KEY_RIGHT or (key.kind == _KEY_CHAR and key.char == "l"):
                        col_idx = (col_idx + 1) % len(col_names)
                        item_idx = min(item_idx, len(columns[col_names[col_idx]]) - 1)
                        item_idx = max(0, item_idx)
                    elif key.kind == _KEY_UP or (key.kind == _KEY_CHAR and key.char == "k"):
                        item_idx = (item_idx - 1) % max(len(col_items), 1)
                    elif key.kind == _KEY_DOWN or (key.kind == _KEY_CHAR and key.char == "j"):
                        item_idx = (item_idx + 1) % max(len(col_items), 1)
                    elif key.kind == _KEY_LEFT:
                        if col_idx > 0 and col_items:
                            item = col_items.pop(item_idx)
                            columns[col_names[col_idx - 1]].append(item)
                            item_idx = 0
                            col_idx -= 1
                    elif key.kind == _KEY_RIGHT:
                        if col_idx < len(col_names) - 1 and col_items:
                            item = col_items.pop(item_idx)
                            columns[col_names[col_idx + 1]].append(item)
                            item_idx = 0
                            col_idx += 1
        except (termios.error, OSError):
            return columns

    # ── Calendar view ─────────────────────────────────────────────────

    def calendar_view(self, year: int, month: int,
                      events: dict[int, str] | None = None,
                      message: str = "Calendar") -> int | None:
        """Interactive calendar view with day selection.

        Args:
            year: year to display
            month: month to display (1-12)
            events: dict of day -> event description
            message: prompt text

        Returns:
            selected day (1-31) or None
        """
        if not self._is_tty:
            return None
        return self._calendar_view_raw(year, month, events or {}, message)

    def _calendar_view_raw(self, year: int, month: int,
                           events: dict[int, str], message: str) -> int | None:
        import calendar as _cal
        fd = self._get_fd()
        day = 1
        cal = _cal.monthcalendar(year, month)
        month_name = _cal.month_name[month]

        def _render() -> str:
            lines = []
            header = f"  {_BOLD}{month_name} {year}{_RESET}"
            lines.append(header)
            lines.append(f"  {'Mo':>3} {'Tu':>3} {'We':>3} {'Th':>3} {'Fr':>3} {'Sa':>3} {'Su':>3}")
            for week in cal:
                week_str = ""
                for d in week:
                    if d == 0:
                        week_str += "    "
                    elif d == day:
                        week_str += f" {_REVERSE}{_CYAN}{d:>2}{_RESET_REVERSE}"
                    elif d in events:
                        week_str += f" {_YELLOW}{d:>2}{_RESET}"
                    else:
                        week_str += f" {d:>2}"
                lines.append(f"  {week_str}")
            event_str = ""
            if day in events:
                event_str = f"  {_DIM}Event: {events[day]}{_RESET}"
            help_text = f"  {_DIM}(arrows=navigate, enter=select, esc=cancel){_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                + "\n".join(lines)
                + f"\n{event_str}\n{help_text}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(5 + 2):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return None
                    if key.kind == _KEY_ENTER:
                        _clear(); return day
                    if key.kind == _KEY_UP:
                        day = max(1, day - 7)
                    elif key.kind == _KEY_DOWN:
                        day = min(31, day + 7)
                    elif key.kind == _KEY_LEFT:
                        day = max(1, day - 1)
                    elif key.kind == _KEY_RIGHT:
                        day = min(31, day + 1)
        except (termios.error, OSError):
            return None

    # ── Color picker ───────────────────────────────────────────────────

    def color_picker_rgb(self, message: str, default: str = "#ffffff") -> str:
        """Interactive hex color picker with RGB sliders.

        Args:
            message: prompt text
            default: default hex color (e.g. "#ff0000")

        Returns:
            selected hex color string
        """
        if not self._is_tty:
            return self._ask_fallback(message, default)
        return self._color_picker_rgb_raw(message, default)

    def _color_picker_rgb_raw(self, message: str, default: str) -> str:
        fd = self._get_fd()
        r, g, b = self._hex_to_rgb(default)
        focus = 0  # 0=r, 1=g, 2=b
        labels = ["R", "G", "B"]

        def _render() -> str:
            hex_str = f"#{r:02x}{g:02x}{b:02x}"
            bar_w = 20
            def _bar(val: int, color: str) -> str:
                filled = int(val / 255 * bar_w)
                return f"\u2588{color}{'\u2588' * filled}{_RESET}\u2591{' ' * (bar_w - filled)}"
            r_bar = _bar(r, _RED if focus == 0 else "")
            g_bar = _bar(g, _GREEN if focus == 1 else "")
            b_bar = _bar(b, _CYAN if focus == 2 else "")
            lines = [
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_BOLD}{hex_str}{_RESET}",
                f"{_ERASE_LINE}\r  {_RED if focus==0 else ''}R {r:>3}{_RESET} {r_bar}",
                f"{_ERASE_LINE}\r  {_GREEN if focus==1 else ''}G {g:>3}{_RESET} {g_bar}",
                f"{_ERASE_LINE}\r  {_CYAN if focus==2 else ''}B {b:>3}{_RESET} {b_bar}",
                f"{_ERASE_LINE}\r  {_DIM}Left/Right: channel  Up/Down: adjust  Enter: confirm{_RESET}{_HIDE_CURSOR}",
            ]
            return "\n".join(lines)

        def _clear() -> None:
            for _ in range(5):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return default
                    if key.kind == _KEY_ENTER:
                        _clear(); return f"#{r:02x}{g:02x}{b:02x}"
                    if key.kind == _KEY_LEFT:
                        focus = (focus + 1) % 3
                    elif key.kind == _KEY_RIGHT:
                        focus = (focus + 2) % 3
                    elif key.kind == _KEY_UP:
                        vals = [r, g, b]
                        vals[focus] = min(255, vals[focus] + 5)
                        r, g, b = vals
                    elif key.kind == _KEY_DOWN:
                        vals = [r, g, b]
                        vals[focus] = max(0, vals[focus] - 5)
                        r, g, b = vals
        except (termios.error, OSError):
            return default

    def _hex_to_rgb(self, hex_str: str) -> tuple[int, int, int]:
        h = hex_str.lstrip("#")
        if len(h) == 3:
            h = h[0]*2 + h[1]*2 + h[2]*2
        try:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        except (ValueError, IndexError):
            return (255, 255, 255)

    # ── Confirm with timeout ───────────────────────────────────────────

    def confirm_timeout(self, message: str, timeout: float = 5.0,
                        default: bool = True) -> bool:
        """Confirm with an auto-timeout. Returns default if not answered.

        Args:
            message: prompt text
            timeout: seconds before auto-confirming
            default: value to return on timeout

        Returns:
            True/False based on user choice or timeout
        """
        if not self._is_tty:
            return self._confirm_fallback(message, default)
        return self._confirm_timeout_raw(message, timeout, default)

    def _confirm_timeout_raw(self, message: str, timeout: float,
                             default: bool) -> bool:
        fd = self._get_fd()
        selected = not default  # opposite of default for visual
        import select as _select
        import time as _t

        def _render(remaining: float) -> str:
            if selected:
                yes_text = f"{_REVERSE}{_GREEN} Yes {_RESET_REVERSE}{_RESET}"
                no_text = " No "
            else:
                yes_text = " Yes "
                no_text = f"{_REVERSE}{_RED} No {_RESET_REVERSE}{_RESET}"
            bar_w = 20
            filled = int(remaining / timeout * bar_w)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  "
                f"{yes_text}  {no_text}  "
                f"{_DIM}{bar} {remaining:.0f}s{_RESET}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                start = _t.monotonic()
                while True:
                    elapsed = _t.monotonic() - start
                    remaining = max(0, timeout - elapsed)
                    if remaining <= 0:
                        _clear(); return default
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render(remaining)}")
                    sys.stdout.flush()
                    r, _, _ = _select.select([fd], [], [], 0.1)
                    if r:
                        key = _read_raw_key(fd)
                        if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                            _clear(); return default
                        if key.kind == _KEY_ENTER:
                            _clear(); return selected
                        if key.kind in (_KEY_LEFT, _KEY_RIGHT):
                            selected = not selected
                        if key.kind == _KEY_CHAR:
                            if key.char in ("y", "Y"):
                                _clear(); return True
                            if key.char in ("n", "N"):
                                _clear(); return False
        except (termios.error, OSError):
            return default

    # ── Spin until ─────────────────────────────────────────────────────

    def spin_until(self, message: str, async_fn: "Callable[[], Any]",
                   check: "Callable[[Any], bool]",
                   interval: float = 0.1, timeout: float = 0) -> Any:
        """Wait for an async function's result to satisfy a condition.

        Args:
            message: message to display
            async_fn: function that returns a value each poll
            check: function that returns True when result is acceptable
            interval: poll interval in seconds
            timeout: max seconds (0 = forever)

        Returns:
            the result from async_fn when check passes, or None on timeout
        """
        if not self._is_tty:
            import time as _t
            start = _t.monotonic()
            while True:
                result = async_fn()
                if check(result):
                    return result
                _t.sleep(interval)
                if timeout and _t.monotonic() - start >= timeout:
                    return None
        return self._spin_until_raw(message, async_fn, check, interval, timeout)

    def _spin_until_raw(self, message: str, async_fn: "Callable[[], Any]",
                        check: "Callable[[Any], bool]",
                        interval: float, timeout: float) -> "Any":
        import time as _t
        fd = self._get_fd()
        frames = ["\u25cf", "\u25cf\u25cf", "\u25cf\u25cf\u25cf", "\u25cf"]
        idx = 0
        start = _t.monotonic()
        try:
            with _RawTerminal(fd):
                while True:
                    result = async_fn()
                    if check(result):
                        sys.stdout.write(
                            f"\r{_ERASE_LINE}\r  {_GREEN}\u2714{_RESET} {_BOLD}{message}{_RESET} "
                            f"{_GREEN}done{_RESET}{_SHOW_CURSOR}\n"
                        )
                        sys.stdout.flush()
                        return result
                    elapsed = _t.monotonic() - start
                    if timeout and elapsed >= timeout:
                        sys.stdout.write(
                            f"\r{_ERASE_LINE}\r  {_RED}\u2716{_RESET} {_BOLD}{message}{_RESET} "
                            f"{_RED}timeout{_RESET}{_SHOW_CURSOR}\n"
                        )
                        sys.stdout.flush()
                        return None
                    frame = frames[idx % len(frames)]
                    sys.stdout.write(
                        f"\r{_ERASE_LINE}\r  {_CYAN}{frame}{_RESET} {_BOLD}{message}{_RESET} "
                        f"{_DIM}{elapsed:.1f}s{_RESET}{_HIDE_CURSOR}"
                    )
                    sys.stdout.flush()
                    _t.sleep(interval)
                    idx += 1
        except (termios.error, OSError):
            import time as _t
            start = _t.monotonic()
            while True:
                result = async_fn()
                if check(result):
                    return result
                _t.sleep(interval)
                if timeout and _t.monotonic() - start >= timeout:
                    return None

    # ── Time picker ────────────────────────────────────────────────────

    def time_picker(self, message: str, default: str = "") -> str:
        """Interactive time picker with hour/minute/AM-PM navigation."""
        import datetime as _dt
        if not self._is_tty:
            now = _dt.datetime.now()
            return self._ask_fallback(message, default or now.strftime("%I:%M %p"))
        return self._time_picker_raw(message, default)

    def _time_picker_raw(self, message: str, default: str) -> str:
        import datetime as _dt
        fd = self._get_fd()
        now = _dt.datetime.now()
        h, m = now.hour, now.minute
        ampm = "AM" if h < 12 else "PM"
        if default:
            try:
                parts = default.replace("  ", " ").split()
                hm = parts[0].split(":")
                h, m = int(hm[0]), int(hm[1])
                ampm = parts[1].upper() if len(parts) > 1 else ampm
            except (ValueError, IndexError):
                pass
        if h > 12: h -= 12
        if h == 0: h = 12
        focus = 0

        def _render() -> str:
            hc = _CYAN if focus == 0 else ""
            mc = _CYAN if focus == 1 else ""
            ac = _CYAN if focus == 2 else ""
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  "
                f"{hc}{_BOLD}{h:>2}{_RESET}:"
                f"{mc}{_BOLD}{m:>02}{_RESET} "
                f"{ac}{_BOLD}{ampm}{_RESET}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return default or now.strftime("%I:%M %p")
                    if key.kind == _KEY_ENTER:
                        _clear(); return f"{h:>2}:{m:02d} {ampm}"
                    if key.kind == _KEY_LEFT:
                        focus = (focus - 1) % 3
                    elif key.kind == _KEY_RIGHT:
                        focus = (focus + 1) % 3
                    elif key.kind == _KEY_UP:
                        if focus == 0: h = h % 12 + 1
                        elif focus == 1: m = (m + 5) % 60
                        else: ampm = "PM" if ampm == "AM" else "AM"
                    elif key.kind == _KEY_DOWN:
                        if focus == 0: h = (h - 2) % 12 + 1
                        elif focus == 1: m = (m - 5) % 60
                        else: ampm = "PM" if ampm == "AM" else "AM"
        except (termios.error, OSError):
            return default or now.strftime("%I:%M %p")

    # ── Progress ETA ───────────────────────────────────────────────────

    def progress_eta(self, label: str, current: int, total: int,
                     elapsed: float = 0) -> None:
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

    # ── Select with search ─────────────────────────────────────────────

    def select_with_search(self, title: str, options: list[str]) -> str:
        """Select with prominent search bar and live filtering."""
        if not options:
            return ""
        if len(options) == 1:
            return options[0]
        if not self._is_tty:
            return self._select_fallback(title, options)
        return self._select_with_search_raw(title, options)

    def _select_with_search_raw(self, title: str, options: list[str]) -> str:
        fd = self._get_fd()
        query = ""
        cursor = 0
        scroll = 0
        max_visible = min(len(options), 15)

        def _get_filtered() -> list[str]:
            if query:
                return [o for o in options if query.lower() in o.lower()]
            return list(options)

        def _render() -> list[str]:
            filtered = _get_filtered()
            lines: list[str] = []
            lines.append(f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}")
            lines.append(f"{_ERASE_LINE}\r  {_BOLD}\u26b2 Search:{_RESET} {_BOLD}{query}{_RESET}\u2502{_HIDE_CURSOR}")
            if not filtered:
                lines.append(f"{_ERASE_LINE}\r  {_DIM}No matching options{_RESET}")
            else:
                for i in range(min(max_visible, len(filtered))):
                    idx = scroll + i
                    if idx >= len(filtered): break
                    text = _truncate(filtered[idx], 50)
                    if idx == cursor:
                        lines.append(f"{_ERASE_LINE}\r{_REVERSE} > {text}{_RESET_REVERSE}{_RESET}")
                    else:
                        lines.append(f"{_ERASE_LINE}\r   {text}")
            lines.append(f"{_ERASE_LINE}\r  {_DIM}Enter: select  Esc: cancel  Type to filter{_RESET}{_HIDE_CURSOR}")
            return lines

        def _write_lines(lines: list[str]) -> None:
            sys.stdout.write(_HIDE_CURSOR)
            for line in lines: sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        def _clear(line_count: int) -> None:
            for _ in range(line_count): sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR); sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                prev_count = 0
                while True:
                    _clear(prev_count)
                    filtered = _get_filtered()
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count); return options[0]
                    if key.kind == _KEY_ENTER:
                        _clear(prev_count)
                        return filtered[cursor] if 0 <= cursor < len(filtered) else options[0]
                    if key.kind == _KEY_UP:
                        cursor = max(0, cursor - 1)
                        if cursor < scroll: scroll = cursor
                    elif key.kind == _KEY_DOWN:
                        cursor = min(len(filtered) - 1, cursor + 1)
                        if cursor >= scroll + max_visible: scroll = cursor - max_visible + 1
                    elif key.kind == _KEY_BACKSPACE:
                        if query: query = query[:-1]; cursor = 0; scroll = 0
                    elif key.kind == _KEY_CHAR:
                        query += key.char; cursor = 0; scroll = 0
        except (termios.error, OSError):
            return self._select_fallback(title, options)

    # ── Table select ───────────────────────────────────────────────────

    def table_select(self, headers: list[str], rows: list[list[str]],
                     title: str = "Select row") -> int | None:
        """Interactive table with row selection. Returns selected row index."""
        if not rows:
            return None
        if not self._is_tty:
            return 0
        return self._table_select_raw(headers, rows, title)

    def _table_select_raw(self, headers: list[str], rows: list[list[str]],
                          title: str) -> int | None:
        fd = self._get_fd()
        cursor = 0
        scroll = 0
        max_visible = min(len(rows), 12)
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths): widths[i] = max(widths[i], len(str(cell)))

        def _render() -> list[str]:
            lines: list[str] = []
            lines.append(f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}")
            lines.append(f"{_ERASE_LINE}\r  {'  '.join(f'{_BOLD}{h.ljust(w)}{_RESET}' for h, w in zip(headers, widths))}")
            lines.append(f"{_ERASE_LINE}\r  {_DIM}{'  '.join('\u2500' * w for w in widths)}{_RESET}")
            for i in range(min(max_visible, len(rows))):
                idx = scroll + i
                if idx >= len(rows): break
                cells = "  ".join(str(rows[idx][j]).ljust(widths[j]) for j in range(min(len(widths), len(rows[idx]))))
                if idx == cursor:
                    lines.append(f"{_ERASE_LINE}\r{_REVERSE} > {cells}{_RESET_REVERSE}{_RESET}")
                else:
                    lines.append(f"{_ERASE_LINE}\r   {cells}")
            lines.append(f"{_ERASE_LINE}\r  {_DIM}Enter: select  Esc: cancel{_RESET}{_HIDE_CURSOR}")
            return lines

        def _write_lines(lines: list[str]) -> None:
            sys.stdout.write(_HIDE_CURSOR)
            for line in lines: sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        def _clear(line_count: int) -> None:
            for _ in range(line_count): sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR); sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                prev_count = 0
                while True:
                    _clear(prev_count)
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count); return None
                    if key.kind == _KEY_ENTER:
                        _clear(prev_count); return cursor if 0 <= cursor < len(rows) else None
                    if key.kind == _KEY_UP:
                        cursor = max(0, cursor - 1)
                        if cursor < scroll: scroll = cursor
                    elif key.kind == _KEY_DOWN:
                        cursor = min(len(rows) - 1, cursor + 1)
                        if cursor >= scroll + max_visible: scroll = cursor - max_visible + 1
        except (termios.error, OSError):
            return 0

    # ── Year picker ────────────────────────────────────────────────────

    def year_picker(self, message: str, default: int = 0,
                    min_year: int = 1900, max_year: int = 2100) -> int:
        """Interactive year picker with arrow keys.

        Args:
            message: prompt text
            default: default year (0 = current year)
            min_year: minimum selectable year
            max_year: maximum selectable year

        Returns:
            selected year as integer
        """
        import datetime as _dt
        if not self._is_tty:
            return int(self._ask_fallback(message, str(default or _dt.date.today().year)))
        return self._year_picker_raw(message, default or _dt.date.today().year, min_year, max_year)

    def _year_picker_raw(self, message: str, year: int, min_y: int, max_y: int) -> int:
        fd = self._get_fd()

        def _render() -> str:
            return f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_BOLD}{_CYAN}{year:>4}{_RESET}{_HIDE_CURSOR}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return year
                    if key.kind == _KEY_ENTER:
                        _clear(); return year
                    if key.kind == _KEY_UP:
                        year = min(max_y, year + 1)
                    elif key.kind == _KEY_DOWN:
                        year = max(min_y, year - 1)
                    elif key.kind == _KEY_HOME:
                        year = min_y
                    elif key.kind == _KEY_END:
                        year = max_y
        except (termios.error, OSError):
            return year

    # ── Month picker ───────────────────────────────────────────────────

    def month_picker(self, message: str, default: int = 0) -> int:
        """Interactive month picker (1-12) with names.

        Args:
            message: prompt text
            default: default month (0 = current month)

        Returns:
            selected month as integer (1-12)
        """
        import datetime as _dt
        if not self._is_tty:
            return int(self._ask_fallback(message, str(default or _dt.date.today().month)))
        return self._month_picker_raw(message, default or _dt.date.today().month)

    def _month_picker_raw(self, message: str, month: int) -> int:
        fd = self._get_fd()
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]

        def _render() -> str:
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  "
                f"{_BOLD}{_CYAN}{months[month - 1]:>9}{_RESET} ({month:>2}/12){_HIDE_CURSOR}"
            )

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return month
                    if key.kind == _KEY_ENTER:
                        _clear(); return month
                    if key.kind == _KEY_UP:
                        month = month % 12 + 1
                    elif key.kind == _KEY_DOWN:
                        month = (month - 2) % 12 + 1
                    elif key.kind == _KEY_LEFT:
                        month = max(1, month - 1)
                    elif key.kind == _KEY_RIGHT:
                        month = min(12, month + 1)
        except (termios.error, OSError):
            return month

    # ── Confirm list ───────────────────────────────────────────────────

    def confirm_list(self, title: str, items: list[str],
                     default: bool = True) -> list[str]:
        """Confirm each item in a list with y/N.

        Args:
            title: prompt text
            items: list of items to confirm
            default: default answer for each item

        Returns:
            list of confirmed items
        """
        if not items:
            return []
        if not self._is_tty:
            return self._confirm_multi_fallback(title, items, default)
        return self._confirm_list_raw(title, items, default)

    def _confirm_list_raw(self, title: str, items: list[str],
                          default: bool) -> list[str]:
        fd = self._get_fd()
        cursor = 0
        answers: dict[int, bool] = {}

        def _render() -> list[str]:
            lines: list[str] = []
            lines.append(f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}")
            for i, item in enumerate(items):
                ans = answers.get(i, default)
                if ans:
                    icon = f"{_GREEN}\u2714{_RESET}"
                else:
                    icon = f"{_RED}\u2718{_RESET}"
                if i == cursor:
                    lines.append(f"{_ERASE_LINE}\r{_REVERSE} {icon} {item}{_RESET_REVERSE}{_RESET}")
                else:
                    lines.append(f"{_ERASE_LINE}\r  {icon} {item}")
            lines.append(f"{_ERASE_LINE}\r  {_DIM}Space: toggle  Enter: confirm  y/n: toggle all{_RESET}{_HIDE_CURSOR}")
            return lines

        def _write_lines(lines: list[str]) -> None:
            sys.stdout.write(_HIDE_CURSOR)
            for line in lines: sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        def _clear(line_count: int) -> None:
            for _ in range(line_count): sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR); sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                prev_count = 0
                while True:
                    _clear(prev_count)
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count); return []
                    if key.kind == _KEY_ENTER:
                        _clear(prev_count)
                        return [items[i] for i in range(len(items)) if answers.get(i, default)]
                    if key.kind == _KEY_UP:
                        cursor = max(0, cursor - 1)
                    elif key.kind == _KEY_DOWN:
                        cursor = min(len(items) - 1, cursor + 1)
                    elif key.kind == _KEY_SPACE:
                        answers[cursor] = not answers.get(cursor, default)
                    elif key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            for i in range(len(items)): answers[i] = True
                        elif key.char in ("n", "N"):
                            for i in range(len(items)): answers[i] = False
        except (termios.error, OSError):
            return self._confirm_multi_fallback(title, items, default)

    # ── Table edit ─────────────────────────────────────────────────────

    def table_edit(self, headers: list[str], rows: list[list[str]],
                   title: str = "Edit table") -> list[list[str]]:
        """Interactive table with cell editing.

        Args:
            headers: column headers
            rows: table data rows
            title: prompt text

        Returns:
            edited rows
        """
        if not rows:
            return []
        if not self._is_tty:
            return rows
        return self._table_edit_raw(headers, rows, title)

    def _table_edit_raw(self, headers: list[str], rows: list[list[str]],
                        title: str) -> list[list[str]]:
        fd = self._get_fd()
        cursor_row = 0
        cursor_col = 0
        editing = False
        buf: list[str] = []
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths): widths[i] = max(widths[i], len(str(cell)))

        def _render() -> list[str]:
            lines: list[str] = []
            lines.append(f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}  {_DIM}(Tab: cell  Enter: edit/save  Esc: done){_RESET}")
            lines.append(f"{_ERASE_LINE}\r  {'  '.join(f'{_BOLD}{h.ljust(w)}{_RESET}' for h, w in zip(headers, widths))}")
            lines.append(f"{_ERASE_LINE}\r  {_DIM}{'  '.join('\u2500' * w for w in widths)}{_RESET}")
            for r_idx, row in enumerate(rows):
                cells = []
                for c_idx in range(min(len(widths), len(row))):
                    val = "".join(buf) if editing and r_idx == cursor_row and c_idx == cursor_col else str(row[c_idx])
                    val = val.ljust(widths[c_idx])
                    if r_idx == cursor_row and c_idx == cursor_col:
                        if editing:
                            cells.append(f"{_REVERSE}{val}{_RESET_REVERSE}")
                        else:
                            cells.append(f"{_CYAN}{val}{_RESET}")
                    else:
                        cells.append(val)
                lines.append(f"{_ERASE_LINE}\r  {'  '.join(cells)}")
            return lines

        def _write_lines(lines: list[str]) -> None:
            sys.stdout.write(_HIDE_CURSOR)
            for line in lines: sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        def _clear(line_count: int) -> None:
            for _ in range(line_count): sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR); sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                prev_count = 0
                while True:
                    _clear(prev_count)
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        if editing:
                            editing = False; buf.clear()
                        else:
                            _clear(prev_count); return rows
                    elif key.kind == _KEY_ENTER:
                        if editing:
                            rows[cursor_row][cursor_col] = "".join(buf)
                            editing = False; buf.clear()
                        else:
                            editing = True; buf = list(str(rows[cursor_row][cursor_col]))
                    elif not editing:
                        if key.kind == _KEY_TAB:
                            cursor_col = (cursor_col + 1) % len(widths)
                        elif key.kind == _KEY_UP:
                            cursor_row = max(0, cursor_row - 1)
                        elif key.kind == _KEY_DOWN:
                            cursor_row = min(len(rows) - 1, cursor_row + 1)
                        elif key.kind == _KEY_LEFT:
                            cursor_col = max(0, cursor_col - 1)
                        elif key.kind == _KEY_RIGHT:
                            cursor_col = min(len(widths) - 1, cursor_col + 1)
                    else:
                        if key.kind == _KEY_BACKSPACE:
                            if buf: buf.pop()
                        elif key.kind == _KEY_DELETE:
                            if buf: buf.clear()
                        elif key.kind == _KEY_CHAR:
                            buf.append(key.char)
        except (termios.error, OSError):
            return rows

    # ── Duration picker ────────────────────────────────────────────────

    def duration_picker(self, message: str, default: int = 0) -> int:
        """Interactive duration picker in seconds with h/m/s display.

        Args:
            message: prompt text
            default: default duration in seconds

        Returns:
            selected duration in seconds
        """
        if not self._is_tty:
            return int(self._ask_fallback(message, str(default)))
        return self._duration_picker_raw(message, default)

    def _duration_picker_raw(self, message: str, total: int) -> int:
        fd = self._get_fd()
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        focus = 0  # 0=h, 1=m, 2=s

        def _fmt() -> str:
            return f"{h:>2}h {m:>02}m {s:>02}s"

        def _render() -> str:
            parts = [f"{h:>2}h", f"{m:>02}m", f"{s:>02}s"]
            colored = []
            for i, p in enumerate(parts):
                if i == focus:
                    colored.append(f"{_BOLD}{_CYAN}{p}{_RESET}")
                else:
                    colored.append(p)
            return f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {' '.join(colored)}{_HIDE_CURSOR}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return total
                    if key.kind == _KEY_ENTER:
                        _clear(); return h * 3600 + m * 60 + s
                    if key.kind == _KEY_LEFT:
                        focus = (focus - 1) % 3
                    elif key.kind == _KEY_RIGHT:
                        focus = (focus + 1) % 3
                    elif key.kind == _KEY_UP:
                        if focus == 0: h = min(99, h + 1)
                        elif focus == 1: m = (m + 5) % 60
                        else: s = (s + 5) % 60
                    elif key.kind == _KEY_DOWN:
                        if focus == 0: h = max(0, h - 1)
                        elif focus == 1: m = (m - 5) % 60
                        else: s = (s - 5) % 60
        except (termios.error, OSError):
            return total

    # ── Confirm text ───────────────────────────────────────────────────

    def confirm_text(self, message: str, target: str,
                     hint: str = "") -> bool:
        """Confirm by typing exact text.

        Args:
            message: prompt text
            target: text the user must type
            hint: optional hint shown to user

        Returns:
            True if typed text matches target
        """
        if not self._is_tty:
            raw = self._ask_fallback(f"{message} (type '{target}')", "")
            return raw.strip() == target
        return self._confirm_text_raw(message, target, hint)

    def _confirm_text_raw(self, message: str, target: str, hint: str) -> bool:
        fd = self._get_fd()
        buf: list[str] = []

        def _render() -> str:
            typed = "".join(buf)
            match = typed == target
            color = _GREEN if match else _RED
            h = f"\n  {_DIM}{hint}{_RESET}" if hint else ""
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}{h}\n"
                f"  {_DIM}Type{_RESET} {_BOLD}{target}{_RESET} {_DIM}to confirm{_RESET}: "
                f"{color}{typed}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            if hint: sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return False
                    if key.kind == _KEY_ENTER:
                        _clear(); return "".join(buf) == target
                    if key.kind == _KEY_BACKSPACE:
                        if buf: buf.pop()
                    elif key.kind == _KEY_CHAR:
                        buf.append(key.char)
        except (termios.error, OSError):
            raw = self._ask_fallback(f"{message} (type '{target}')", "")
            return raw.strip() == target

    # ── Week picker ────────────────────────────────────────────────────

    def week_picker(self, message: str, default: int = 0) -> int:
        """Interactive week-of-year picker (1-52).

        Args:
            message: prompt text
            default: default week (0 = current week)

        Returns:
            selected week number (1-52)
        """
        import datetime as _dt
        if not self._is_tty:
            return int(self._ask_fallback(message, str(default or _dt.date.today().isocalendar()[1])))
        return self._week_picker_raw(message, default or _dt.date.today().isocalendar()[1])

    def _week_picker_raw(self, message: str, week: int) -> int:
        fd = self._get_fd()

        def _render() -> str:
            import datetime as _dt
            jan1 = _dt.date(_dt.date.today().year, 1, 1)
            start = jan1 + _dt.timedelta(weeks=week - 1)
            end = start + _dt.timedelta(days=6)
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  "
                f"{_BOLD}{_CYAN}Week {week:>2}{_RESET}  "
                f"{_DIM}{start.strftime('%b %d')} - {end.strftime('%b %d')}{_RESET}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return week
                    if key.kind == _KEY_ENTER:
                        _clear(); return week
                    if key.kind == _KEY_UP:
                        week = min(52, week + 1)
                    elif key.kind == _KEY_DOWN:
                        week = max(1, week - 1)
        except (termios.error, OSError):
            return week

    # ── Quarter picker ─────────────────────────────────────────────────

    def quarter_picker(self, message: str, default: int = 0) -> int:
        """Interactive quarter picker (1-4).

        Args:
            message: prompt text
            default: default quarter (0 = current quarter)

        Returns:
            selected quarter (1-4)
        """
        import datetime as _dt
        if not self._is_tty:
            return int(self._ask_fallback(message, str(default or (_dt.date.today().month - 1) // 3 + 1)))
        return self._quarter_picker_raw(message, default or (_dt.date.today().month - 1) // 3 + 1)

    def _quarter_picker_raw(self, message: str, q: int) -> int:
        fd = self._get_fd()
        quarters = {1: "Q1 (Jan-Mar)", 2: "Q2 (Apr-Jun)", 3: "Q3 (Jul-Sep)", 4: "Q4 (Oct-Dec)"}

        def _render() -> str:
            return f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_BOLD}{_CYAN}{quarters[q]}{_RESET}{_HIDE_CURSOR}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return q
                    if key.kind == _KEY_ENTER:
                        _clear(); return q
                    if key.kind in (_KEY_UP, _KEY_RIGHT):
                        q = q % 4 + 1
                    elif key.kind in (_KEY_DOWN, _KEY_LEFT):
                        q = (q - 2) % 4 + 1
        except (termios.error, OSError):
            return q

    # ── Confirm delete ─────────────────────────────────────────────────

    def confirm_delete(self, item: str, count: int = 1) -> bool:
        """Confirm deletion of item(s) with type-to-confirm.

        Args:
            item: name/description of item to delete
            count: number of items (for plural display)

        Returns:
            True if confirmed
        """
        word = f"{count} items" if count != 1 else item
        return self.confirm_dangerous(f"Delete {word}", phrase="delete")

    # ── Confirm overwrite ──────────────────────────────────────────────

    def confirm_overwrite(self, path: str) -> bool:
        """Confirm overwriting an existing file.

        Args:
            path: file path to overwrite

        Returns:
            True if confirmed
        """
        return self.confirm_dangerous(f"Overwrite {path}", phrase="overwrite")

    # ── Progress ring ──────────────────────────────────────────────────

    def progress_ring(self, label: str, current: int, total: int) -> None:
        """Display a circular progress indicator with Unicode blocks.

        Args:
            label: label text
            current: current progress value
            total: total target value
        """
        frac = current / max(total, 1)
        blocks = ["\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588"]
        idx = int(frac * (len(blocks) - 1))
        ring = blocks[idx] * 8
        pct = f"{frac * 100:.0f}%"
        self._io.write(f"  {_CYAN}{label}{_RESET} {_BOLD}{ring}{_RESET} {pct}")

    # ── Timezone picker ────────────────────────────────────────────────

    def timezone_picker(self, message: str, default: str = "UTC") -> str:
        """Interactive timezone picker from common timezones.

        Args:
            message: prompt text
            default: default timezone

        Returns:
            selected timezone string
        """
        timezones = [
            "UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific",
            "Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Moscow",
            "Asia/Tokyo", "Asia/Shanghai", "Asia/Kolkata", "Asia/Dubai",
            "Australia/Sydney", "Pacific/Auckland", "America/Sao_Paulo",
            "Africa/Cairo", "Africa/Lagos", "Asia/Singapore", "Asia/Seoul",
        ]
        if not self._is_tty:
            return self._ask_fallback(message, default)
        return self._select_with_search_raw(message, timezones)

    # ── Currency picker ────────────────────────────────────────────────

    def currency_picker(self, message: str, default: str = "USD") -> str:
        """Interactive currency picker from common currencies.

        Args:
            message: prompt text
            default: default currency code

        Returns:
            selected 3-letter currency code
        """
        currencies = [
            "USD - US Dollar", "EUR - Euro", "GBP - British Pound",
            "JPY - Japanese Yen", "CNY - Chinese Yuan", "KRW - Korean Won",
            "INR - Indian Rupee", "BRL - Brazilian Real", "CAD - Canadian Dollar",
            "AUD - Australian Dollar", "CHF - Swiss Franc", "MXN - Mexican Peso",
            "SGD - Singapore Dollar", "HKD - Hong Kong Dollar", "SEK - Swedish Krona",
            "NOK - Norwegian Krone", "DKK - Danish Krone", "PLN - Polish Zloty",
            "THB - Thai Baht", "ZAR - South African Rand",
        ]
        if not self._is_tty:
            return self._ask_fallback(message, default)
        result = self._select_with_search_raw(message, currencies)
        return result[:3] if result else default

    # ── Language picker ────────────────────────────────────────────────

    def language_picker(self, message: str, default: str = "en") -> str:
        """Interactive language picker from common languages.

        Args:
            message: prompt text
            default: default language code

        Returns:
            selected 2-letter language code
        """
        languages = [
            "en - English", "es - Spanish", "fr - French", "de - German",
            "it - Italian", "pt - Portuguese", "ru - Russian", "zh - Chinese",
            "ja - Japanese", "ko - Korean", "ar - Arabic", "hi - Hindi",
            "nl - Dutch", "sv - Swedish", "pl - Polish", "tr - Turkish",
            "vi - Vietnamese", "th - Thai", "uk - Ukrainian", "cs - Czech",
        ]
        if not self._is_tty:
            return self._ask_fallback(message, default)
        result = self._select_with_search_raw(message, languages)
        return result[:2] if result else default

    # ── Confirm with preview ───────────────────────────────────────────

    def confirm_with_preview(self, message: str, preview: str,
                             default: bool = False) -> bool:
        """Confirm with a preview of what will happen.

        Args:
            message: confirmation prompt
            preview: text preview of the action/result
            default: default answer

        Returns:
            True if confirmed
        """
        if not self._is_tty:
            return self._confirm_fallback(message, default)
        return self._confirm_with_preview_raw(message, preview, default)

    def _confirm_with_preview_raw(self, message: str, preview: str,
                                  default: bool) -> bool:
        fd = self._get_fd()
        selected = not default

        def _render() -> str:
            if selected:
                yes_text = f"{_REVERSE}{_GREEN} Yes {_RESET_REVERSE}{_RESET}"
                no_text = " No "
            else:
                yes_text = " Yes "
                no_text = f"{_REVERSE}{_RED} No {_RESET_REVERSE}{_RESET}"
            preview_lines = preview.split("\n")[:5]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                f"{preview_str}\n"
                f"  {yes_text}  {no_text}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            line_count = 2 + min(len(preview.split("\n")), 5)
            for _ in range(line_count):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return default
                    if key.kind == _KEY_ENTER:
                        _clear(); return selected
                    if key.kind in (_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear(); return True
                        if key.char in ("n", "N"):
                            _clear(); return False
        except (termios.error, OSError):
            return default

    def progress_bar(self, label: str, current: int, total: int,
                     width: int = 30) -> None:
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

    # ── Date range picker ──────────────────────────────────────────────

    def date_range_picker(self, message: str,
                          default_start: str = "", default_end: str = "") -> tuple[str, str]:
        """Pick a date range (start and end dates).

        Args:
            message: prompt text
            default_start: default start date (YYYY-MM-DD)
            default_end: default end date (YYYY-MM-DD)

        Returns:
            tuple of (start_date, end_date) strings
        """
        if not self._is_tty:
            s = self._ask_fallback(f"{message} start", default_start or "2025-01-01")
            e = self._ask_fallback(f"{message} end", default_end or "2025-12-31")
            return (s.strip(), e.strip())
        return self._date_range_picker_raw(message, default_start, default_end)

    def _date_range_picker_raw(self, message: str, ds: str, de: str) -> tuple[str, str]:
        import datetime as _dt
        today = _dt.date.today()
        if not ds:
            ds = today.strftime("%Y-%m-%d")
        if not de:
            de = today.strftime("%Y-%m-%d")
        start = self.date_picker(f"{message} start date", default=ds)
        end = self.date_picker(f"{message} end date", default=de)
        return (start, end)

    # ── Color picker ───────────────────────────────────────────────────

    def color_picker(self, message: str, default: str = "#ffffff") -> str:
        """Interactive color picker with palette and hex input.

        Args:
            message: prompt text
            default: default hex color

        Returns:
            selected hex color string
        """
        if not self._is_tty:
            return self._ask_fallback(message, default)
        return self._color_picker_raw(message, default)

    def _color_picker_raw(self, message: str, default: str) -> str:
        fd = self._get_fd()
        palette = [
            ("#ff0000", "Red"), ("#ff8000", "Orange"), ("#ffff00", "Yellow"),
            ("#00ff00", "Green"), ("#00ffff", "Cyan"), ("#0080ff", "Blue"),
            ("#8000ff", "Purple"), ("#ff00ff", "Magenta"), ("#ffffff", "White"),
            ("#808080", "Gray"), ("#000000", "Black"), ("#804000", "Brown"),
        ]
        colors = [c[0] for c in palette]
        names = [c[1] for c in palette]
        idx = 0
        if default in colors:
            idx = colors.index(default)

        def _render() -> str:
            c = colors[idx]
            n = names[idx]
            r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows, enter){_RESET}\n"
                f"  {c} {n}  "
                f"{_BOLD}████{_RESET}  "
                f"{_BOLD}{_CYAN}>>{_RESET} {c}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(2):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return colors[idx]
                    if key.kind == _KEY_ENTER:
                        _clear(); return colors[idx]
                    if key.kind in (_KEY_UP, _KEY_RIGHT):
                        idx = (idx + 1) % len(colors)
                    elif key.kind in (_KEY_DOWN, _KEY_LEFT):
                        idx = (idx - 1) % len(colors)
        except (termios.error, OSError):
            return default

    # ── Time range picker ──────────────────────────────────────────────

    def time_range_picker(self, message: str,
                          default_start: str = "", default_end: str = "") -> tuple[str, str]:
        """Pick a time range (start and end times).

        Args:
            message: prompt text
            default_start: default start time (HH:MM)
            default_end: default end time (HH:MM)

        Returns:
            tuple of (start_time, end_time) strings
        """
        if not self._is_tty:
            s = self._ask_fallback(f"{message} start", default_start or "09:00")
            e = self._ask_fallback(f"{message} end", default_end or "17:00")
            return (s.strip(), e.strip())
        return self._time_range_picker_raw(message, default_start, default_end)

    def _time_range_picker_raw(self, message: str, ds: str, de: str) -> tuple[str, str]:
        import datetime as _dt
        now = _dt.datetime.now()
        if not ds:
            ds = now.strftime("%H:%M")
        if not de:
            de = now.strftime("%H:%M")
        start = self.time_picker(f"{message} start time", default=ds)
        end = self.time_picker(f"{message} end time", default=de)
        return (start, end)

    # ── Number range picker ────────────────────────────────────────────

    def number_range_picker(self, message: str, min_val: int = 0,
                            max_val: int = 100, default: int = 0,
                            step: int = 1) -> int:
        """Pick a number from a range using arrow keys.

        Args:
            message: prompt text
            min_val: minimum value
            max_val: maximum value
            default: default value
            step: increment step

        Returns:
            selected number
        """
        if not self._is_tty:
            return int(self._ask_fallback(message, str(default)))
        return self._number_range_picker_raw(message, min_val, max_val, default, step)

    def _number_range_picker_raw(self, message: str, min_val: int, max_val: int,
                                 default: int, step: int) -> int:
        fd = self._get_fd()
        val = default

        def _render() -> str:
            pct = (val - min_val) / max(max_val - min_val, 1)
            bar_w = 20
            filled = int(pct * bar_w)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  "
                f"{_BOLD}{_CYAN}{val}{_RESET}\n"
                f"  {bar}  {_DIM}{min_val} - {max_val}{_RESET}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(2):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return val
                    if key.kind == _KEY_ENTER:
                        _clear(); return val
                    if key.kind in (_KEY_UP, _KEY_RIGHT):
                        val = min(max_val, val + step)
                    elif key.kind in (_KEY_DOWN, _KEY_LEFT):
                        val = max(min_val, val - step)
                    elif key.kind == _KEY_HOME:
                        val = min_val
                    elif key.kind == _KEY_END:
                        val = max_val
        except (termios.error, OSError):
            return default

    # ── Confirm with details ───────────────────────────────────────────

    def confirm_with_details(self, message: str, details: dict[str, str],
                             default: bool = False) -> bool:
        """Confirm with key-value details displayed.

        Args:
            message: confirmation prompt
            details: dict of label → value to display
            default: default answer

        Returns:
            True if confirmed
        """
        if not self._is_tty:
            return self._confirm_fallback(message, default)
        return self._confirm_with_details_raw(message, details, default)

    def _confirm_with_details_raw(self, message: str, details: dict[str, str],
                                  default: bool) -> bool:
        fd = self._get_fd()
        selected = not default

        def _render() -> str:
            if selected:
                yes_text = f"{_REVERSE}{_GREEN} Yes {_RESET_REVERSE}{_RESET}"
                no_text = " No "
            else:
                yes_text = " Yes "
                no_text = f"{_REVERSE}{_RED} No {_RESET_REVERSE}{_RESET}"
            detail_lines = "\n".join(
                f"  {_DIM}{k}:{_RESET} {v}" for k, v in details.items()
            )
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                f"{detail_lines}\n"
                f"  {yes_text}  {no_text}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            line_count = 2 + len(details)
            for _ in range(line_count):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return default
                    if key.kind == _KEY_ENTER:
                        _clear(); return selected
                    if key.kind in (_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear(); return True
                        if key.char in ("n", "N"):
                            _clear(); return False
        except (termios.error, OSError):
            return default

    # ── Spinner with status ────────────────────────────────────────────

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

    # ── Select with filter ─────────────────────────────────────────────

    def select_with_filter(self, message: str, options: list[str],
                           default: str = "") -> str:
        """Select from options with type-to-filter.

        Args:
            message: prompt text
            options: list of selectable options
            default: default selected option

        Returns:
            selected option string
        """
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_filter_raw(message, options, default)

    def _select_with_filter_raw(self, message: str, options: list[str],
                                default: str) -> str:
        fd = self._get_fd()
        idx = 0
        query = ""
        if default in options:
            idx = options.index(default)

        def _filtered() -> list[int]:
            if not query:
                return list(range(len(options)))
            return [i for i, o in enumerate(options)
                    if query.lower() in o.lower()]

        def _render() -> str:
            filtered = _filtered()
            if not filtered:
                vis = ["  (no matches)"]
            else:
                vis = []
                for fi in filtered[:10]:
                    prefix = ">> " if fi == idx else "   "
                    color = _CYAN if fi == idx else ""
                    reset = _RESET if fi == idx else ""
                    vis.append(f"  {prefix}{color}{options[fi]}{reset}")
            filter_str = f"  {_DIM}filter: {query}{_RESET}" if query else ""
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(type to filter, arrows){_RESET}\n"
                f"{filter_str}\n"
                + "\n".join(vis)
                + f"{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + min(len(_filtered()), 10)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return options[idx]
                    if key.kind == _KEY_ENTER:
                        _clear(); return options[idx]
                    if key.kind == _KEY_UP:
                        filtered = _filtered()
                        if filtered:
                            cur = filtered.index(idx) if idx in filtered else 0
                            idx = filtered[(cur - 1) % len(filtered)]
                    elif key.kind == _KEY_DOWN:
                        filtered = _filtered()
                        if filtered:
                            cur = filtered.index(idx) if idx in filtered else 0
                            idx = filtered[(cur + 1) % len(filtered)]
                    elif key.kind == _KEY_BACKSPACE:
                        if query:
                            query = query[:-1]
                            filtered = _filtered()
                            if filtered and idx not in filtered:
                                idx = filtered[0]
                    elif key.kind == _KEY_CHAR:
                        query += key.char
                        filtered = _filtered()
                        if filtered and idx not in filtered:
                            idx = filtered[0]
        except (termios.error, OSError):
            return options[idx] if idx < len(options) else default

    # ── Confirm with preview and edit ──────────────────────────────────

    def confirm_with_preview_and_edit(self, message: str, preview: str,
                                      edit_prompt: str = "Edit: ",
                                      default: bool = False) -> tuple[bool, str]:
        """Confirm with preview, with option to edit before confirming.

        Args:
            message: confirmation prompt
            preview: text preview of the action/result
            edit_prompt: prompt text for editing
            default: default answer

        Returns:
            tuple of (confirmed, edited_text)
        """
        if not self._is_tty:
            return (self._confirm_fallback(message, default), preview)
        return self._confirm_with_preview_and_edit_raw(message, preview, edit_prompt, default)

    def _confirm_with_preview_and_edit_raw(self, message: str, preview: str,
                                           edit_prompt: str, default: bool) -> tuple[bool, str]:
        fd = self._get_fd()
        selected = not default
        text = preview

        def _render() -> str:
            if selected:
                yes_text = f"{_REVERSE}{_GREEN} Yes {_RESET_REVERSE}{_RESET}"
                no_text = " No "
                edit_text = " Edit "
            else:
                yes_text = " Yes "
                no_text = f"{_REVERSE}{_RED} No {_RESET_REVERSE}{_RESET}"
                edit_text = " Edit "
            preview_lines = text.split("\n")[:5]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                f"{preview_str}\n"
                f"  {yes_text}  {no_text}  {edit_text}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + min(len(text.split("\n")), 5)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return (default, text)
                    if key.kind == _KEY_ENTER:
                        _clear(); return (selected, text)
                    if key.kind in (_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear(); return (True, text)
                        if key.char in ("n", "N"):
                            _clear(); return (False, text)
                        if key.char in ("e", "E"):
                            _clear()
                            edited = self._ask_fallback(edit_prompt, text)
                            return (True, edited)
        except (termios.error, OSError):
            return (default, text)

    # ── Progress bar colored ───────────────────────────────────────────

    def progress_bar_colored(self, label: str, current: int, total: int,
                             width: int = 30) -> None:
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

    # ── Spinner with progress ──────────────────────────────────────────

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

    # ── Select with icons ──────────────────────────────────────────────

    def select_with_icons(self, message: str, options: list[tuple[str, str]],
                          default: str = "") -> str:
        """Select from options with icons.

        Args:
            message: prompt text
            options: list of (icon, label) tuples
            default: default selected label

        Returns:
            selected label string
        """
        if not self._is_tty:
            labels = [o[1] for o in options]
            return self._select_fallback(message, labels)
        return self._select_with_icons_raw(message, options, default)

    def _select_with_icons_raw(self, message: str, options: list[tuple[str, str]],
                               default: str) -> str:
        fd = self._get_fd()
        labels = [o[1] for o in options]
        icons = [o[0] for o in options]
        idx = 0
        if default in labels:
            idx = labels.index(default)

        def _render() -> str:
            lines = []
            for i, (icon, label) in enumerate(zip(icons, labels)):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{color}{icon} {label}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows){_RESET}\n"
                + "\n".join(lines)
                + f"{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(2 + len(options)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return labels[idx]
                    if key.kind == _KEY_ENTER:
                        _clear(); return labels[idx]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % len(options)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % len(options)
        except (termios.error, OSError):
            return labels[idx]

    # ── Confirm with warning ───────────────────────────────────────────

    def confirm_with_warning(self, message: str, warning: str,
                             default: bool = False) -> bool:
        """Confirm with a warning message displayed.

        Args:
            message: confirmation prompt
            warning: warning text to display
            default: default answer

        Returns:
            True if confirmed
        """
        if not self._is_tty:
            return self._confirm_fallback(message, default)
        return self._confirm_with_warning_raw(message, warning, default)

    def _confirm_with_warning_raw(self, message: str, warning: str,
                                  default: bool) -> bool:
        fd = self._get_fd()
        selected = not default

        def _render() -> str:
            if selected:
                yes_text = f"{_REVERSE}{_GREEN} Yes {_RESET_REVERSE}{_RESET}"
                no_text = " No "
            else:
                yes_text = " Yes "
                no_text = f"{_REVERSE}{_RED} No {_RESET_REVERSE}{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                f"  {_YELLOW}\u26a0 {warning}{_RESET}\n"
                f"  {yes_text}  {no_text}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return default
                    if key.kind == _KEY_ENTER:
                        _clear(); return selected
                    if key.kind in (_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear(); return True
                        if key.char in ("n", "N"):
                            _clear(); return False
        except (termios.error, OSError):
            return default

    # ── Progress bar with ETA ──────────────────────────────────────────

    def progress_bar_eta(self, label: str, current: int, total: int,
                         elapsed: float, width: int = 30) -> None:
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
        self._io.write(f"  {_CYAN}{label}{_RESET} {_BOLD}{bar}{_RESET} {pct} {_DIM}ETA: {eta}{_RESET}")

    # ── Spinner with dots ──────────────────────────────────────────────

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
            sys.stdout.write(
                f"\r  {_CYAN}\u25f7{_RESET} {message}{dot}  {_HIDE_CURSOR}"
            )
            sys.stdout.flush()
            time.sleep(0.25)
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    # ── Select with pagination ─────────────────────────────────────────

    def select_with_pagination(self, message: str, options: list[str],
                               page_size: int = 10, default: str = "") -> str:
        """Select from options with page navigation for large lists.

        Args:
            message: prompt text
            options: list of selectable options
            page_size: number of items per page
            default: default selected option

        Returns:
            selected option string
        """
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_pagination_raw(message, options, page_size, default)

    def _select_with_pagination_raw(self, message: str, options: list[str],
                                    page_size: int, default: str) -> str:
        fd = self._get_fd()
        idx = 0
        page = 0
        if default in options:
            idx = options.index(default)
            page = idx // page_size

        def _total_pages() -> int:
            return (len(options) + page_size - 1) // page_size

        def _page_items() -> list[tuple[int, str]]:
            start = page * page_size
            end = min(start + page_size, len(options))
            return [(i, options[i]) for i in range(start, end)]

        def _render() -> str:
            items = _page_items()
            lines = []
            for i, label in items:
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{color}{label}{_RESET}")
            page_info = f"  {_DIM}page {page + 1}/{_total_pages()} (PgUp/PgDn){_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows, enter){_RESET}\n"
                + "\n".join(lines)
                + f"\n{page_info}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(2 + len(_page_items()) + 1):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return options[idx]
                    if key.kind == _KEY_ENTER:
                        _clear(); return options[idx]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % len(options)
                        page = idx // page_size
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % len(options)
                        page = idx // page_size
                    elif key.kind == _KEY_PAGE_UP:
                        page = max(0, page - 1)
                        idx = page * page_size
                    elif key.kind == _KEY_PAGE_DOWN:
                        page = min(_total_pages() - 1, page + 1)
                        idx = min(page * page_size, len(options) - 1)
        except (termios.error, OSError):
            return options[idx]

    # ── Select with search and preview ─────────────────────────────────

    def select_with_search_and_preview(self, message: str, options: list[str],
                                       preview_fn: Callable[[str], str]) -> str:
        """Select from options with type-to-filter and live preview.

        Args:
            message: prompt text
            options: list of selectable options
            preview_fn: function that returns preview text for an option

        Returns:
            selected option string
        """
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_search_and_preview_raw(message, options, preview_fn)

    def _select_with_search_and_preview_raw(self, message: str, options: list[str],
                                            preview_fn: Callable[[str], str]) -> str:
        fd = self._get_fd()
        query = ""
        idx = 0

        def _filtered() -> list[int]:
            if not query:
                return list(range(len(options)))
            return [i for i, o in enumerate(options)
                    if query.lower() in o.lower()]

        def _render() -> str:
            filtered = _filtered()
            preview_text = preview_fn(options[idx]) if filtered else ""
            preview_lines = preview_text.split("\n")[:8]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            if not filtered:
                vis = ["  (no matches)"]
            else:
                vis = []
                for fi in filtered[:8]:
                    prefix = ">> " if fi == idx else "   "
                    color = _CYAN if fi == idx else ""
                    reset = _RESET if fi == idx else ""
                    vis.append(f"  {prefix}{color}{options[fi]}{reset}")
            filter_str = f"  {_DIM}filter: {query}{_RESET}" if query else ""
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(type to filter, arrows){_RESET}\n"
                f"{filter_str}\n"
                + "\n".join(vis)
                + f"\n{preview_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            filtered = _filtered()
            preview_text = preview_fn(options[idx]) if filtered else ""
            for _ in range(3 + min(len(filtered), 8) + min(len(preview_text.split("\n")), 8)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return options[idx]
                    if key.kind == _KEY_ENTER:
                        _clear(); return options[idx]
                    if key.kind == _KEY_UP:
                        filtered = _filtered()
                        if filtered:
                            cur = filtered.index(idx) if idx in filtered else 0
                            idx = filtered[(cur - 1) % len(filtered)]
                    elif key.kind == _KEY_DOWN:
                        filtered = _filtered()
                        if filtered:
                            cur = filtered.index(idx) if idx in filtered else 0
                            idx = filtered[(cur + 1) % len(filtered)]
                    elif key.kind == _KEY_BACKSPACE:
                        if query:
                            query = query[:-1]
                            filtered = _filtered()
                            if filtered and idx not in filtered:
                                idx = filtered[0]
                    elif key.kind == _KEY_CHAR:
                        query += key.char
                        filtered = _filtered()
                        if filtered and idx not in filtered:
                            idx = filtered[0]
        except (termios.error, OSError):
            return options[idx] if idx < len(options) else options[0]

    # ── Progress bar with status ───────────────────────────────────────

    def progress_bar_with_status(self, label: str, current: int, total: int,
                                 status: str = "", width: int = 30) -> None:
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

    # ── Spinner with ETA ───────────────────────────────────────────────

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

    # ── Select with grouping ───────────────────────────────────────────

    def select_with_grouping(self, message: str,
                             groups: dict[str, list[str]],
                             default: str = "") -> str:
        """Select from categorized options with group headers.

        Args:
            message: prompt text
            groups: dict of group_name → list of options
            default: default selected option

        Returns:
            selected option string
        """
        if not self._is_tty:
            all_options = [o for opts in groups.values() for o in opts]
            return self._select_fallback(message, all_options)
        return self._select_with_grouping_raw(message, groups, default)

    def _select_with_grouping_raw(self, message: str,
                                  groups: dict[str, list[str]],
                                  default: str) -> str:
        fd = self._get_fd()
        flat: list[tuple[str | None, str]] = []
        for group_name, items in groups.items():
            flat.append((None, group_name))
            for item in items:
                flat.append((group_name, item))
        idx = 1
        for i, (_, item) in enumerate(flat):
            if item == default:
                idx = i
                break

        def _render() -> str:
            lines = []
            for i, (group, item) in enumerate(flat):
                if group is None:
                    lines.append(f"  {_BOLD}{_YELLOW}{item}{_RESET}")
                else:
                    prefix = ">> " if i == idx else "   "
                    color = _CYAN if i == idx else ""
                    reset = _RESET if i == idx else ""
                    lines.append(f"  {prefix}{color}{item}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows){_RESET}\n"
                + "\n".join(lines)
                + f"{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(2 + len(flat)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return flat[idx][1]
                    if key.kind == _KEY_ENTER:
                        _clear(); return flat[idx][1]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % len(flat)
                        while flat[idx][0] is None and idx > 0:
                            idx -= 1
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % len(flat)
                        while flat[idx][0] is None and idx < len(flat) - 1:
                            idx += 1
        except (termios.error, OSError):
            return flat[idx][1]

    # ── Multi-select with preview ──────────────────────────────────────

    def multi_select_with_preview(self, message: str, options: list[str],
                                  preview_fn: Callable[[str], str]) -> list[str]:
        """Multi-select from options with live preview.

        Args:
            message: prompt text
            options: list of selectable options
            preview_fn: function that returns preview text for an option

        Returns:
            list of selected option strings
        """
        if not self._is_tty:
            return self._select_multi_fallback(message, options)
        return self._multi_select_with_preview_raw(message, options, preview_fn)

    def _multi_select_with_preview_raw(self, message: str, options: list[str],
                                       preview_fn: Callable[[str], str]) -> list[str]:
        fd = self._get_fd()
        idx = 0
        selected: set[int] = set()

        def _render() -> str:
            preview_text = preview_fn(options[idx])
            preview_lines = preview_text.split("\n")[:6]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            lines = []
            for i, opt in enumerate(options):
                check = f"{_GREEN}\u2714{_RESET}" if i in selected else " "
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{check} {color}{opt}{_RESET}")
            count_str = f"  {_DIM}{len(selected)} selected{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(space=toggle, enter=confirm){_RESET}\n"
                + "\n".join(lines)
                + f"\n{count_str}\n{preview_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + len(options) + min(len(preview_fn(options[idx]).split("\n")), 6)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return [options[i] for i in sorted(selected)]
                    if key.kind == _KEY_ENTER:
                        _clear(); return [options[i] for i in sorted(selected)]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % len(options)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % len(options)
                    elif key.kind == _KEY_CHAR and key.char == " ":
                        if idx in selected:
                            selected.discard(idx)
                        else:
                            selected.add(idx)
        except (termios.error, OSError):
            return [options[i] for i in sorted(selected)]

    # ── Progress bar indeterminate ─────────────────────────────────────

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
        frames = ["\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588",
                   "\u2587", "\u2586", "\u2585", "\u2584", "\u2583", "\u2582"]
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

    # ── Table with search ──────────────────────────────────────────────

    def table_with_search(self, headers: list[str], rows: list[list[str]],
                          title: str = "") -> list[list[str]]:
        """Display a searchable table with type-to-filter.

        Args:
            headers: column headers
            rows: table data rows
            title: optional title

        Returns:
            filtered rows (all rows if no filter applied)
        """
        if not self._is_tty:
            return rows
        return self._table_with_search_raw(headers, rows, title)

    def _table_with_search_raw(self, headers: list[str], rows: list[list[str]],
                               title: str) -> list[list[str]]:
        fd = self._get_fd()
        query = ""
        col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
                      for i, h in enumerate(headers)]

        def _filtered() -> list[list[str]]:
            if not query:
                return rows
            q = query.lower()
            return [r for r in rows if any(q in str(c).lower() for c in r)]

        def _render() -> str:
            filtered = _filtered()
            lines = []
            header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
            lines.append(f"  {_BOLD}{header_line}{_RESET}")
            lines.append(f"  {_DIM}{'─' * sum(col_widths + [3 * (len(headers) - 1)])}{_RESET}")
            for r in filtered[:20]:
                line = "  ".join(str(r[i]).ljust(col_widths[i]) if i < len(r) else " " * col_widths[i]
                                 for i in range(len(headers)))
                lines.append(f"  {line}")
            filter_str = f"  {_DIM}filter: {query} ({len(filtered)} rows){_RESET}" if query else f"  {_DIM}{len(rows)} rows{_RESET}"
            title_str = f"  {_BOLD}{title}{_RESET}\n" if title else ""
            return (
                f"{_ERASE_LINE}\r{title_str}"
                + "\n".join(lines)
                + f"\n{filter_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            filtered = _filtered()
            line_count = 2 + min(len(filtered), 20) + 1
            if title:
                line_count += 1
            for _ in range(line_count):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return _filtered()
                    if key.kind == _KEY_ENTER:
                        _clear(); return _filtered()
                    elif key.kind == _KEY_BACKSPACE:
                        if query:
                            query = query[:-1]
                    elif key.kind == _KEY_CHAR:
                        query += key.char
        except (termios.error, OSError):
            return rows

    # ── Select with countdown ──────────────────────────────────────────

    def select_with_countdown(self, message: str, options: list[str],
                              timeout: int = 10, default: int = 0) -> str:
        """Select from options with auto-select countdown.

        Args:
            message: prompt text
            options: list of selectable options
            timeout: seconds before auto-selecting default
            default: index of default option to auto-select

        Returns:
            selected option string
        """
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_countdown_raw(message, options, timeout, default)

    def _select_with_countdown_raw(self, message: str, options: list[str],
                                   timeout: int, default: int) -> str:
        fd = self._get_fd()
        idx = default
        remaining = timeout

        def _render() -> str:
            lines = []
            for i, opt in enumerate(options):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{color}{opt}{_RESET}")
            countdown = f"  {_YELLOW}auto-select in {remaining}s{_RESET}" if remaining > 0 else f"  {_GREEN}selected!{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows, enter){_RESET}\n"
                + "\n".join(lines)
                + f"\n{countdown}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + len(options)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                start = time.time()
                while True:
                    elapsed = int(time.time() - start)
                    remaining = max(0, timeout - elapsed)
                    if remaining == 0:
                        _clear(); return options[idx]
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    remaining_time = 1.0 - (time.time() - start) % 1.0
                    key = _read_raw_key(fd, timeout=remaining_time)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return options[idx]
                    if key.kind == _KEY_ENTER:
                        _clear(); return options[idx]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % len(options)
                        start = time.time()
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % len(options)
                        start = time.time()
        except (termios.error, OSError):
            return options[idx]

    # ── Confirm with countdown ─────────────────────────────────────────

    def confirm_with_countdown(self, message: str, timeout: int = 10,
                               default: bool = False) -> bool:
        """Confirm with auto-confirm countdown.

        Args:
            message: confirmation prompt
            timeout: seconds before auto-confirming
            default: default answer

        Returns:
            True if confirmed
        """
        if not self._is_tty:
            return self._confirm_fallback(message, default)
        return self._confirm_with_countdown_raw(message, timeout, default)

    def _confirm_with_countdown_raw(self, message: str, timeout: int,
                                    default: bool) -> bool:
        fd = self._get_fd()
        selected = not default
        remaining = timeout

        def _render() -> str:
            if selected:
                yes_text = f"{_REVERSE}{_GREEN} Yes {_RESET_REVERSE}{_RESET}"
                no_text = " No "
            else:
                yes_text = " Yes "
                no_text = f"{_REVERSE}{_RED} No {_RESET_REVERSE}{_RESET}"
            countdown = f"  {_YELLOW}auto in {remaining}s{_RESET}" if remaining > 0 else ""
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                f"  {yes_text}  {no_text}{countdown}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                start = time.time()
                while True:
                    elapsed = int(time.time() - start)
                    remaining = max(0, timeout - elapsed)
                    if remaining == 0:
                        _clear(); return default
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    remaining_time = 1.0 - (time.time() - start) % 1.0
                    key = _read_raw_key(fd, timeout=remaining_time)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return default
                    if key.kind == _KEY_ENTER:
                        _clear(); return selected
                    if key.kind in (_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                        start = time.time()
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear(); return True
                        if key.char in ("n", "N"):
                            _clear(); return False
                        start = time.time()
        except (termios.error, OSError):
            return default

    # ── Progress bar striped ───────────────────────────────────────────

    def progress_bar_stripe(self, label: str, current: int, total: int,
                            width: int = 30) -> None:
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

    # ── Spinner with dots and ETA ──────────────────────────────────────

    def spinner_with_dots_eta(self, message: str, elapsed: float,
                              progress: float = 0) -> None:
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

    def _spinner_with_dots_eta_raw(self, message: str, elapsed: float,
                                   progress: float) -> None:
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

    # ── Confirm with phrase ────────────────────────────────────────────

    def confirm_with_phrase(self, message: str, phrase: str = "yes") -> bool:
        """Confirm by typing a specific phrase.

        Args:
            message: confirmation prompt
            phrase: required phrase to type

        Returns:
            True if phrase matched
        """
        if not self._is_tty:
            raw = self._ask_fallback(f"{message} (type '{phrase}')", "")
            return raw.strip().lower() == phrase.lower()
        return self._confirm_with_phrase_raw(message, phrase)

    def _confirm_with_phrase_raw(self, message: str, phrase: str) -> bool:
        fd = self._get_fd()
        buf: list[str] = []

        def _render() -> str:
            typed = "".join(buf)
            match = typed.lower() == phrase.lower()
            if match:
                status = f"{_GREEN}\u2714 matched{_RESET}"
            else:
                status = f"{_DIM}type '{phrase}' to confirm{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                f"  {_CYAN}{typed}{_RESET}{'█' if len(typed) < len(phrase) else ''}\n"
                f"  {status}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return False
                    if key.kind == _KEY_ENTER:
                        _clear(); return "".join(buf).lower() == phrase.lower()
                    if key.kind == _KEY_BACKSPACE:
                        if buf: buf.pop()
                    elif key.kind == _KEY_CHAR:
                        buf.append(key.char)
                        if "".join(buf).lower() == phrase.lower():
                            _clear(); return True
        except (termios.error, OSError):
            return False

    # ── Progress bar gradient ──────────────────────────────────────────

    def progress_bar_gradient(self, label: str, current: int, total: int,
                              width: int = 30) -> None:
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
        gradient_blocks = ["\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588"]
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

    # ── Spinner pulse ──────────────────────────────────────────────────

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
            sys.stdout.write(
                f"\r  {_CYAN}{dots[i % len(dots)]}{_RESET} {message}  {_HIDE_CURSOR}"
            )
            sys.stdout.flush()
            time.sleep(0.3)
            i += 1
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    # ── Select with preview and icons ──────────────────────────────────

    def select_with_preview_and_icons(self, message: str,
                                      options: list[tuple[str, str]],
                                      preview_fn: Callable[[str], str]) -> str:
        """Select from icon+label options with live preview.

        Args:
            message: prompt text
            options: list of (icon, label) tuples
            preview_fn: function that returns preview text for a label

        Returns:
            selected label string
        """
        if not self._is_tty:
            labels = [o[1] for o in options]
            return self._select_fallback(message, labels)
        return self._select_with_preview_and_icons_raw(message, options, preview_fn)

    def _select_with_preview_and_icons_raw(self, message: str,
                                           options: list[tuple[str, str]],
                                           preview_fn: Callable[[str], str]) -> str:
        fd = self._get_fd()
        labels = [o[1] for o in options]
        icons = [o[0] for o in options]
        idx = 0

        def _render() -> str:
            preview_text = preview_fn(labels[idx])
            preview_lines = preview_text.split("\n")[:6]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            lines = []
            for i, (icon, label) in enumerate(zip(icons, labels)):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{color}{icon} {label}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows){_RESET}\n"
                + "\n".join(lines)
                + f"\n{preview_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(2 + len(options) + min(len(preview_fn(labels[idx]).split("\n")), 6)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return labels[idx]
                    if key.kind == _KEY_ENTER:
                        _clear(); return labels[idx]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % len(options)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % len(options)
        except (termios.error, OSError):
            return labels[idx]

    # ── Multi confirm ──────────────────────────────────────────────────

    def multi_confirm(self, message: str, items: list[str],
                      default: bool = True) -> dict[str, bool]:
        """Confirm multiple items with toggle.

        Args:
            message: prompt text
            items: list of items to confirm
            default: default state for all items

        Returns:
            dict of item → confirmed status
        """
        if not self._is_tty:
            return {item: default for item in items}
        return self._multi_confirm_raw(message, items, default)

    def _multi_confirm_raw(self, message: str, items: list[str],
                           default: bool) -> dict[str, bool]:
        fd = self._get_fd()
        idx = 0
        states = {item: default for item in items}

        def _render() -> str:
            lines = []
            for i, item in enumerate(items):
                check = f"{_GREEN}\u2714{_RESET}" if states[item] else f"{_RED}\u2718{_RESET}"
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{check} {color}{item}{_RESET}")
            count = sum(1 for v in states.values() if v)
            count_str = f"  {_DIM}{count}/{len(items)} confirmed{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(space=toggle, enter=confirm){_RESET}\n"
                + "\n".join(lines)
                + f"\n{count_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + len(items)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return states
                    if key.kind == _KEY_ENTER:
                        _clear(); return states
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % len(items)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % len(items)
                    elif key.kind == _KEY_CHAR and key.char == " ":
                        states[items[idx]] = not states[items[idx]]
        except (termios.error, OSError):
            return states

    # ── Progress bar segmented ─────────────────────────────────────────

    def progress_bar_segmented(self, label: str, segments: list[tuple[str, int]],
                               width: int = 30) -> None:
        """Display a segmented progress bar with different colors per segment.

        Args:
            label: label text
            segments: list of (color_name, value) tuples
                      color_name: "green", "yellow", "red", "cyan"
            width: bar width in characters
        """
        total = sum(v for _, v in segments)
        colors = {
            "green": _GREEN, "yellow": _YELLOW, "red": _RED,
            "cyan": _CYAN, "magenta": "\033[35m",
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

    # ── Spinner wave ───────────────────────────────────────────────────

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
            sys.stdout.write(
                f"\r  {_CYAN}{wave_str}{_RESET} {message}  {_HIDE_CURSOR}"
            )
            sys.stdout.flush()
            time.sleep(0.15)
            i += 1
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    # ── Confirm list with preview ──────────────────────────────────────

    def confirm_list_with_preview(self, message: str, items: list[str],
                                  preview_fn: Callable[[str], str],
                                  default: bool = True) -> list[str]:
        """Confirm a list of items with preview, returning confirmed items."""
        if not self._is_tty:
            return items if default else []
        return self._confirm_list_with_preview_raw(message, items, preview_fn, default)

    def _confirm_list_with_preview_raw(self, message: str, items: list[str],
                                       preview_fn: Callable[[str], str],
                                       default: bool) -> list[str]:
        fd = self._get_fd()
        idx = 0
        states = {item: default for item in items}

        def _render() -> str:
            preview_text = preview_fn(items[idx])
            preview_lines = preview_text.split("\n")[:4]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            lines = []
            for i, item in enumerate(items):
                check = f"{_GREEN}\u2714{_RESET}" if states[item] else f"{_RED}\u2718{_RESET}"
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{check} {color}{item}{_RESET}")
            count = sum(1 for v in states.values() if v)
            count_str = f"  {_DIM}{count}/{len(items)} confirmed{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(space=toggle, enter=confirm){_RESET}\n"
                + "\n".join(lines)
                + f"\n{count_str}\n{preview_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(4 + len(items) + 4):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return [item for item, v in states.items() if v]
                    if key.kind == _KEY_ENTER:
                        _clear(); return [item for item, v in states.items() if v]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % len(items)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % len(items)
                    elif key.kind == _KEY_CHAR and key.char == " ":
                        states[items[idx]] = not states[items[idx]]
        except (termios.error, OSError):
            return [item for item, v in states.items() if v]

    # ── Progress bar multi segment ─────────────────────────────────────

    def progress_bar_multi_segment(self, label: str,
                                   segments: list[tuple[str, int, str]],
                                   width: int = 30) -> None:
        """Display a multi-segment progress bar with labels."""
        total = sum(v for _, v, _ in segments)
        colors_map = {
            "green": _GREEN, "yellow": _YELLOW, "red": _RED,
            "cyan": _CYAN, "magenta": "\033[35m", "blue": "\033[34m",
        }
        bar = ""
        for name, val, color in segments:
            frac = val / max(total, 1)
            filled = int(frac * width)
            c = colors_map.get(color, _CYAN)
            bar += f"{c}{'█' * filled}"
        remaining = width - sum(int(v / max(total, 1) * width) for _, v, _ in segments)
        bar += f"{_DIM}{'░' * max(remaining, 0)}{_RESET}"
        labels = " ".join(f"{colors_map.get(c, _CYAN)}{n}{_RESET}: {v}" for n, v, c in segments)
        self._io.write(f"  {_CYAN}{label}{_RESET} {bar}\n  {_DIM}{labels}{_RESET}")

    # ── Spinner bounce ─────────────────────────────────────────────────

    def spinner_bounce(self, message: str, duration: float = 2.0) -> None:
        """Display a bouncing animation spinner."""
        if not self._is_tty:
            self._io.write(f"  {message}...")
            return
        self._spinner_bounce_raw(message, duration)

    def _spinner_bounce_raw(self, message: str, duration: float) -> None:
        frames = ["\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588",
                   "\u2587", "\u2586", "\u2585", "\u2584", "\u2583", "\u2582"]
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

    # ── Select with confirm ────────────────────────────────────────────

    def select_with_confirm(self, message: str, options: list[str],
                            default: str = "") -> str:
        """Select an option and confirm with y/n."""
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_confirm_raw(message, options, default)

    def _select_with_confirm_raw(self, message: str, options: list[str],
                                 default: str) -> str:
        fd = self._get_fd()
        idx = 0
        if default in options:
            idx = options.index(default)

        def _render() -> str:
            lines = []
            for i, opt in enumerate(options):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{color}{opt}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows, enter to confirm){_RESET}\n"
                + "\n".join(lines)
                + f"{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(2 + len(options)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return options[idx]
                    if key.kind == _KEY_ENTER:
                        _clear(); return options[idx]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % len(options)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % len(options)
        except (termios.error, OSError):
            return options[idx]

    # ── Confirm with preview and timeout ───────────────────────────────

    def confirm_with_preview_and_timeout(self, message: str, preview: str,
                                         timeout: int = 10,
                                         default: bool = False) -> bool:
        """Confirm with preview and auto-confirm countdown."""
        if not self._is_tty:
            return self._confirm_fallback(message, default)
        return self._confirm_with_preview_and_timeout_raw(message, preview, timeout, default)

    def _confirm_with_preview_and_timeout_raw(self, message: str, preview: str,
                                              timeout: int, default: bool) -> bool:
        fd = self._get_fd()
        selected = not default
        remaining = timeout

        def _render() -> str:
            if selected:
                yes_text = f"{_REVERSE}{_GREEN} Yes {_RESET_REVERSE}{_RESET}"
                no_text = " No "
            else:
                yes_text = " Yes "
                no_text = f"{_REVERSE}{_RED} No {_RESET_REVERSE}{_RESET}"
            preview_lines = preview.split("\n")[:5]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            countdown = f"  {_YELLOW}auto in {remaining}s{_RESET}" if remaining > 0 else ""
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                f"{preview_str}\n"
                f"  {yes_text}  {no_text}{countdown}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + min(len(preview.split("\n")), 5)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                start = time.time()
                while True:
                    elapsed = int(time.time() - start)
                    remaining = max(0, timeout - elapsed)
                    if remaining == 0:
                        _clear(); return default
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    remaining_time = 1.0 - (time.time() - start) % 1.0
                    key = _read_raw_key(fd, timeout=remaining_time)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return default
                    if key.kind == _KEY_ENTER:
                        _clear(); return selected
                    if key.kind in (_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                        start = time.time()
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear(); return True
                        if key.char in ("n", "N"):
                            _clear(); return False
                        start = time.time()
        except (termios.error, OSError):
            return default

    # ── Progress bar animated ──────────────────────────────────────────

    def progress_bar_animated(self, label: str, current: int, total: int,
                              width: int = 30) -> None:
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

    # ── Spinner clock ──────────────────────────────────────────────────

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

    # ── Select with preview and confirm ────────────────────────────────

    def select_with_preview_and_confirm(self, message: str, options: list[str],
                                        preview_fn: Callable[[str], str],
                                        default: str = "") -> str:
        """Select with preview and confirm."""
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_preview_and_confirm_raw(message, options, preview_fn, default)

    def _select_with_preview_and_confirm_raw(self, message: str, options: list[str],
                                             preview_fn: Callable[[str], str],
                                             default: str) -> str:
        fd = self._get_fd()
        idx = 0
        if default in options:
            idx = options.index(default)

        def _render() -> str:
            preview_text = preview_fn(options[idx])
            preview_lines = preview_text.split("\n")[:5]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            lines = []
            for i, opt in enumerate(options):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{color}{opt}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows, enter=confirm){_RESET}\n"
                + "\n".join(lines)
                + f"\n{preview_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(2 + len(options) + 6):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return options[idx]
                    if key.kind == _KEY_ENTER:
                        _clear(); return options[idx]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % len(options)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % len(options)
        except (termios.error, OSError):
            return options[idx]

    # ── Confirm with preview and countdown ─────────────────────────────

    def confirm_with_preview_and_countdown(self, message: str, preview: str,
                                           timeout: int = 10,
                                           default: bool = True) -> bool:
        """Confirm with preview and countdown, auto-confirming on timeout."""
        if not self._is_tty:
            return self._confirm_fallback(message, default)
        return self._confirm_with_preview_and_countdown_raw(message, preview, timeout, default)

    def _confirm_with_preview_and_countdown_raw(self, message: str, preview: str,
                                                timeout: int, default: bool) -> bool:
        fd = self._get_fd()
        selected = not default

        def _render() -> str:
            if selected:
                yes_text = f"{_REVERSE}{_GREEN} Yes {_RESET_REVERSE}{_RESET}"
                no_text = " No "
            else:
                yes_text = " Yes "
                no_text = f"{_REVERSE}{_RED} No {_RESET_REVERSE}{_RESET}"
            preview_lines = preview.split("\n")[:4]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            remaining = timeout
            countdown = f"  {_YELLOW}auto in {remaining}s{_RESET}" if remaining > 0 else ""
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                f"{preview_str}\n"
                f"  {yes_text}  {no_text}{countdown}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + min(len(preview.split("\n")), 4)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                start = time.time()
                while True:
                    elapsed = int(time.time() - start)
                    remaining = max(0, timeout - elapsed)
                    if remaining == 0:
                        _clear(); return default
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    remaining_time = 1.0 - (time.time() - start) % 1.0
                    key = _read_raw_key(fd, timeout=remaining_time)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return default
                    if key.kind == _KEY_ENTER:
                        _clear(); return selected
                    if key.kind in (_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                        start = time.time()
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear(); return True
                        if key.char in ("n", "N"):
                            _clear(); return False
                        start = time.time()
        except (termios.error, OSError):
            return default

    # ── Progress bar with status and ETA ───────────────────────────────

    def progress_bar_with_status_and_eta(self, label: str, current: int,
                                         total: int, status: str,
                                         width: int = 30,
                                         elapsed: float = 0.0) -> None:
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
        self._io.write(
            f"  {_CYAN}{label}{_RESET} {bar} {pct}{status_display}{eta_display}"
        )

    # ── Spinner with messages ──────────────────────────────────────────

    def spinner_with_messages(self, message: str,
                              messages: list[str],
                              duration: float = 3.0) -> None:
        """Display a spinner that cycles through different messages."""
        if not self._is_tty:
            self._io.write(f"  {message}...")
            return
        self._spinner_with_messages_raw(message, messages, duration)

    def _spinner_with_messages_raw(self, message: str, messages: list[str],
                                   duration: float) -> None:
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

    # ── Select table with preview ──────────────────────────────────────

    def select_table_with_preview(self, headers: list[str],
                                  rows: list[list[str]],
                                  preview_fn: Callable[[list[str]], str]) -> list[str]:
        """Select a row from a table with live preview."""
        if not self._is_tty:
            return rows[0] if rows else []
        return self._select_table_with_preview_raw(headers, rows, preview_fn)

    def _select_table_with_preview_raw(self, headers: list[str],
                                       rows: list[list[str]],
                                       preview_fn: Callable[[list[str]], str]) -> list[str]:
        fd = self._get_fd()
        idx = 0

        def _render() -> str:
            col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
                         for i, h in enumerate(headers)]
            preview_text = preview_fn(rows[idx]) if rows else ""
            preview_lines = preview_text.split("\n")[:5]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            header = "  " + "  ".join(f"{_BOLD}{h:<{col_widths[i]}}{_RESET}"
                                      for i, h in enumerate(headers))
            lines = [header]
            for i, row in enumerate(rows):
                prefix = ">>" if i == idx else "  "
                cells = "  ".join(f"{c:<{col_widths[j]}}" for j, c in enumerate(row))
                if i == idx:
                    lines.append(f"  {_CYAN}{prefix} {cells}{_RESET}")
                else:
                    lines.append(f"  {prefix} {cells}")
            return (
                f"{_ERASE_LINE}\r  {_BOLD}Table{_RESET}  {_DIM}(arrows, enter=select){_RESET}\n"
                + "\n".join(lines)
                + f"\n{preview_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + len(rows) + 6):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return rows[idx]
                    if key.kind == _KEY_ENTER:
                        _clear(); return rows[idx]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % len(rows)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % len(rows)
        except (termios.error, OSError):
            return rows[idx] if rows else []

    # ── Confirm with preview, edit, and timeout ────────────────────────

    def confirm_with_preview_and_edit_with_timeout(self, message: str,
                                                   preview: str,
                                                   edit_prompt: str = "Edit:",
                                                   timeout: int = 10,
                                                   default: bool = True) -> tuple[bool, str]:
        """Confirm with preview, optional edit, and timeout."""
        if not self._is_tty:
            return (self._confirm_fallback(message, default), preview)
        return self._confirm_with_preview_and_edit_with_timeout_raw(
            message, preview, edit_prompt, timeout, default)

    def _confirm_with_preview_and_edit_with_timeout_raw(self, message: str,
                                                       preview: str,
                                                       edit_prompt: str,
                                                       timeout: int,
                                                       default: bool) -> tuple[bool, str]:
        fd = self._get_fd()
        selected = not default
        text = preview

        def _render() -> str:
            if selected:
                yes_text = f"{_REVERSE}{_GREEN} Yes {_RESET_REVERSE}{_RESET}"
                no_text = " No "
                edit_text = " Edit "
            else:
                yes_text = " Yes "
                no_text = f"{_REVERSE}{_RED} No {_RESET_REVERSE}{_RESET}"
                edit_text = " Edit "
            preview_lines = text.split("\n")[:4]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            remaining = timeout
            countdown = f"  {_YELLOW}auto in {remaining}s{_RESET}" if remaining > 0 else ""
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                f"{preview_str}\n"
                f"  {yes_text}  {no_text}  {edit_text}{countdown}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + min(len(preview.split("\n")), 4) + 2):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                start = time.time()
                while True:
                    elapsed = int(time.time() - start)
                    remaining = max(0, timeout - elapsed)
                    if remaining == 0:
                        _clear(); return (default, text)
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    remaining_time = 1.0 - (time.time() - start) % 1.0
                    key = _read_raw_key(fd, timeout=remaining_time)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return (default, text)
                    if key.kind == _KEY_ENTER:
                        _clear(); return (selected, text)
                    if key.kind == _KEY_LEFT:
                        selected = True; start = time.time()
                    elif key.kind == _KEY_RIGHT:
                        selected = False; start = time.time()
                    elif key.kind == _KEY_TAB:
                        selected = not selected; start = time.time()
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear(); return (True, text)
                        if key.char in ("n", "N"):
                            _clear(); return (False, text)
                        if key.char in ("e", "E"):
                            _clear()
                            edited = self._ask_fallback(edit_prompt, text)
                            return (True, edited)
                        start = time.time()
        except (termios.error, OSError):
            return (default, text)

    # ── Progress bar with ETA and status ───────────────────────────────

    def progress_bar_with_eta_and_status(self, label: str, current: int,
                                         total: int, status: str,
                                         elapsed: float = 0.0,
                                         width: int = 30) -> None:
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
        self._io.write(
            f"  {_CYAN}{label}{_RESET} {bar} {pct}{status_display}{eta_display}"
        )

    # ── Spinner with dots and status ───────────────────────────────────

    def spinner_with_dots_and_status(self, message: str, status: str,
                                     duration: float = 2.0) -> None:
        """Display a spinner with dots and status text."""
        if not self._is_tty:
            self._io.write(f"  {message}...")
            return
        self._spinner_with_dots_and_status_raw(message, status, duration)

    def _spinner_with_dots_and_status_raw(self, message: str, status: str,
                                          duration: float) -> None:
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

    # ── Select with preview and countdown ──────────────────────────────

    def select_with_preview_and_countdown(self, message: str,
                                          options: list[str],
                                          preview_fn: Callable[[str], str],
                                          timeout: int = 10,
                                          default: str = "") -> str:
        """Select with preview and auto-confirm countdown."""
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_preview_and_countdown_raw(
            message, options, preview_fn, timeout, default)

    def _select_with_preview_and_countdown_raw(self, message: str,
                                               options: list[str],
                                               preview_fn: Callable[[str], str],
                                               timeout: int,
                                               default: str) -> str:
        fd = self._get_fd()
        idx = 0
        if default in options:
            idx = options.index(default)

        def _render() -> str:
            preview_text = preview_fn(options[idx])
            preview_lines = preview_text.split("\n")[:5]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            lines = []
            for i, opt in enumerate(options):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{color}{opt}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows, enter=select){_RESET}\n"
                + "\n".join(lines)
                + f"\n{preview_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(2 + len(options) + 6):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                start = time.time()
                while True:
                    elapsed = int(time.time() - start)
                    remaining = max(0, timeout - elapsed)
                    if remaining == 0:
                        _clear(); return options[idx]
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    remaining_time = 1.0 - (time.time() - start) % 1.0
                    key = _read_raw_key(fd, timeout=remaining_time)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return options[idx]
                    if key.kind == _KEY_ENTER:
                        _clear(); return options[idx]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % len(options)
                        start = time.time()
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % len(options)
                        start = time.time()
        except (termios.error, OSError):
            return options[idx]

    # ── Multi-select with filter ───────────────────────────────────────

    def multi_select_with_filter(self, message: str, options: list[str],
                                 default: list[str] | None = None) -> list[str]:
        """Multi-select with type-to-filter."""
        if not self._is_tty:
            return (default or [])[:1]
        return self._multi_select_with_filter_raw(message, options, default or [])

    def _multi_select_with_filter_raw(self, message: str, options: list[str],
                                      default: list[str]) -> list[str]:
        fd = self._get_fd()
        query = ""
        idx = 0
        selected = set(default)

        def _filtered() -> list[str]:
            if not query:
                return options
            q = query.lower()
            return [o for o in options if q in o.lower()]

        def _render() -> str:
            flt = _filtered()
            lines = []
            for i, opt in enumerate(flt):
                check = f"{_GREEN}\u2714{_RESET}" if opt in selected else f"{_RED}\u2718{_RESET}"
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{check} {color}{opt}{_RESET}")
            if not lines:
                lines = [f"  {_DIM}(no matches){_RESET}"]
            count = f"  {_DIM}{len(selected)}/{len(options)} selected{_RESET}"
            query_display = f"  {_BOLD}Filter:{_RESET} {query}{_CYAN}\u2502{_RESET}" if query else f"  {_DIM}Type to filter{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows=move, space=toggle, enter=confirm){_RESET}\n"
                + query_display
                + "\n" + "\n".join(lines)
                + f"\n{count}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + len(options) + 2):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    flt = _filtered()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return sorted(selected)
                    if key.kind == _KEY_ENTER:
                        _clear(); return sorted(selected)
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % max(len(flt), 1)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % max(len(flt), 1)
                    elif key.kind == _KEY_CHAR and key.char == " ":
                        if flt[idx] in selected:
                            selected.discard(flt[idx])
                        else:
                            selected.add(flt[idx])
                    elif key.kind == _KEY_CHAR:
                        if key.char == "\x7f":
                            query = query[:-1]
                        else:
                            query += key.char
                        idx = 0
                    elif key.kind == _KEY_BACKSPACE:
                        query = query[:-1]
                        idx = 0
        except (termios.error, OSError):
            return sorted(selected)

    # ── Confirm with countdown and preview ─────────────────────────────

    def confirm_with_countdown_and_preview(self, message: str,
                                           preview: str,
                                           timeout: int = 10,
                                           default: bool = True) -> bool:
        """Confirm with preview and countdown, auto-confirming on timeout."""
        if not self._is_tty:
            return self._confirm_fallback(message, default)
        return self._confirm_with_countdown_and_preview_raw(message, preview, timeout, default)

    def _confirm_with_countdown_and_preview_raw(self, message: str,
                                                preview: str,
                                                timeout: int,
                                                default: bool) -> bool:
        fd = self._get_fd()
        selected = not default

        def _render() -> str:
            if selected:
                yes_text = f"{_REVERSE}{_GREEN} Yes {_RESET_REVERSE}{_RESET}"
                no_text = " No "
            else:
                yes_text = " Yes "
                no_text = f"{_REVERSE}{_RED} No {_RESET_REVERSE}{_RESET}"
            preview_lines = preview.split("\n")[:4]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            remaining = timeout
            countdown = f"  {_YELLOW}auto in {remaining}s{_RESET}" if remaining > 0 else ""
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                f"{preview_str}\n"
                f"  {yes_text}  {no_text}{countdown}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + min(len(preview.split("\n")), 4)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                start = time.time()
                while True:
                    elapsed = int(time.time() - start)
                    remaining = max(0, timeout - elapsed)
                    if remaining == 0:
                        _clear(); return default
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    remaining_time = 1.0 - (time.time() - start) % 1.0
                    key = _read_raw_key(fd, timeout=remaining_time)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return default
                    if key.kind == _KEY_ENTER:
                        _clear(); return selected
                    if key.kind in (_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                        start = time.time()
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear(); return True
                        if key.char in ("n", "N"):
                            _clear(); return False
                        start = time.time()
        except (termios.error, OSError):
            return default

    # ── Progress bar with steps ────────────────────────────────────────

    def progress_bar_with_steps(self, label: str, steps: list[str],
                                current_step: int, width: int = 30) -> None:
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

    # ── Spinner with ETA message ───────────────────────────────────────

    def spinner_with_eta_message(self, message: str, total: int,
                                 duration: float = 3.0) -> None:
        """Display a spinner with ETA message."""
        if not self._is_tty:
            self._io.write(f"  {message}...")
            return
        self._spinner_with_eta_message_raw(message, total, duration)

    def _spinner_with_eta_message_raw(self, message: str, total: int,
                                      duration: float) -> None:
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

    # ── Table with search and preview ──────────────────────────────────

    def table_with_search_and_preview(self, headers: list[str],
                                      rows: list[list[str]],
                                      preview_fn: Callable[[list[str]], str]) -> list[str]:
        """Searchable table with live preview."""
        if not self._is_tty:
            return rows[0] if rows else []
        return self._table_with_search_and_preview_raw(headers, rows, preview_fn)

    def _table_with_search_and_preview_raw(self, headers: list[str],
                                           rows: list[list[str]],
                                           preview_fn: Callable[[list[str]], str]) -> list[str]:
        fd = self._get_fd()
        query = ""
        idx = 0

        def _filtered() -> list[list[str]]:
            if not query:
                return rows
            q = query.lower()
            return [r for r in rows if any(q in c.lower() for c in r)]

        def _render() -> str:
            flt = _filtered()
            col_widths = [max(len(h), max((len(r[i]) for r in flt), default=0))
                         for i, h in enumerate(headers)]
            preview_text = preview_fn(flt[idx]) if flt else ""
            preview_lines = preview_text.split("\n")[:4]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            header = "  " + "  ".join(f"{_BOLD}{h:<{col_widths[i]}}{_RESET}"
                                      for i, h in enumerate(headers))
            lines = [header]
            for i, row in enumerate(flt):
                prefix = ">>" if i == idx else "  "
                cells = "  ".join(f"{c:<{col_widths[j]}}" for j, c in enumerate(row))
                if i == idx:
                    lines.append(f"  {_CYAN}{prefix} {cells}{_RESET}")
                else:
                    lines.append(f"  {prefix} {cells}")
            if not flt:
                lines.append(f"  {_DIM}(no matches){_RESET}")
            query_display = f"  {_BOLD}Search:{_RESET} {query}{_CYAN}\u2502{_RESET}" if query else f"  {_DIM}Type to search{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}Table{_RESET}  {_DIM}(arrows, type=search){_RESET}\n"
                + query_display
                + "\n" + "\n".join(lines)
                + f"\n{preview_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + len(rows) + 4):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    flt = _filtered()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return flt[idx] if flt else rows[0]
                    if key.kind == _KEY_ENTER:
                        _clear(); return flt[idx] if flt else rows[0]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % max(len(flt), 1)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % max(len(flt), 1)
                    elif key.kind == _KEY_CHAR:
                        if key.char == "\x7f":
                            query = query[:-1]
                        else:
                            query += key.char
                        idx = 0
                    elif key.kind == _KEY_BACKSPACE:
                        query = query[:-1]
                        idx = 0
        except (termios.error, OSError):
            flt = _filtered()
            return flt[0] if flt else rows[0]

    # ── Select with filter and confirm ─────────────────────────────────

    def select_with_filter_and_confirm(self, message: str,
                                       options: list[str],
                                       default: str = "") -> str:
        """Select with type-to-filter and confirm."""
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_filter_and_confirm_raw(message, options, default)

    def _select_with_filter_and_confirm_raw(self, message: str,
                                            options: list[str],
                                            default: str) -> str:
        fd = self._get_fd()
        query = ""
        idx = 0
        if default in options:
            idx = options.index(default)

        def _filtered() -> list[str]:
            if not query:
                return options
            q = query.lower()
            return [o for o in options if q in o.lower()]

        def _render() -> str:
            flt = _filtered()
            lines = []
            for i, opt in enumerate(flt):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{color}{opt}{_RESET}")
            if not lines:
                lines = [f"  {_DIM}(no matches){_RESET}"]
            query_display = f"  {_BOLD}Filter:{_RESET} {query}{_CYAN}\u2502{_RESET}" if query else f"  {_DIM}Type to filter{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows, type=filter, enter=confirm){_RESET}\n"
                + query_display
                + "\n" + "\n".join(lines)
                + f"{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(2 + len(options)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    flt = _filtered()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return flt[idx] if flt else options[0]
                    if key.kind == _KEY_ENTER:
                        _clear(); return flt[idx] if flt else options[0]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % max(len(flt), 1)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % max(len(flt), 1)
                    elif key.kind == _KEY_CHAR:
                        if key.char == "\x7f":
                            query = query[:-1]
                        else:
                            query += key.char
                        idx = 0
                    elif key.kind == _KEY_BACKSPACE:
                        query = query[:-1]
                        idx = 0
        except (termios.error, OSError):
            flt = _filtered()
            return flt[0] if flt else options[0]

    # ── Select with filter and preview ─────────────────────────────────

    def select_with_filter_and_preview(self, message: str,
                                       options: list[str],
                                       preview_fn: Callable[[str], str]) -> str:
        """Select from options with type-to-filter and live preview."""
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_filter_and_preview_raw(message, options, preview_fn)

    def _select_with_filter_and_preview_raw(self, message: str,
                                            options: list[str],
                                            preview_fn: Callable[[str], str]) -> str:
        fd = self._get_fd()
        query = ""
        idx = 0

        def _filtered() -> list[str]:
            if not query:
                return options
            q = query.lower()
            return [o for o in options if q in o.lower()]

        def _render() -> str:
            flt = _filtered()
            preview_text = preview_fn(flt[idx]) if flt else ""
            preview_lines = preview_text.split("\n")[:5]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            lines = []
            for i, opt in enumerate(flt):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{color}{opt}{_RESET}")
            if not lines:
                lines = [f"  {_DIM}(no matches){_RESET}"]
            query_display = f"  {_BOLD}Filter:{_RESET} {query}{_CYAN}\u2502{_RESET}" if query else f"  {_DIM}Type to filter{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows, type=filter){_RESET}\n"
                + query_display
                + "\n" + "\n".join(lines)
                + f"\n{preview_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + len(options) + 6):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    flt = _filtered()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return flt[idx] if flt else options[0]
                    if key.kind == _KEY_ENTER:
                        _clear(); return flt[idx] if flt else options[0]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % max(len(flt), 1)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % max(len(flt), 1)
                    elif key.kind == _KEY_CHAR:
                        if key.char == "\x7f":
                            query = query[:-1]
                        else:
                            query += key.char
                        idx = 0
                    elif key.kind == _KEY_BACKSPACE:
                        query = query[:-1]
                        idx = 0
        except (termios.error, OSError):
            flt = _filtered()
            return flt[0] if flt else options[0]

    # ── Select with tags ───────────────────────────────────────────────

    def select_with_tags(self, message: str, options: list[str],
                         tags: dict[str, list[str]]) -> str:
        """Select from options filtered by tags.

        Args:
            message: prompt text
            options: list of selectable options
            tags: dict of option → list of tags

        Returns:
            selected option string
        """
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_tags_raw(message, options, tags)

    def _select_with_tags_raw(self, message: str, options: list[str],
                              tags: dict[str, list[str]]) -> str:
        fd = self._get_fd()
        idx = 0
        active_tag: str | None = None

        def _filtered() -> list[str]:
            if not active_tag:
                return options
            return [o for o in options if active_tag in tags.get(o, [])]

        def _all_tags() -> list[str]:
            seen: set[str] = set()
            for tag_list in tags.values():
                for t in tag_list:
                    seen.add(t)
            return sorted(seen)

        def _render() -> str:
            filtered = _filtered()
            all_tags = _all_tags()
            tag_line = " ".join(
                f"{_GREEN}{t}{_RESET}" if t == active_tag else f"{_DIM}{t}{_RESET}"
                for t in all_tags
            )
            if not filtered:
                vis = ["  (no matches)"]
            else:
                vis = []
                for fi, opt in enumerate(filtered[:10]):
                    prefix = ">> " if fi == idx % len(filtered) else "   "
                    color = _CYAN if fi == idx % len(filtered) else ""
                    reset = _RESET if fi == idx % len(filtered) else ""
                    opt_tags = tags.get(opt, [])
                    tag_str = f"  {_DIM}[{', '.join(opt_tags)}]{_RESET}" if opt_tags else ""
                    vis.append(f"  {prefix}{color}{opt}{reset}{tag_str}")
            tag_str = f"  {_DIM}tags: {tag_line}{_RESET}" if all_tags else ""
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows, tab=next tag){_RESET}\n"
                f"{tag_str}\n"
                + "\n".join(vis)
                + f"{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(3 + min(len(_filtered()), 10)):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    filtered = _filtered()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return filtered[idx % len(filtered)] if filtered else options[0]
                    if key.kind == _KEY_ENTER:
                        _clear(); return filtered[idx % len(filtered)] if filtered else options[0]
                    if key.kind == _KEY_UP:
                        idx = (idx - 1) % max(len(filtered), 1)
                    elif key.kind == _KEY_DOWN:
                        idx = (idx + 1) % max(len(filtered), 1)
                    elif key.kind == _KEY_TAB:
                        all_tags = _all_tags()
                        if all_tags:
                            if active_tag is None:
                                active_tag = all_tags[0]
                            else:
                                ci = all_tags.index(active_tag)
                                active_tag = all_tags[(ci + 1) % len(all_tags)]
                            idx = 0
        except (termios.error, OSError):
            filtered = _filtered()
            return filtered[0] if filtered else options[0]

    # ── Select with preview and grouping ───────────────────────────────

    def select_with_preview_and_grouping(self, message: str,
                                         groups: dict[str, list[str]],
                                         preview_fn: Callable[[str], str]) -> str:
        """Select from categorized options with group headers and live preview."""
        if not self._is_tty:
            all_options = [o for opts in groups.values() for o in opts]
            return self._select_fallback(message, all_options)
        return self._select_with_preview_and_grouping_raw(message, groups, preview_fn)

    def _select_with_preview_and_grouping_raw(self, message: str,
                                              groups: dict[str, list[str]],
                                              preview_fn: Callable[[str], str]) -> str:
        fd = self._get_fd()
        flat: list[tuple[str | None, str]] = []
        for group_name, items in groups.items():
            flat.append((None, group_name))
            for item in items:
                flat.append((group_name, item))
        idx = 1
        selectable = [i for i, (g, _) in enumerate(flat) if g is not None]

        def _render() -> str:
            preview_text = preview_fn(flat[idx][1]) if flat[idx][0] is not None else ""
            preview_lines = preview_text.split("\n")[:6]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            lines = []
            for i, (group, item) in enumerate(flat):
                if group is None:
                    lines.append(f"  {_BOLD}{_YELLOW}{item}{_RESET}")
                else:
                    prefix = ">> " if i == idx else "   "
                    color = _CYAN if i == idx else ""
                    reset = _RESET if i == idx else ""
                    lines.append(f"  {prefix}{color}{item}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows){_RESET}\n"
                + "\n".join(lines)
                + f"\n{preview_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            for _ in range(2 + len(flat) + 6):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR); sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(); return flat[idx][1]
                    if key.kind == _KEY_ENTER:
                        _clear(); return flat[idx][1]
                    if key.kind == _KEY_UP:
                        pos = selectable.index(idx) if idx in selectable else 0
                        idx = selectable[(pos - 1) % len(selectable)]
                    elif key.kind == _KEY_DOWN:
                        pos = selectable.index(idx) if idx in selectable else 0
                        idx = selectable[(pos + 1) % len(selectable)]
        except (termios.error, OSError):
            return flat[idx][1]

    # ── Table sort ─────────────────────────────────────────────────────

    def table_sort(self, headers: list[str], rows: list[list[str]],
                   title: str = "Sort table") -> list[list[str]]:
        """Interactive table with column sorting via arrow keys.

        Args:
            headers: column headers
            rows: table data rows
            title: prompt text

        Returns:
            sorted rows
        """
        if not rows:
            return []
        if not self._is_tty:
            return rows
        return self._table_sort_raw(headers, rows, title)

    def _table_sort_raw(self, headers: list[str], rows: list[list[str]],
                        title: str) -> list[list[str]]:
        import locale as _loc
        fd = self._get_fd()
        sort_col = 0
        sort_asc = True
        sorted_rows = list(rows)
        cursor = 0
        scroll = 0
        max_visible = min(len(rows), 12)
        widths = [len(h) for h in headers]
        for row in sorted_rows:
            for i, cell in enumerate(row):
                if i < len(widths): widths[i] = max(widths[i], len(str(cell)))

        def _do_sort() -> None:
            nonlocal sorted_rows, cursor, scroll
            def _key(row):
                val = row[sort_col] if sort_col < len(row) else ""
                try:
                    return (0, float(val))
                except ValueError:
                    return (1, val.lower())
            sorted_rows.sort(key=_key, reverse=not sort_asc)
            cursor = 0; scroll = 0

        _do_sort()

        def _render() -> list[str]:
            lines: list[str] = []
            lines.append(f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}  {_DIM}(Tab: column  Space: asc/desc  Enter: confirm){_RESET}")
            hdr_parts = []
            for i, (h, w) in enumerate(zip(headers, widths)):
                arrow = " \u25b2" if sort_asc and i == sort_col else (" \u25bc" if not sort_asc and i == sort_col else "")
                if i == sort_col:
                    hdr_parts.append(f"{_BOLD}{_CYAN}{h.ljust(w)}{arrow}{_RESET}")
                else:
                    hdr_parts.append(f"{_BOLD}{h.ljust(w)}{_RESET}")
            lines.append(f"{_ERASE_LINE}\r  {'  '.join(hdr_parts)}")
            lines.append(f"{_ERASE_LINE}\r  {_DIM}{'  '.join('\u2500' * w for w in widths)}{_RESET}")
            for i in range(min(max_visible, len(sorted_rows))):
                idx = scroll + i
                if idx >= len(sorted_rows): break
                cells = "  ".join(str(sorted_rows[idx][j]).ljust(widths[j]) for j in range(min(len(widths), len(sorted_rows[idx]))))
                if idx == cursor:
                    lines.append(f"{_ERASE_LINE}\r{_REVERSE} > {cells}{_RESET_REVERSE}{_RESET}")
                else:
                    lines.append(f"{_ERASE_LINE}\r   {cells}")
            lines.append(f"{_ERASE_LINE}\r  {_DIM}{len(sorted_rows)} rows{_RESET}{_HIDE_CURSOR}")
            return lines

        def _write_lines(lines: list[str]) -> None:
            sys.stdout.write(_HIDE_CURSOR)
            for line in lines: sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        def _clear(line_count: int) -> None:
            for _ in range(line_count): sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR); sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                prev_count = 0
                while True:
                    _clear(prev_count)
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)
                    key = _read_raw_key(fd)
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count); return sorted_rows
                    if key.kind == _KEY_ENTER:
                        _clear(prev_count); return sorted_rows
                    if key.kind == _KEY_TAB:
                        sort_col = (sort_col + 1) % len(headers)
                        _do_sort()
                    elif key.kind == _KEY_SPACE:
                        sort_asc = not sort_asc
                        _do_sort()
                    elif key.kind == _KEY_UP:
                        cursor = max(0, cursor - 1)
                        if cursor < scroll: scroll = cursor
                    elif key.kind == _KEY_DOWN:
                        cursor = min(len(sorted_rows) - 1, cursor + 1)
                        if cursor >= scroll + max_visible: scroll = cursor - max_visible + 1
        except (termios.error, OSError):
            return sorted_rows

    # ── Notify ─────────────────────────────────────────────────────────

    def notify(self, title: str, message: str, level: str = "info") -> None:
        """Display a styled notification banner.

        Args:
            title: notification title
            message: notification message
            level: "info", "success", "warn", "error"
        """
        icons = {"info": "\u2139", "success": "\u2714", "warn": "\u26a0", "error": "\u2716"}
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

    def table(self, headers: list[str], rows: list[list[str]],
              title: str = "") -> None:
        """Display data in a formatted table with aligned columns.

        Args:
            headers: column header strings
            rows: list of rows, each a list of cell strings
            title: optional title above the table
        """
        if not headers:
            return
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        def _fmt_row(cells: list[str], color: str = "") -> str:
            parts = []
            for i, w in enumerate(widths):
                val = str(cells[i]) if i < len(cells) else ""
                parts.append(val.ljust(w))
            row_str = "  ".join(parts)
            if color:
                return f"{color}{row_str}{_RESET}"
            return row_str

        lines: list[str] = []
        if title:
            lines.append(f"  {_BOLD}{_CYAN}{title}{_RESET}")

        header_str = "  ".join(f"{_BOLD}{h.ljust(w)}{_RESET}" for h, w in zip(headers, widths))
        lines.append(f"  {_BOLD}{header_str}{_RESET}")
        lines.append(f"  {_DIM}{'  '.join('\u2500' * w for w in widths)}{_RESET}")

        for idx, row in enumerate(rows):
            color = _DIM if idx % 2 == 1 else ""
            lines.append(f"  {_fmt_row(row, color)}")

        for line in lines:
            self._io.write(line)

    # ── Diff display ──────────────────────────────────────────────────

    def diff(self, left_label: str, left_lines: list[str],
             right_label: str, right_lines: list[str],
             title: str = "") -> None:
        """Show a side-by-side diff with colored additions/removals.

        Args:
            left_label: label for left pane
            left_lines: content lines of left side
            right_label: label for right pane
            right_lines: content lines of right side
            title: optional title above the diff
        """
        width = _terminal_width()
        col_w = (width - 5) // 2

        def _trunc(s: str, w: int) -> str:
            if len(s) > w:
                return s[:w - 1] + "\u2026"
            return s

        if title:
            self._io.write(f"  {_BOLD}{_CYAN}{title}{_RESET}")

        header = f"  {_DIM}{left_label:^{col_w}}{_RESET}  \u2502  {_DIM}{right_label:^{col_w}}{_RESET}"
        self._io.write(header)
        sep = f"  {'  \u2500' * col_w}\u2500  \u253c\u2500  {'\u2500  ' * col_w}\u2500"
        self._io.write(f"{_DIM}{sep}{_RESET}")

        max_len = max(len(left_lines), len(right_lines))
        left_set = set(left_lines)
        right_set = set(right_lines)

        for i in range(max_len):
            l = left_lines[i] if i < len(left_lines) else ""
            r = right_lines[i] if i < len(right_lines) else ""

            if l and l not in right_set:
                l_display = f"{_RED}{_trunc(l, col_w)}{_RESET}"
            else:
                l_display = _trunc(l, col_w)

            if r and r not in left_set:
                r_display = f"{_GREEN}{_trunc(r, col_w)}{_RESET}"
            else:
                r_display = _trunc(r, col_w)

            self._io.write(f"  {l_display:<{col_w + 10}}  \u2502  {r_display}")

    def _edit_raw(self, message: str, default: str,
                  validator: "Callable[[str], str | None] | None") -> str:
        """Interactive edit using raw terminal input."""
        fd = self._get_fd()
        suffix = f" [{_DIM}{default}{_RESET}]" if default else ""
        buf: list[str] = list(default) if default else []
        cursor = 0
        error = ""

        def _render() -> str:
            display = "".join(buf)
            before = display[:cursor]
            after = display[cursor:]
            err = f"\n  {_RED}\u2718 {error}{_RESET}" if error else ""
            return f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}{suffix}: {before}{_REVERSE}|{_RESET_REVERSE}{after}{_HIDE_CURSOR}{err}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            if error:
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()

                    key = _read_raw_key(fd)

                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return default

                    if key.kind == _KEY_ENTER:
                        text = "".join(buf)
                        if validator:
                            err = validator(text)
                            if err:
                                error = err
                                continue
                        _clear()
                        return text if text else default

                    error = ""

                    if key.kind == _KEY_BACKSPACE:
                        if cursor > 0:
                            buf.pop(cursor - 1)
                            cursor -= 1

                    elif key.kind == _KEY_DELETE:
                        if cursor < len(buf):
                            buf.pop(cursor)

                    elif key.kind == _KEY_LEFT:
                        cursor = max(0, cursor - 1)

                    elif key.kind == _KEY_RIGHT:
                        cursor = min(len(buf), cursor + 1)

                    elif key.kind == _KEY_HOME:
                        cursor = 0

                    elif key.kind == _KEY_END:
                        cursor = len(buf)

                    elif key.kind in (_KEY_UP, _KEY_DOWN):
                        pass

                    elif key.kind == _KEY_CHAR:
                        buf.insert(cursor, key.char)
                        cursor += 1

        except (termios.error, OSError):
            return self._ask_fallback(message, default)

    # ── Screen helpers ─────────────────────────────────────────────────

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

    # ── Select with details ────────────────────────────────────────────

    def select_with_details(self, title: str, options: list[str],
                            details: list[str]) -> str:
        """Show an interactive selector with a detail pane below the list.

        Args:
            title: prompt text
            options: list of option strings
            details: parallel list of description strings (same length as options)

        Returns:
            the selected option string
        """
        if not options:
            return ""
        if len(options) == 1:
            return options[0]
        if not self._is_tty:
            return self._select_fallback(title, options)
        return self._select_with_details_raw(title, options, details)

    def _select_with_details_raw(self, title: str, options: list[str],
                                 details: list[str]) -> str:
        """Interactive select with details pane using raw terminal input."""
        fd = self._get_fd()
        query = ""
        cursor = 0
        scroll = 0
        max_visible = min(len(options), 15)
        width = _terminal_width()

        def _get_filtered() -> list[tuple[int, str]]:
            if query:
                return [(i, o) for i, o in enumerate(options) if query.lower() in o.lower()]
            return [(i, o) for i, o in enumerate(options)]

        def _render() -> list[str]:
            filtered = _get_filtered()
            lines: list[str] = []
            lines.append(f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}")
            if query:
                lines.append(f"{_ERASE_LINE}\r  Filter: {_DIM}{query}{_RESET}{_HIDE_CURSOR}")
            else:
                lines.append("")

            if not filtered:
                lines.append(f"{_ERASE_LINE}\r  {_DIM}No matching options{_RESET}")
            else:
                for i in range(min(max_visible, len(filtered))):
                    idx = scroll + i
                    if idx >= len(filtered):
                        break
                    orig_idx, text = filtered[idx]
                    display = _truncate(text, width - 5)
                    if idx == cursor:
                        lines.append(f"{_ERASE_LINE}\r{_REVERSE} > {display}{_RESET_REVERSE}{_RESET}")
                    else:
                        lines.append(f"{_ERASE_LINE}\r   {display}")

                # Detail pane for selected item
                if 0 <= cursor < len(filtered):
                    orig_idx = filtered[cursor][0]
                    if orig_idx < len(details):
                        detail = _truncate(details[orig_idx], width - 6)
                        lines.append(f"{_ERASE_LINE}\r")
                        lines.append(f"{_ERASE_LINE}\r  {_DIM}{detail}{_RESET}")

            if len(filtered) > max_visible:
                lines.append(f"{_ERASE_LINE}\r   {_DIM}({len(filtered)} items, {cursor + 1}/{len(filtered)}){_RESET}")

            lines.append(f"{_ERASE_LINE}\r  {_DIM}Enter: select  Esc: cancel  Type to filter{_RESET}{_HIDE_CURSOR}")
            return lines

        def _write_lines(lines: list[str]) -> None:
            sys.stdout.write(_HIDE_CURSOR)
            for line in lines:
                sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        def _clear(line_count: int) -> None:
            for _ in range(line_count):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                prev_count = 0
                while True:
                    _clear(prev_count)
                    filtered = _get_filtered()
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)

                    key = _read_raw_key(fd)

                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count)
                        return options[0]

                    if key.kind == _KEY_ENTER:
                        _clear(prev_count)
                        if 0 <= cursor < len(filtered):
                            return filtered[cursor][1]
                        return options[0]

                    if key.kind == _KEY_UP:
                        cursor = max(0, cursor - 1)
                        if cursor < scroll:
                            scroll = cursor

                    elif key.kind == _KEY_DOWN:
                        cursor = min(len(filtered) - 1, cursor + 1)
                        if cursor >= scroll + max_visible:
                            scroll = cursor - max_visible + 1

                    elif key.kind == _KEY_BACKSPACE:
                        if query:
                            query = query[:-1]
                            cursor = 0
                            scroll = 0

                    elif key.kind == _KEY_CHAR:
                        query += key.char
                        cursor = 0
                        scroll = 0

        except (termios.error, OSError):
            return self._select_fallback(title, options)

    # ── Select with preview ────────────────────────────────────────────

    def select_with_preview(self, title: str, options: list[str],
                            preview_fn: "Callable[[str], str]") -> str:
        """Show an interactive selector with a live preview panel.

        Args:
            title: prompt text
            options: list of option strings
            preview_fn: function that takes an option and returns preview text

        Returns:
            the selected option string
        """
        if not options:
            return ""
        if len(options) == 1:
            return options[0]
        if not self._is_tty:
            return self._select_fallback(title, options)
        return self._select_with_preview_raw(title, options, preview_fn)

    def _select_with_preview_raw(self, title: str, options: list[str],
                                 preview_fn: "Callable[[str], str]") -> str:
        """Interactive select with live preview using raw terminal input."""
        fd = self._get_fd()
        query = ""
        cursor = 0
        scroll = 0
        max_visible = min(len(options), 15)
        width = _terminal_width()
        list_w = min(width // 2, 40)
        preview_w = width - list_w - 3

        def _get_filtered() -> list[str]:
            if query:
                return [o for o in options if query.lower() in o.lower()]
            return list(options)

        def _render() -> list[str]:
            filtered = _get_filtered()
            lines: list[str] = []
            lines.append(f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}")
            if query:
                lines.append(f"{_ERASE_LINE}\r  Filter: {_DIM}{query}{_RESET}{_HIDE_CURSOR}")
            else:
                lines.append("")

            if not filtered:
                lines.append(f"{_ERASE_LINE}\r  {_DIM}No matching options{_RESET}")
            else:
                for i in range(min(max_visible, len(filtered))):
                    idx = scroll + i
                    if idx >= len(filtered):
                        break
                    text = _truncate(filtered[idx], list_w - 5)
                    if idx == cursor:
                        lines.append(f"{_ERASE_LINE}\r{_REVERSE} > {text}{_RESET_REVERSE}{_RESET}")
                    else:
                        lines.append(f"{_ERASE_LINE}\r   {text}")

                if 0 <= cursor < len(filtered):
                    try:
                        preview_text = preview_fn(filtered[cursor])
                    except Exception:
                        preview_text = "(preview unavailable)"
                    for i, pline in enumerate(preview_text.split("\n")[:max_visible + 2]):
                        t = _truncate(pline, preview_w)
                        if i < len(lines):
                            lines[i] = f"{lines[i]}{_DIM}\u2502{_RESET} {t}"
                        else:
                            lines.append(f"{' ' * (list_w + 2)}{_DIM}\u2502{_RESET} {t}")

            if len(filtered) > max_visible:
                lines.append(f"{_ERASE_LINE}\r   {_DIM}({len(filtered)} items, {cursor + 1}/{len(filtered)}){_RESET}")

            lines.append(f"{_ERASE_LINE}\r  {_DIM}Enter: select  Esc: cancel  Type to filter{_RESET}{_HIDE_CURSOR}")
            return lines

        def _write_lines(lines: list[str]) -> None:
            sys.stdout.write(_HIDE_CURSOR)
            for line in lines:
                sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        def _clear(line_count: int) -> None:
            for _ in range(line_count):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                prev_count = 0
                while True:
                    _clear(prev_count)
                    filtered = _get_filtered()
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)

                    key = _read_raw_key(fd)

                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count)
                        return options[0]

                    if key.kind == _KEY_ENTER:
                        _clear(prev_count)
                        if 0 <= cursor < len(filtered):
                            return filtered[cursor]
                        return options[0]

                    if key.kind == _KEY_UP:
                        cursor = max(0, cursor - 1)
                        if cursor < scroll:
                            scroll = cursor

                    elif key.kind == _KEY_DOWN:
                        cursor = min(len(filtered) - 1, cursor + 1)
                        if cursor >= scroll + max_visible:
                            scroll = cursor - max_visible + 1

                    elif key.kind == _KEY_BACKSPACE:
                        if query:
                            query = query[:-1]
                            cursor = 0
                            scroll = 0

                    elif key.kind == _KEY_CHAR:
                        query += key.char
                        cursor = 0
                        scroll = 0

        except (termios.error, OSError):
            return self._select_fallback(title, options)

    def confirm_multi(self, title: str, items: list[str],
                      default: bool = True) -> list[str]:
        """Show a multi-confirm prompt: list items and ask y/N for each.

        Args:
            title: prompt text
            items: list of items to confirm
            default: default answer for each item

        Returns:
            list of items that were confirmed (y)
        """
        if not items:
            return []
        if not self._is_tty:
            return self._confirm_multi_fallback(title, items, default)
        return self._confirm_multi_raw(title, items, default)

    def _confirm_multi_raw(self, title: str, items: list[str],
                           default: bool) -> list[str]:
        """Interactive multi-confirm using raw terminal input."""
        fd = self._get_fd()
        cursor = 0
        answers: dict[int, bool] = {}
        hint = "Y/n" if default else "y/N"
        width = _terminal_width()

        def _render() -> list[str]:
            lines: list[str] = []
            lines.append(f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}")
            lines.append("")

            for i, item in enumerate(items):
                answered = answers.get(i)
                if answered is True:
                    check = f"{_GREEN}\u2713{_RESET}"
                elif answered is False:
                    check = f"{_RED}\u2717{_RESET}"
                else:
                    check = "\u25cb"

                display = _truncate(item, width - 8)
                if i == cursor:
                    lines.append(f"{_ERASE_LINE}\r{_REVERSE} > {check} {display}{_RESET_REVERSE}{_RESET}")
                else:
                    lines.append(f"{_ERASE_LINE}\r   {check} {display}")

            n_yes = sum(1 for v in answers.values() if v)
            n_no = sum(1 for v in answers.values() if not v)
            lines.append(f"{_ERASE_LINE}\r")
            lines.append(f"{_ERASE_LINE}\r  {_DIM}y: yes  n: no  a: yes all  Esc: finish ({n_yes} yes, {n_no} no){_RESET}{_HIDE_CURSOR}")
            return lines

        def _write_lines(lines: list[str]) -> None:
            sys.stdout.write(_HIDE_CURSOR)
            for line in lines:
                sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        def _clear(line_count: int) -> None:
            for _ in range(line_count):
                sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}")
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with _RawTerminal(fd):
                prev_count = 0
                while True:
                    _clear(prev_count)
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)

                    key = _read_raw_key(fd)

                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count)
                        break

                    if key.kind == _KEY_ENTER:
                        cursor = min(len(items) - 1, cursor + 1)
                        if cursor >= len(items):
                            break

                    if key.kind == _KEY_UP:
                        cursor = max(0, cursor - 1)

                    elif key.kind == _KEY_DOWN:
                        cursor = min(len(items) - 1, cursor + 1)

                    elif key.kind == _KEY_CHAR:
                        c = key.char.lower()
                        if c == "y":
                            answers[cursor] = True
                            cursor = min(len(items) - 1, cursor + 1)
                        elif c == "n":
                            answers[cursor] = False
                            cursor = min(len(items) - 1, cursor + 1)
                        elif c == "a":
                            for i in range(len(items)):
                                if i not in answers:
                                    answers[i] = True
                            break

        except (termios.error, OSError):
            return self._confirm_multi_fallback(title, items, default)

        return [items[i] for i in sorted(answers) if answers[i]]

    def _confirm_multi_fallback(self, title: str, items: list[str],
                                default: bool) -> list[str]:
        """Line-mode fallback for non-TTY."""
        self._io.write(f"  {title}")
        for i, item in enumerate(items, 1):
            self._io.write(f"    {i}. {item}")
        hint = "Y/n" if default else "y/N"
        self._io.write(f"  Confirm all? [{hint}] ", end="")
        try:
            raw = self._io.read("").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return []
        if raw in ("y", "yes", ""):
            return list(items)
        return []

    # ── Progress bar ───────────────────────────────────────────────────

    def progress(self, label: str, current: int, total: int,
                 bar_width: int = 20) -> None:
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

    # ── Spinner ────────────────────────────────────────────────────────

    def spinner(self, message: str = "", rate: float = 0.1) -> "_InteractiveSpinner":
        """Return a context manager that shows a spinner while a task runs.

        Usage::

            with prompt.spinner("Loading") as s:
                do_work()
            s.ok("done")
            s.fail("error")
        """
        return _InteractiveSpinner(self, message, rate)


class _InteractiveSpinner:
    """Context manager that drives a spinner animation for InteractivePrompt."""

    _FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]

    def __init__(self, prompt: InteractivePrompt, message: str, rate: float) -> None:
        self._prompt = prompt
        self._message = message
        self._rate = rate
        self._stop = False
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_InteractiveSpinner":
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
