"""The confirm_* family. — InteractivePrompt mixin.

Every confirm_* variant: dangerous, timeout, list, text, delete, overwrite, and the preview/details/warning/countdown/phrase combinators, all layered on the core confirm.
"""

from __future__ import annotations

import sys
import termios
import time
from collections.abc import Callable

from domain.shell._internal.interactive_base import InteractiveBase
from domain.shell._internal.interactive_term import (
    _BOLD,
    _CURSOR_UP,
    _CYAN,
    _DIM,
    _ERASE_LINE,
    _GREEN,
    _HIDE_CURSOR,
    _KEY_BACKSPACE,
    _KEY_CHAR,
    _KEY_CTRL_C,
    _KEY_DOWN,
    _KEY_ENTER,
    _KEY_ESC,
    _KEY_LEFT,
    _KEY_RIGHT,
    _KEY_SPACE,
    _KEY_TAB,
    _KEY_UP,
    _RED,
    _RESET,
    _RESET_REVERSE,
    _REVERSE,
    _SHOW_CURSOR,
    _YELLOW,
    _rewind,
)
from domain.shell._internal.interactive_widget import _WidgetPrompts


class _ConfirmPrompts(_WidgetPrompts, InteractiveBase):
    """The confirm_* family..

    Split out of InteractivePrompt by .splitwork/split_interactive.py.
    """

    def confirm_action(self, action: str, details: str = "", danger: bool = False) -> bool:
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

    def _confirm_action_raw(self, action: str, details: str, danger: bool) -> bool:
        selected = False

        prev_rows = 0  # rows the frame on screen occupies

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
            # Erases the row under the cursor first, so the frame's last line
            # goes too. The old spelling moved up *before* erasing and left it.
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return False
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return selected
                    if key.matches(_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear()
                            return True
                        if key.char in ("n", "N"):
                            _clear()
                            return False
        except (termios.error, OSError):
            return self._confirm_fallback(f"{action}? (y/N)", False)

    def confirm_dangerous(self, action: str, phrase: str = "yes, I am sure") -> bool:
        """Confirm a dangerous action by typing a phrase."""
        if not self._is_tty:
            raw = self._ask_fallback(f"{action}. Type '{phrase}' to confirm", "")
            return raw.strip().lower() == phrase.lower()
        return self._confirm_dangerous_raw(action, phrase)

    def _confirm_dangerous_raw(self, action: str, phrase: str) -> bool:
        buf: list[str] = []

        prev_rows = 0  # rows the frame on screen occupies

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
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return False
                    if key.kind == _KEY_ENTER:
                        typed = "".join(buf).strip().lower()
                        _clear()
                        return typed == phrase.lower()
                    if key.kind == _KEY_BACKSPACE:
                        if buf:
                            buf.pop()
                    elif key.kind == _KEY_CHAR:
                        buf.append(key.char)
        except (termios.error, OSError):
            raw = self._ask_fallback(f"{action}. Type '{phrase}' to confirm", "")
            return raw.strip().lower() == phrase.lower()

    def confirm_timeout(self, message: str, timeout: float = 5.0, default: bool = True) -> bool:
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

    def _confirm_timeout_raw(self, message: str, timeout: float, default: bool) -> bool:
        selected = not default  # opposite of default for visual
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
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                start = _t.monotonic()
                while True:
                    elapsed = _t.monotonic() - start
                    remaining = max(0, timeout - elapsed)
                    if remaining <= 0:
                        _clear()
                        return default
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render(remaining)}")
                    sys.stdout.flush()
                    pending = self._session.read_key(timeout=0.1)
                    if pending:
                        key = pending
                        if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                            _clear()
                            return default
                        if key.kind == _KEY_ENTER:
                            _clear()
                            return selected
                        if key.matches(_KEY_LEFT, _KEY_RIGHT):
                            selected = not selected
                        if key.kind == _KEY_CHAR:
                            if key.char in ("y", "Y"):
                                _clear()
                                return True
                            if key.char in ("n", "N"):
                                _clear()
                                return False
        except (termios.error, OSError):
            return default

    def confirm_list(self, title: str, items: list[str], default: bool = True) -> list[str]:
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

    def _confirm_list_raw(self, title: str, items: list[str], default: bool) -> list[str]:
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
            lines.append(
                f"{_ERASE_LINE}\r  {_DIM}Space: toggle  Enter: confirm  y/n: toggle all{_RESET}{_HIDE_CURSOR}"
            )
            return lines

        def _write_lines(lines: list[str]) -> None:
            sys.stdout.write(_HIDE_CURSOR)
            for line in lines:
                sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

        def _clear(line_count: int) -> None:
            _rewind(line_count)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                prev_count = 0
                while True:
                    _clear(prev_count)
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count)
                        return []
                    if key.kind == _KEY_ENTER:
                        _clear(prev_count)
                        return [items[i] for i in range(len(items)) if answers.get(i, default)]
                    if key.matches(_KEY_UP):
                        cursor = max(0, cursor - 1)
                    elif key.matches(_KEY_DOWN):
                        cursor = min(len(items) - 1, cursor + 1)
                    elif key.kind == _KEY_SPACE:
                        answers[cursor] = not answers.get(cursor, default)
                    elif key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            for i in range(len(items)):
                                answers[i] = True
                        elif key.char in ("n", "N"):
                            for i in range(len(items)):
                                answers[i] = False
        except (termios.error, OSError):
            return self._confirm_multi_fallback(title, items, default)

    def confirm_text(self, message: str, target: str, hint: str = "") -> bool:
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
        buf: list[str] = []

        prev_rows = 0  # rows the frame on screen occupies

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
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return False
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return "".join(buf) == target
                    if key.kind == _KEY_BACKSPACE:
                        if buf:
                            buf.pop()
                    elif key.kind == _KEY_CHAR:
                        buf.append(key.char)
        except (termios.error, OSError):
            raw = self._ask_fallback(f"{message} (type '{target}')", "")
            return raw.strip() == target

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

    def confirm_overwrite(self, path: str) -> bool:
        """Confirm overwriting an existing file.

        Args:
            path: file path to overwrite

        Returns:
            True if confirmed
        """
        return self.confirm_dangerous(f"Overwrite {path}", phrase="overwrite")

    def confirm_with_preview(self, message: str, preview: str, default: bool = False) -> bool:
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

    def _confirm_with_preview_raw(self, message: str, preview: str, default: bool) -> bool:
        selected = not default

        prev_rows = 0  # rows the frame on screen occupies

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
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return default
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return selected
                    if key.matches(_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear()
                            return True
                        if key.char in ("n", "N"):
                            _clear()
                            return False
        except (termios.error, OSError):
            return default

    def confirm_with_details(
        self, message: str, details: dict[str, str], default: bool = False
    ) -> bool:
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

    def _confirm_with_details_raw(
        self, message: str, details: dict[str, str], default: bool
    ) -> bool:
        selected = not default

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            if selected:
                yes_text = f"{_REVERSE}{_GREEN} Yes {_RESET_REVERSE}{_RESET}"
                no_text = " No "
            else:
                yes_text = " Yes "
                no_text = f"{_REVERSE}{_RED} No {_RESET_REVERSE}{_RESET}"
            detail_lines = "\n".join(f"  {_DIM}{k}:{_RESET} {v}" for k, v in details.items())
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                f"{detail_lines}\n"
                f"  {yes_text}  {no_text}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return default
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return selected
                    if key.matches(_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear()
                            return True
                        if key.char in ("n", "N"):
                            _clear()
                            return False
        except (termios.error, OSError):
            return default

    def confirm_with_preview_and_edit(
        self,
        message: str,
        preview: str,
        edit_prompt: str = "Edit: ",
        default: bool = False,
    ) -> tuple[bool, str]:
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

    def _confirm_with_preview_and_edit_raw(
        self, message: str, preview: str, edit_prompt: str, default: bool
    ) -> tuple[bool, str]:
        selected = not default
        text = preview

        prev_rows = 0  # rows the frame on screen occupies

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
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return (default, text)
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return (selected, text)
                    if key.matches(_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear()
                            return (True, text)
                        if key.char in ("n", "N"):
                            _clear()
                            return (False, text)
                        if key.char in ("e", "E"):
                            _clear()
                            edited = self._ask_fallback(edit_prompt, text)
                            return (True, edited)
        except (termios.error, OSError):
            return (default, text)

    def confirm_with_warning(self, message: str, warning: str, default: bool = False) -> bool:
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

    def _confirm_with_warning_raw(self, message: str, warning: str, default: bool) -> bool:
        selected = not default

        prev_rows = 0  # rows the frame on screen occupies

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
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return default
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return selected
                    if key.matches(_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear()
                            return True
                        if key.char in ("n", "N"):
                            _clear()
                            return False
        except (termios.error, OSError):
            return default

    def confirm_with_countdown(
        self, message: str, timeout: int = 10, default: bool = False
    ) -> bool:
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

    def _confirm_with_countdown_raw(self, message: str, timeout: int, default: bool) -> bool:
        selected = not default
        remaining = timeout

        prev_rows = 0  # rows the frame on screen occupies

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
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                start = time.time()
                while True:
                    elapsed = int(time.time() - start)
                    remaining = max(0, timeout - elapsed)
                    if remaining == 0:
                        _clear()
                        return default
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    # Countdown does not tick. This used to pass a deadline to a decoder with no
                    # timeout parameter, so every countdown prompt raised TypeError on its first
                    # frame; the deadline was dropped rather than revived, and a timed session
                    # read is the separate repair.
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return default
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return selected
                    if key.matches(_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                        start = time.time()
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear()
                            return True
                        if key.char in ("n", "N"):
                            _clear()
                            return False
                        start = time.time()
        except (termios.error, OSError):
            return default

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
        buf: list[str] = []

        prev_rows = 0  # rows the frame on screen occupies

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
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return False
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return "".join(buf).lower() == phrase.lower()
                    if key.kind == _KEY_BACKSPACE:
                        if buf:
                            buf.pop()
                    elif key.kind == _KEY_CHAR:
                        buf.append(key.char)
                        if "".join(buf).lower() == phrase.lower():
                            _clear()
                            return True
        except (termios.error, OSError):
            return False

    def confirm_list_with_preview(
        self,
        message: str,
        items: list[str],
        preview_fn: Callable[[str], str],
        default: bool = True,
    ) -> list[str]:
        """Confirm a list of items with preview, returning confirmed items."""
        if not self._is_tty:
            return items if default else []
        return self._confirm_list_with_preview_raw(message, items, preview_fn, default)

    def _confirm_list_with_preview_raw(
        self,
        message: str,
        items: list[str],
        preview_fn: Callable[[str], str],
        default: bool,
    ) -> list[str]:
        idx = 0
        states = dict.fromkeys(items, default)

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            preview_text = preview_fn(items[idx])
            preview_lines = preview_text.split("\n")[:4]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            lines = []
            for i, item in enumerate(items):
                check = f"{_GREEN}\u2714{_RESET}" if states[item] else f"{_RED}\u2718{_RESET}"
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                lines.append(f"  {prefix}{check} {color}{item}{_RESET}")
            count = sum(1 for v in states.values() if v)
            count_str = f"  {_DIM}{count}/{len(items)} confirmed{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(space=toggle, enter=confirm){_RESET}\n"
                + "\n".join(lines)
                + f"\n{count_str}\n{preview_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return [item for item, v in states.items() if v]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return [item for item, v in states.items() if v]
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % len(items)
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % len(items)
                    elif key.kind == _KEY_CHAR and key.char == " ":
                        states[items[idx]] = not states[items[idx]]
        except (termios.error, OSError):
            return [item for item, v in states.items() if v]

    def confirm_with_preview_and_timeout(
        self, message: str, preview: str, timeout: int = 10, default: bool = False
    ) -> bool:
        """Confirm with preview and auto-confirm countdown."""
        if not self._is_tty:
            return self._confirm_fallback(message, default)
        return self._confirm_with_preview_and_timeout_raw(message, preview, timeout, default)

    def _confirm_with_preview_and_timeout_raw(
        self, message: str, preview: str, timeout: int, default: bool
    ) -> bool:
        selected = not default
        remaining = timeout

        prev_rows = 0  # rows the frame on screen occupies

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
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                start = time.time()
                while True:
                    elapsed = int(time.time() - start)
                    remaining = max(0, timeout - elapsed)
                    if remaining == 0:
                        _clear()
                        return default
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    # Countdown does not tick. This used to pass a deadline to a decoder with no
                    # timeout parameter, so every countdown prompt raised TypeError on its first
                    # frame; the deadline was dropped rather than revived, and a timed session
                    # read is the separate repair.
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return default
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return selected
                    if key.matches(_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                        start = time.time()
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear()
                            return True
                        if key.char in ("n", "N"):
                            _clear()
                            return False
                        start = time.time()
        except (termios.error, OSError):
            return default

    def confirm_with_preview_and_countdown(
        self, message: str, preview: str, timeout: int = 10, default: bool = True
    ) -> bool:
        """Confirm with preview and countdown, auto-confirming on timeout."""
        if not self._is_tty:
            return self._confirm_fallback(message, default)
        return self._confirm_with_preview_and_countdown_raw(message, preview, timeout, default)

    def _confirm_with_preview_and_countdown_raw(
        self, message: str, preview: str, timeout: int, default: bool
    ) -> bool:
        selected = not default

        prev_rows = 0  # rows the frame on screen occupies

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
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                start = time.time()
                while True:
                    elapsed = int(time.time() - start)
                    remaining = max(0, timeout - elapsed)
                    if remaining == 0:
                        _clear()
                        return default
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    # Countdown does not tick. This used to pass a deadline to a decoder with no
                    # timeout parameter, so every countdown prompt raised TypeError on its first
                    # frame; the deadline was dropped rather than revived, and a timed session
                    # read is the separate repair.
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return default
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return selected
                    if key.matches(_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                        start = time.time()
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear()
                            return True
                        if key.char in ("n", "N"):
                            _clear()
                            return False
                        start = time.time()
        except (termios.error, OSError):
            return default

    def confirm_with_preview_and_edit_with_timeout(
        self,
        message: str,
        preview: str,
        edit_prompt: str = "Edit:",
        timeout: int = 10,
        default: bool = True,
    ) -> tuple[bool, str]:
        """Confirm with preview, optional edit, and timeout."""
        if not self._is_tty:
            return (self._confirm_fallback(message, default), preview)
        return self._confirm_with_preview_and_edit_with_timeout_raw(
            message, preview, edit_prompt, timeout, default
        )

    def _confirm_with_preview_and_edit_with_timeout_raw(
        self, message: str, preview: str, edit_prompt: str, timeout: int, default: bool
    ) -> tuple[bool, str]:
        selected = not default
        text = preview

        prev_rows = 0  # rows the frame on screen occupies

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
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                start = time.time()
                while True:
                    elapsed = int(time.time() - start)
                    remaining = max(0, timeout - elapsed)
                    if remaining == 0:
                        _clear()
                        return (default, text)
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    # Countdown does not tick. This used to pass a deadline to a decoder with no
                    # timeout parameter, so every countdown prompt raised TypeError on its first
                    # frame; the deadline was dropped rather than revived, and a timed session
                    # read is the separate repair.
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return (default, text)
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return (selected, text)
                    if key.matches(_KEY_LEFT):
                        selected = True
                        start = time.time()
                    elif key.matches(_KEY_RIGHT):
                        selected = False
                        start = time.time()
                    elif key.kind == _KEY_TAB:
                        selected = not selected
                        start = time.time()
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear()
                            return (True, text)
                        if key.char in ("n", "N"):
                            _clear()
                            return (False, text)
                        if key.char in ("e", "E"):
                            _clear()
                            edited = self._ask_fallback(edit_prompt, text)
                            return (True, edited)
                        start = time.time()
        except (termios.error, OSError):
            return (default, text)

    def confirm_with_countdown_and_preview(
        self, message: str, preview: str, timeout: int = 10, default: bool = True
    ) -> bool:
        """Confirm with preview and countdown, auto-confirming on timeout."""
        if not self._is_tty:
            return self._confirm_fallback(message, default)
        return self._confirm_with_countdown_and_preview_raw(message, preview, timeout, default)

    def _confirm_with_countdown_and_preview_raw(
        self, message: str, preview: str, timeout: int, default: bool
    ) -> bool:
        selected = not default

        prev_rows = 0  # rows the frame on screen occupies

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
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                start = time.time()
                while True:
                    elapsed = int(time.time() - start)
                    remaining = max(0, timeout - elapsed)
                    if remaining == 0:
                        _clear()
                        return default
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    # Countdown does not tick. This used to pass a deadline to a decoder with no
                    # timeout parameter, so every countdown prompt raised TypeError on its first
                    # frame; the deadline was dropped rather than revived, and a timed session
                    # read is the separate repair.
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return default
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return selected
                    if key.matches(_KEY_LEFT, _KEY_RIGHT):
                        selected = not selected
                        start = time.time()
                    if key.kind == _KEY_CHAR:
                        if key.char in ("y", "Y"):
                            _clear()
                            return True
                        if key.char in ("n", "N"):
                            _clear()
                            return False
                        start = time.time()
        except (termios.error, OSError):
            return default
