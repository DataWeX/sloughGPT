"""Core input primitives. — InteractivePrompt mixin.

The five prompts everything else composes: select, select_multi, confirm, ask and confirm_multi, each with its raw and fallback paths. Every cross-family call in the class lands here, which makes this the substrate the other families are written against.
"""

from __future__ import annotations

import sys
import termios

from domain.shell._internal.interactive_base import InteractiveBase
from domain.shell._internal.interactive_term import (
    _BOLD,
    _CURSOR_UP,
    _CYAN,
    _DIM,
    _ERASE_LINE,
    _GREEN,
    _HIDE_CURSOR,
    _KEY_ALT,
    _KEY_BACKSPACE,
    _KEY_CHAR,
    _KEY_CTRL,
    _KEY_CTRL_C,
    _KEY_DELETE,
    _KEY_DOWN,
    _KEY_END,
    _KEY_ENTER,
    _KEY_ESC,
    _KEY_HOME,
    _KEY_LEFT,
    _KEY_RIGHT,
    _KEY_UP,
    _RED,
    _RESET,
    _RESET_REVERSE,
    _REVERSE,
    _SHOW_CURSOR,
    _terminal_width,
    _truncate,
)
from domain.shell._internal.textedit import (
    delete_back,
    delete_forward,
    word_backward,
    word_forward,
)


class _InputPrompts(InteractiveBase):
    """Core input primitives..

    Split out of InteractivePrompt by .splitwork/split_interactive.py.
    """

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
        query = ""
        qcur = 0  # caret into `query`; starts at the end, as it is empty
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
                lines.append(
                    f"{_ERASE_LINE}\r  Filter: {_DIM}"
                    f"{self._filter_view(query, qcur)}{_RESET}{_HIDE_CURSOR}"
                )
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
                lines.append(
                    f"{_ERASE_LINE}\r   {_DIM}({len(filtered)} items, {cursor + 1}/{len(filtered)}){_RESET}"
                )

            lines.append(
                f"{_ERASE_LINE}\r  {_DIM}Enter: select  Esc: cancel  Type to filter{_RESET}{_HIDE_CURSOR}"
            )
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
            with self._session.raw():
                prev_count = 0
                while True:
                    _clear(prev_count)
                    filtered = _get_filtered()
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)

                    key = self._session.read_key()

                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count)
                        return options[0]

                    if key.kind == _KEY_ENTER:
                        _clear(prev_count)
                        if 0 <= cursor < len(filtered):
                            return filtered[cursor]
                        return options[0]

                    if key.matches(_KEY_UP):
                        cursor = max(0, cursor - 1)
                        scroll = min(scroll, cursor)

                    elif key.matches(_KEY_DOWN):
                        cursor = min(len(filtered) - 1, cursor + 1)
                        if cursor >= scroll + max_visible:
                            scroll = cursor - max_visible + 1

                    else:
                        # Everything else belongs to the filter box. The helper
                        # owns the caret and reports whether the *text*
                        # changed, so a caret-only move — arrows, Home, End —
                        # repositions without resetting the selection, while
                        # any real edit still snaps back to the top.
                        query, qcur, edited = self._filter_key(query, qcur, key)
                        if edited:
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
        query = ""
        qcur = 0  # caret into `query`, starts at the end
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
                lines.append(
                    f"{_ERASE_LINE}\r  Filter: {self._filter_view(query, qcur)}{_HIDE_CURSOR}"
                )
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
                lines.append(
                    f"{_ERASE_LINE}\r   ({len(filtered)} items, {cursor + 1}/{len(filtered)})"
                )

            n_checked = len(checked)
            lines.append(
                f"{_ERASE_LINE}\r  Space: toggle  Enter: confirm ({n_checked} selected)  Esc: cancel{_HIDE_CURSOR}"
            )
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
            with self._session.raw():
                prev_count = 0
                while True:
                    _clear(prev_count)
                    filtered = _get_filtered()
                    lines = _render()
                    _write_lines(lines)
                    prev_count = len(lines)

                    key = self._session.read_key()

                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear(prev_count)
                        return []

                    if key.kind == _KEY_ENTER:
                        _clear(prev_count)
                        return [options[i] for i in sorted(checked)]

                    if key.matches(_KEY_UP):
                        cursor = max(0, cursor - 1)
                        scroll = min(scroll, cursor)

                    elif key.matches(_KEY_DOWN):
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

                    else:
                        # One editor owns every caret key. `edited` is
                        # False when only the caret moved, so arrows
                        # reposition without re-running the site's filter work.
                        query, qcur, edited = self._filter_key(query, qcur, key)
                        if edited:
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
        hint = "Y/n" if default else "y/N"
        selected = default

        def _render() -> str:
            if selected:
                yes_text = f"{_REVERSE}{_GREEN} Yes {_RESET_REVERSE}{_RESET}"
                no_text = " No "
            else:
                yes_text = " Yes "
                no_text = f"{_REVERSE}{_RED} No {_RESET_REVERSE}{_RESET}"
            return f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET} [{hint}]  {yes_text}  {no_text}{_HIDE_CURSOR}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()

                    key = self._session.read_key()

                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return not default

                    if key.kind == _KEY_ENTER:
                        _clear()
                        return selected

                    if key.matches(_KEY_UP, _KEY_DOWN):
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
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{_render()}")
                    sys.stdout.flush()

                    key = self._session.read_key()

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

                    elif key.matches(_KEY_LEFT):
                        cursor = max(0, cursor - 1)

                    elif key.matches(_KEY_RIGHT):
                        cursor = min(len(buf), cursor + 1)

                    elif key.matches(_KEY_LEFT, ctrl=True):
                        # Ctrl+Left — word back (readline backward-word).
                        # `matches` names the modifier, so the plain-Left
                        # branch above cannot swallow this chord.
                        cursor = word_backward(buf, cursor)

                    elif key.matches(_KEY_RIGHT, ctrl=True):
                        # Ctrl+Right — word forward (readline forward-word)
                        cursor = word_forward(buf, cursor)

                    elif key.matches(_KEY_CTRL, char="w"):
                        # Ctrl+W — erase the word before the caret
                        # (readline unix-word-rubout). No kill ring in a
                        # prompt, so the removed text is discarded.
                        cursor, _ = delete_back(buf, cursor)

                    elif key.matches(_KEY_ALT, char="d"):
                        # Alt+D — erase the word after the caret (kill-word).
                        # The caret is returned unchanged by design.
                        cursor, _ = delete_forward(buf, cursor)

                    elif key.kind == _KEY_HOME:
                        cursor = 0

                    elif key.kind == _KEY_END:
                        cursor = len(buf)

                    elif key.matches(_KEY_UP, _KEY_DOWN):
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

    def confirm_multi(self, title: str, items: list[str], default: bool = True) -> list[str]:
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

    def _confirm_multi_raw(self, title: str, items: list[str], default: bool) -> list[str]:
        """Interactive multi-confirm using raw terminal input."""
        cursor = 0
        answers: dict[int, bool] = {}
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
                    lines.append(
                        f"{_ERASE_LINE}\r{_REVERSE} > {check} {display}{_RESET_REVERSE}{_RESET}"
                    )
                else:
                    lines.append(f"{_ERASE_LINE}\r   {check} {display}")

            n_yes = sum(1 for v in answers.values() if v)
            n_no = sum(1 for v in answers.values() if not v)
            lines.append(f"{_ERASE_LINE}\r")
            lines.append(
                f"{_ERASE_LINE}\r  {_DIM}y: yes  n: no  a: yes all  Esc: finish ({n_yes} yes, {n_no} no){_RESET}{_HIDE_CURSOR}"
            )
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
                        break

                    if key.kind == _KEY_ENTER:
                        cursor = min(len(items) - 1, cursor + 1)
                        if cursor >= len(items):
                            break

                    if key.matches(_KEY_UP):
                        cursor = max(0, cursor - 1)

                    elif key.matches(_KEY_DOWN):
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

    def _confirm_multi_fallback(self, title: str, items: list[str], default: bool) -> list[str]:
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
