"""Whole-prompt flows. — InteractivePrompt mixin.

Prompts that own a full interaction rather than a single widget: edit, password, pager, multi_choice, file_browser, history_search, process_manager, log_viewer, config_editor, diff_viewer, wizard, form, spinner-free waits.
"""

from __future__ import annotations

import sys
import termios
from collections.abc import Callable

from domain.shell._internal.interactive_base import InteractiveBase
from domain.shell._internal.interactive_input import _InputPrompts
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
    _KEY_PAGE_DOWN,
    _KEY_PAGE_UP,
    _KEY_RIGHT,
    _KEY_SPACE,
    _KEY_UP,
    _RED,
    _RESET,
    _RESET_REVERSE,
    _REVERSE,
    _SHOW_CURSOR,
    _YELLOW,
    _rewind,
    _terminal_height,
)
from domain.shell._internal.textedit import (
    delete_back,
    delete_forward,
    word_backward,
    word_forward,
)


class _PromptPrompts(_InputPrompts, InteractiveBase):
    """Whole-prompt flows..

    Split out of InteractivePrompt by .splitwork/split_interactive.py.
    """

    def edit(
        self,
        message: str,
        default: str = "",
        validator: Callable[[str], str | None] | None = None,
    ) -> str:
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

    def _edit_raw(
        self, message: str, default: str, validator: Callable[[str], str | None] | None
    ) -> str:
        """Interactive edit using raw terminal input."""
        suffix = f" [{_DIM}{default}{_RESET}]" if default else ""
        buf: list[str] = list(default) if default else []
        cursor = 0
        error = ""

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            display = "".join(buf)
            before = display[:cursor]
            after = display[cursor:]
            err = f"\n  {_RED}\u2718 {error}{_RESET}" if error else ""
            return f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}{suffix}: {before}{_REVERSE}|{_RESET_REVERSE}{after}{_HIDE_CURSOR}{err}"

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
                        # (unix-word-rubout). No kill ring in a prompt, so
                        # the removed text is discarded.
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
                        pass

                    elif key.kind == _KEY_CHAR:
                        buf.insert(cursor, key.char)
                        cursor += 1

        except (termios.error, OSError):
            return self._ask_fallback(message, default)

    def password(self, message: str) -> str:
        """Interactive password input with masked characters."""
        if not self._is_tty:
            return self._ask_fallback(message, "")
        return self._password_raw(message)

    def _password_raw(self, message: str) -> str:
        buf: list[str] = []
        cursor = 0

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            masked = "*" * len(buf)
            before = masked[:cursor]
            after = masked[cursor:]
            return f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}: {before}{_REVERSE}|{_RESET_REVERSE}{after}{_HIDE_CURSOR}"

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
                        return ""
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return "".join(buf)
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
                        # Ctrl+Left — word back (backward-word)
                        cursor = word_backward(buf, cursor)
                    elif key.matches(_KEY_RIGHT, ctrl=True):
                        # Ctrl+Right — word forward (forward-word)
                        cursor = word_forward(buf, cursor)
                    elif key.matches(_KEY_CTRL, char="w"):
                        # Ctrl+W — erase the word before the caret
                        cursor, _ = delete_back(buf, cursor)
                    elif key.matches(_KEY_ALT, char="d"):
                        # Alt+D — erase the word after the caret; caret stays
                        cursor, _ = delete_forward(buf, cursor)
                    elif key.kind == _KEY_HOME:
                        cursor = 0
                    elif key.kind == _KEY_END:
                        cursor = len(buf)
                    elif key.kind == _KEY_CHAR:
                        buf.insert(cursor, key.char)
                        cursor += 1
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
        lines = content.split("\n")
        total = len(lines)
        height = _terminal_height() - 3  # title + footer
        offset = 0

        def _render() -> None:
            sys.stdout.write("\033[2J\033[H")  # clear + home
            sys.stdout.write(f"{_BOLD}{_CYAN}{title}{_RESET}  ")
            sys.stdout.write(f"{_DIM}(↑/↓/j/k scroll, q quit){_RESET}\n")
            visible = lines[offset : offset + height]
            for line in visible:
                sys.stdout.write(f"  {line}\n")
            pos = f"Line {offset + 1}-{min(offset + height, total)} of {total}"
            sys.stdout.write(f"{_DIM}{pos}{_RESET}")
            sys.stdout.flush()

        try:
            with self._session.raw():
                _render()
                while True:
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC, _KEY_CHAR) and key.char == "q":
                        break
                    elif key.matches(_KEY_UP) or (key.kind == _KEY_CHAR and key.char == "k"):
                        offset = max(0, offset - 1)
                    elif key.matches(_KEY_DOWN) or (key.kind == _KEY_CHAR and key.char == "j"):
                        offset = min(total - height, offset + 1)
                    elif key.kind == _KEY_PAGE_UP:
                        offset = max(0, offset - height)
                    elif key.kind == _KEY_PAGE_DOWN:
                        offset = min(total - height, offset + height)
                    elif key.kind == _KEY_HOME:
                        offset = 0
                    elif key.kind == _KEY_END:
                        offset = max(0, total - height)
                    offset = max(offset, 0)
                    _render()
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.flush()
        except (termios.error, OSError):
            self._io.write(content)

    def file_browser(self, title: str, start_dir: str = ".", pattern: str = "*") -> str | None:
        """Interactive file browser with directory navigation."""
        import glob as _glob
        import os as _os

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

    def _file_browser_raw(
        self, title: str, start: str, list_fn: Callable[[str], list[str]]
    ) -> str | None:
        import os as _os

        current = start
        cursor = 0
        query = ""
        qcur = 0  # caret into `query`, starts at the end
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
                lines.append(
                    f"{_ERASE_LINE}\r  Filter: {_DIM}{self._filter_view(query, qcur)}{_RESET}{_HIDE_CURSOR}"
                )
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
            lines.append(
                f"{_ERASE_LINE}\r  {_DIM}Enter: select  Esc: cancel  Backspace: parent{_RESET}{_HIDE_CURSOR}"
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
                        return None
                    if key.kind == _KEY_ENTER:
                        if 0 <= cursor < len(filtered):
                            selected = filtered[cursor]
                            if selected.startswith("\U0001f4c1"):
                                dirname = selected[2:].rstrip("/")
                                current = _os.path.abspath(_os.path.join(current, dirname))
                                cursor = 0
                                query = ""
                                qcur = 0  # caret into `query`, starts at the end
                            else:
                                _clear(prev_count)
                                return _os.path.join(current, selected[2:])
                    if key.matches(_KEY_UP):
                        cursor = max(0, cursor - 1)
                    elif key.matches(_KEY_DOWN):
                        cursor = min(len(filtered) - 1, cursor + 1)
                    elif key.kind == _KEY_BACKSPACE and not query:
                        # Empty filter: backspace leaves the directory rather than the text.
                        if current != "/":
                            current = _os.path.dirname(current)
                            cursor = 0
                    else:
                        # One editor owns every caret key. `edited` is
                        # False when only the caret moved, so arrows
                        # reposition without re-running the site's filter work.
                        query, qcur, edited = self._filter_key(query, qcur, key)
                        if edited:
                            cursor = 0
        except (termios.error, OSError):
            return None

    def multi_choice(
        self, title: str, options: list[str], defaults: list[int] | None = None
    ) -> list[str]:
        """Select multiple options with Space to toggle, Enter to confirm."""
        if not options:
            return []
        if not self._is_tty:
            return self._select_multi_fallback(title, options)
        return self._multi_choice_raw(title, options, defaults or [])

    def _multi_choice_raw(self, title: str, options: list[str], defaults: list[int]) -> list[str]:
        selected: set[int] = set(defaults)
        cursor = 0
        query = ""
        qcur = 0  # caret into `query`, starts at the end
        max_visible = min(len(options), 15)

        def _get_filtered() -> list[tuple[int, str]]:
            if query:
                return [(i, o) for i, o in enumerate(options) if query.lower() in o.lower()]
            return [(i, o) for i, o in enumerate(options)]

        def _render() -> list[str]:
            filtered = _get_filtered()
            lines: list[str] = []
            lines.append(
                f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}  {_DIM}({len(selected)} selected){_RESET}"
            )
            if query:
                lines.append(
                    f"{_ERASE_LINE}\r  Filter: {_DIM}{self._filter_view(query, qcur)}{_RESET}{_HIDE_CURSOR}"
                )
            else:
                lines.append("")
            if not filtered:
                lines.append(f"{_ERASE_LINE}\r  {_DIM}No matching options{_RESET}")
            else:
                for i in range(min(max_visible, len(filtered))):
                    orig_idx, text = filtered[i]
                    check = (
                        f"{_GREEN}\u2611{_RESET}"
                        if orig_idx in selected
                        else f"{_DIM}\u2610{_RESET}"
                    )
                    if orig_idx == cursor:
                        lines.append(
                            f"{_ERASE_LINE}\r{_REVERSE} {check} {text}{_RESET_REVERSE}{_RESET}"
                        )
                    else:
                        lines.append(f"{_ERASE_LINE}\r  {check} {text}")
            lines.append(
                f"{_ERASE_LINE}\r  {_DIM}Space: toggle  Enter: confirm  Type to filter{_RESET}{_HIDE_CURSOR}"
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
                        return []
                    if key.kind == _KEY_ENTER:
                        _clear(prev_count)
                        filtered = _get_filtered()
                        return [o for i, o in filtered if i in selected]
                    if key.matches(_KEY_UP):
                        cursor = max(0, cursor - 1)
                    elif key.matches(_KEY_DOWN):
                        cursor = min(len(options) - 1, cursor + 1)
                    elif key.kind == _KEY_SPACE:
                        if cursor in selected:
                            selected.discard(cursor)
                        else:
                            selected.add(cursor)
                    else:
                        # One editor owns every caret key. `edited` is
                        # False when only the caret moved, so arrows
                        # reposition without re-running the site's filter work.
                        query, qcur, edited = self._filter_key(query, qcur, key)
                        if edited:
                            cursor = 0
        except (termios.error, OSError):
            return self._select_multi_fallback(title, options)

    def history_search(self, history: list[str], message: str = "History:") -> str | None:
        """Interactive history search with type-to-filter (Ctrl+R style)."""
        if not self._is_tty:
            return self._select_fallback(message, history[-10:]) if history else None
        return self._history_search_raw(history, message)

    def _history_search_raw(self, history: list[str], message: str) -> str | None:
        query = ""
        qcur = 0  # caret into `query`, starts at the end
        idx = 0

        def _filtered() -> list[str]:
            if not query:
                return list(reversed(history[-50:]))
            q = query.lower()
            return [h for h in reversed(history) if q in h.lower()][:50]

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            flt = _filtered()
            lines = []
            for i, cmd in enumerate(flt):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                display = cmd[:60] + "..." if len(cmd) > 60 else cmd
                lines.append(f"  {prefix}{color}{display}{_RESET}")
            if not lines:
                lines = [f"  {_DIM}(no matches){_RESET}"]
            query_display = (
                f"  {_BOLD}Search:{_RESET} {self._filter_view(query, qcur, _CYAN)}{_RESET}"
                if query
                else f"  {_DIM}Type to search history{_RESET}"
            )
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows, type=search, enter=select){_RESET}\n"
                + query_display
                + "\n"
                + "\n".join(lines)
                + f"{_HIDE_CURSOR}"
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
                    flt = _filtered()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return None
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return flt[idx] if flt else None
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % max(len(flt), 1)
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % max(len(flt), 1)
                    else:
                        # One editor owns every caret key. `edited` is
                        # False when only the caret moved, so arrows
                        # reposition without re-running the site's filter work.
                        query, qcur, edited = self._filter_key(query, qcur, key)
                        if edited:
                            idx = 0
        except (termios.error, OSError):
            flt = _filtered()
            return flt[0] if flt else None

    def process_manager(
        self, processes: list[dict[str, str]], message: str = "Processes:"
    ) -> dict[str, str] | None:
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

    def _process_manager_raw(
        self, processes: list[dict[str, str]], message: str
    ) -> dict[str, str] | None:
        idx = 0

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            lines = []
            for i, proc in enumerate(processes):
                name = proc.get("name", "?")
                status = proc.get("status", "?")
                status_color = (
                    _GREEN if status == "running" else _YELLOW if status == "pending" else _RED
                )
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                reset = _RESET if i == idx else ""
                lines.append(f"  {prefix}{color}{name:<30} {status_color}{status}{_RESET}{reset}")
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows, enter=select){_RESET}\n"
                + "\n".join(lines)
                + f"{_HIDE_CURSOR}"
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
                        return None
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return processes[idx]
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % len(processes)
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % len(processes)
        except (termios.error, OSError):
            return processes[idx] if processes else None

    def log_viewer(self, logs: list[str], message: str = "Logs:") -> str | None:
        """Interactive log viewer with scroll and filter."""
        if not self._is_tty:
            return logs[-1] if logs else None
        return self._log_viewer_raw(logs, message)

    def _log_viewer_raw(self, logs: list[str], message: str) -> str | None:
        query = ""
        qcur = 0  # caret into `query`, starts at the end
        scroll = 0
        max_visible = 20

        def _filtered() -> list[str]:
            if not query:
                return logs
            q = query.lower()
            return [l for l in logs if q in l.lower()]

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            flt = _filtered()
            visible = flt[scroll : scroll + max_visible]
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
            query_display = (
                f"  {_BOLD}Filter:{_RESET} {self._filter_view(query, qcur, _CYAN)}{_RESET}"
                if query
                else f"  {_DIM}Type to filter{_RESET}"
            )
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows=scroll, type=filter){_RESET}\n"
                + query_display
                + "\n"
                + "\n".join(lines)
                + f"\n{pos}{_HIDE_CURSOR}"
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
                    flt = _filtered()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return None
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return flt[scroll] if flt else None
                    if key.matches(_KEY_UP):
                        scroll = max(0, scroll - 1)
                    elif key.matches(_KEY_DOWN):
                        scroll = min(max(0, len(flt) - max_visible), scroll + 1)
                    elif key.kind == _KEY_PAGE_UP:
                        scroll = max(0, scroll - max_visible)
                    elif key.kind == _KEY_PAGE_DOWN:
                        scroll = min(max(0, len(flt) - max_visible), scroll + max_visible)
                    else:
                        # One editor owns every caret key. `edited` is
                        # False when only the caret moved, so arrows
                        # reposition without re-running the site's filter work.
                        query, qcur, edited = self._filter_key(query, qcur, key)
                        if edited:
                            scroll = 0
        except (termios.error, OSError):
            return logs[-1] if logs else None

    def config_editor(
        self, config: dict[str, str | int | float | bool], message: str = "Config:"
    ) -> dict[str, str | int | float | bool]:
        """Interactive config editor for key-value pairs."""
        if not self._is_tty:
            return config
        return self._config_editor_raw(config, message)

    def _config_editor_raw(
        self, config: dict[str, str | int | float | bool], message: str
    ) -> dict[str, str | int | float | bool]:
        keys = list(config.keys())
        idx = 0
        editing = False
        edit_buf = ""
        edit_cur = 0  # caret into edit_buf, at the end like any fresh input

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            lines = []
            for i, key in enumerate(keys):
                val = config[key]
                val_str = str(val)
                if editing and i == idx:
                    # The line is already in reverse video, so the bar is
                    # just the glyph — `""` is how `_filter_view` is told that.
                    display = self._filter_view(edit_buf, edit_cur, "")
                    lines.append(f"  {_REVERSE}{_CYAN}>> {key} = {display}{_RESET_REVERSE}{_RESET}")
                else:
                    prefix = ">> " if i == idx else "   "
                    color = _CYAN if i == idx else ""
                    lines.append(f"  {prefix}{color}{key} = {val_str}{_RESET}")
            help_text = (
                f"  {_DIM}(arrows=navigate, enter=edit, esc=save){_RESET}"
                if not editing
                else f"  {_DIM}(type=value, enter=confirm, esc=cancel){_RESET}"
            )
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                + "\n".join(lines)
                + f"\n{help_text}{_HIDE_CURSOR}"
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
                    if editing:
                        if key.kind == _KEY_ENTER:
                            val = config[keys[idx]]
                            if isinstance(val, bool):
                                config[keys[idx]] = edit_buf.lower() in (
                                    "true",
                                    "1",
                                    "yes",
                                )
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
                        else:
                            # Motion, rubout, word jumps, delete — everything
                            # a text field does that is neither commit nor
                            # cancel. Keys the dispatch does not claim come
                            # back unchanged, so nothing above is swallowed.
                            edit_buf, edit_cur, _ = self._filter_key(edit_buf, edit_cur, key)
                    else:
                        if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                            _clear()
                            return config
                        if key.kind == _KEY_ENTER:
                            editing = True
                            edit_buf = str(config[keys[idx]])
                            edit_cur = len(edit_buf)
                        if key.matches(_KEY_UP):
                            idx = (idx - 1) % len(keys)
                        elif key.matches(_KEY_DOWN):
                            idx = (idx + 1) % len(keys)
        except (termios.error, OSError):
            return config

    def diff_viewer(self, old: str, new: str, message: str = "Diff:") -> str:
        """Interactive side-by-side diff viewer."""
        if not self._is_tty:
            return new
        return self._diff_viewer_raw(old, new, message)

    def _diff_viewer_raw(self, old: str, new: str, message: str) -> str:
        old_lines = old.splitlines()
        new_lines = new.splitlines()
        scroll = 0
        max_visible = 20

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            visible_old = old_lines[scroll : scroll + max_visible]
            visible_new = new_lines[scroll : scroll + max_visible]
            lines = []
            max_len = max(len(visible_old), len(visible_new))
            for i in range(max_len):
                old_line = visible_old[i] if i < len(visible_old) else ""
                new_line = visible_new[i] if i < len(visible_new) else ""
                old_display = old_line[:35] + "..." if len(old_line) > 35 else old_line
                new_display = new_line[:35] + "..." if len(new_line) > 35 else new_line
                if old_line != new_line:
                    lines.append(f"  {_RED}{old_display:<38}{_RESET} {_GREEN}{new_display}{_RESET}")
                else:
                    lines.append(f"  {_DIM}{old_display:<38}{_RESET} {_DIM}{new_display}{_RESET}")
            if not lines:
                lines = [f"  {_DIM}(no differences){_RESET}"]
            pos = f"  {_DIM}{scroll + 1}-{min(scroll + max_visible, max(len(old_lines), len(new_lines)))}/{max(len(old_lines), len(new_lines))}{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows=scroll){_RESET}\n"
                + "\n".join(lines)
                + f"\n{pos}{_HIDE_CURSOR}"
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
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC, _KEY_ENTER):
                        _clear()
                        return new
                    if key.matches(_KEY_UP):
                        scroll = max(0, scroll - 1)
                    elif key.matches(_KEY_DOWN):
                        max_scroll = max(0, max(len(old_lines), len(new_lines)) - max_visible)
                        scroll = min(max_scroll, scroll + 1)
                    elif key.kind == _KEY_PAGE_UP:
                        scroll = max(0, scroll - max_visible)
                    elif key.kind == _KEY_PAGE_DOWN:
                        max_scroll = max(0, max(len(old_lines), len(new_lines)) - max_visible)
                        scroll = min(max_scroll, scroll + max_visible)
        except (termios.error, OSError):
            return new

    def interactive_search(
        self,
        items: list[str],
        preview_fn: Callable[[str], str] | None = None,
        message: str = "Search:",
    ) -> str | None:
        """Interactive search with type-to-filter and optional preview."""
        if not self._is_tty:
            return self._select_fallback(message, items[:10]) if items else None
        return self._interactive_search_raw(items, preview_fn, message)

    def _interactive_search_raw(
        self, items: list[str], preview_fn: Callable[[str], str] | None, message: str
    ) -> str | None:
        query = ""
        qcur = 0  # caret into `query`, starts at the end
        idx = 0

        def _filtered() -> list[str]:
            if not query:
                return items
            q = query.lower()
            return [i for i in items if q in i.lower()]

        prev_rows = 0  # rows the frame on screen occupies

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
                display = item[:50] + "..." if len(item) > 50 else item
                lines.append(f"  {prefix}{color}{display}{_RESET}")
            if not lines:
                lines = [f"  {_DIM}(no matches){_RESET}"]
            count = f"  {_DIM}{len(flt)}/{len(items)} results{_RESET}"
            query_display = (
                f"  {_BOLD}Search:{_RESET} {self._filter_view(query, qcur, _CYAN)}{_RESET}"
                if query
                else f"  {_DIM}Type to search{_RESET}"
            )
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows, type=search){_RESET}\n"
                + query_display
                + "\n"
                + "\n".join(lines)
                + f"\n{count}\n{preview_str}{_HIDE_CURSOR}"
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
                    flt = _filtered()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return None
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return flt[idx] if flt else None
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % max(len(flt), 1)
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % max(len(flt), 1)
                    else:
                        # One editor owns every caret key. `edited` is
                        # False when only the caret moved, so arrows
                        # reposition without re-running the site's filter work.
                        query, qcur, edited = self._filter_key(query, qcur, key)
                        if edited:
                            idx = 0
        except (termios.error, OSError):
            flt = _filtered()
            return flt[0] if flt else None

    def wizard(
        self, steps: list[dict[str, str | list[str] | None]], message: str = "Wizard"
    ) -> dict[str, str]:
        """Multi-step wizard with labeled steps.

        Each step is a dict with:
            'label': step label
            'type': 'input' | 'select' | 'confirm'
            'options': list of options (for select type)
            'default': default value
        """
        if not self._is_tty:
            return {
                s.get("label", f"step{i}"): str(s.get("default", "")) for i, s in enumerate(steps)
            }
        return self._wizard_raw(steps, message)

    def _wizard_raw(
        self, steps: list[dict[str, str | list[str] | None]], message: str
    ) -> dict[str, str]:
        results: dict[str, str] = {}
        current = 0

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            lines = []
            for i, step in enumerate(steps):
                label = str(step.get("label", f"Step {i + 1}"))
                step_type = str(step.get("type", "input"))
                if i < current:
                    lines.append(f"  {_GREEN}\u2713{_RESET} {label}")
                elif i == current:
                    lines.append(
                        f"  {_CYAN}\u25b6{_RESET} {_BOLD}{label}{_RESET}  {_DIM}({step_type}){_RESET}"
                    )
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
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while current < len(steps):
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    step = steps[current]
                    label = str(step.get("label", f"step{current}"))
                    step_type = str(step.get("type", "input"))
                    default = str(step.get("default", ""))
                    if step_type == "confirm":
                        result = self._confirm_fallback(
                            f"  {label}?", default.lower() in ("true", "1", "yes")
                        )
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

    def form(
        self,
        fields: list[dict[str, str | list[str] | None | bool]],
        message: str = "Form",
    ) -> dict[str, str]:
        """Multi-field form with validation.

        Each field is a dict with:
            'label': field label
            'type': 'text' | 'select' | 'toggle' | 'password'
            'options': list of options (for select type)
            'default': default value
            'required': bool
        """
        if not self._is_tty:
            return {
                f.get("label", f"field{i}"): str(f.get("default", "")) for i, f in enumerate(fields)
            }
        return self._form_raw(fields, message)

    def _form_raw(
        self, fields: list[dict[str, str | list[str] | None | bool]], message: str
    ) -> dict[str, str]:
        results: dict[str, str] = {}
        idx = 0
        editing = False
        edit_buf = ""
        edit_cur = 0  # caret into edit_buf, at the end like any fresh input

        prev_rows = 0  # rows the frame on screen occupies

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
                    # A password is masked one `*` per character, so the caret
                    # keeps the same index in mask space as in the buffer it
                    # really edits. The empty style tells `_filter_view` this
                    # line is already running in reverse video and the bar
                    # wants the glyph and nothing else.
                    source = "*" * len(edit_buf) if field_type == "password" else edit_buf
                    display = self._filter_view(source, edit_cur, "")
                    lines.append(
                        f"  {_REVERSE}{_CYAN}{label}{req_str}: {display}{_RESET_REVERSE}{_RESET}"
                    )
                elif i == idx:
                    lines.append(
                        f"  {_REVERSE}{_CYAN}{label}{req_str}: {value}{_RESET_REVERSE}{_RESET}"
                    )
                else:
                    lines.append(f"  {label}{req_str}: {value}")
            help_text = (
                f"  {_DIM}(arrows=navigate, enter=edit, esc=submit){_RESET}"
                if not editing
                else f"  {_DIM}(type=value, enter=confirm, esc=cancel){_RESET}"
            )
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                + "\n".join(lines)
                + f"\n{help_text}{_HIDE_CURSOR}"
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
                        else:
                            # Motion, rubout, word jumps, delete — everything
                            # a text field does that is neither commit nor
                            # cancel. Keys the dispatch does not claim come
                            # back unchanged, so nothing above is swallowed.
                            edit_buf, edit_cur, _ = self._filter_key(edit_buf, edit_cur, key)
                    else:
                        if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                            _clear()
                            return results
                        if key.kind == _KEY_ENTER:
                            editing = True
                            edit_buf = results.get(label, default)
                            edit_cur = len(edit_buf)
                        if key.matches(_KEY_UP):
                            idx = (idx - 1) % len(fields)
                        elif key.matches(_KEY_DOWN):
                            idx = (idx + 1) % len(fields)
        except (termios.error, OSError):
            return results

    def playlist_manager(self, items: list[str], message: str = "Playlist:") -> list[str]:
        """Manage ordered list with move up/down and delete."""
        if not self._is_tty:
            return items
        return self._playlist_manager_raw(items, message)

    def _playlist_manager_raw(self, items: list[str], message: str) -> list[str]:
        idx = 0

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            lines = []
            for i, item in enumerate(items):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
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
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC, _KEY_ENTER):
                        _clear()
                        return items
                    if key.matches(_KEY_UP) or (key.kind == _KEY_CHAR and key.char == "k"):
                        idx = (idx - 1) % max(len(items), 1)
                    elif key.matches(_KEY_DOWN) or (key.kind == _KEY_CHAR and key.char == "j"):
                        idx = (idx + 1) % max(len(items), 1)
                    elif key.kind == _KEY_CHAR and key.char == "u":
                        if idx > 0:
                            items[idx], items[idx - 1] = items[idx - 1], items[idx]
                            idx -= 1
                    elif key.kind == _KEY_CHAR and key.char == "n":
                        if idx < len(items) - 1:
                            items[idx], items[idx + 1] = items[idx + 1], items[idx]
                            idx += 1
                    elif key.kind == _KEY_CHAR and key.char == "d" or key.kind == _KEY_DELETE:
                        if items:
                            items.pop(idx)
                            idx = idx % max(len(items), 1)
        except (termios.error, OSError):
            return items
