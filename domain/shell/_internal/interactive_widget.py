"""Single-purpose widgets. — InteractivePrompt mixin.

Compact widgets with no nested prompt: countdown, slider, toggle, tag_input, hierarchical_menu, multi_confirm, spin_until, spin_wait.
"""

from __future__ import annotations

import sys
import termios
import time
from collections.abc import Callable
from typing import Any

from domain.shell._internal.interactive_base import InteractiveBase
from domain.shell._internal.interactive_prompt import _PromptPrompts
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
    _rewind,
)


class _WidgetPrompts(_PromptPrompts, InteractiveBase):
    """Single-purpose widgets..

    Split out of InteractivePrompt by .splitwork/split_interactive.py.
    """

    def countdown(self, seconds: int, message: str = "Starting in") -> bool:
        """Show a visual countdown timer. Returns True if completed."""
        if not self._is_tty:
            for i in range(seconds, 0, -1):
                self._io.write(f"  {message} {i}...")
            return True
        return self._countdown_raw(seconds, message)

    def _countdown_raw(self, seconds: int, message: str) -> bool:
        remaining = seconds
        try:
            with self._session.raw():
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

    def slider(
        self,
        message: str,
        min_val: int = 0,
        max_val: int = 100,
        default: int = 50,
        step: int = 1,
    ) -> int:
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

    def _slider_raw(self, message: str, min_val: int, max_val: int, default: int, step: int) -> int:
        value = default
        bar_w = 30

        prev_rows = 0  # rows the frame on screen occupies

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
                        return value
                    if key.matches(_KEY_LEFT):
                        value = max(min_val, value - step)
                    elif key.matches(_KEY_RIGHT):
                        value = min(max_val, value + step)
                    elif key.kind == _KEY_HOME:
                        value = min_val
                    elif key.kind == _KEY_END:
                        value = max_val
        except (termios.error, OSError):
            return default

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
        on = default

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            if on:
                on_text = f"{_REVERSE}{_GREEN} ON {_RESET_REVERSE}{_RESET}"
                off_text = "OFF "
            else:
                on_text = " ON "
                off_text = f"{_REVERSE}{_RED}OFF{_RESET_REVERSE}{_RESET}"
            return f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {on_text}  {off_text}{_HIDE_CURSOR}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
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
                        return on
                    if key.matches(_KEY_LEFT, _KEY_RIGHT):
                        on = not on
                    if key.kind == _KEY_CHAR:
                        if key.char in ("1", "t", "T"):
                            on = True
                        elif key.char in ("0", "f", "F"):
                            on = False
        except (termios.error, OSError):
            return default

    def tag_input(
        self,
        message: str,
        defaults: list[str] | None = None,
        placeholder: str = "Add tag...",
    ) -> list[str]:
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

    def _tag_input_raw(self, message: str, defaults: list[str], placeholder: str) -> list[str]:
        tags: list[str] = list(defaults)
        buf: list[str] = []

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            tag_str = ""
            for t in tags:
                tag_str += f" {_GREEN}\u25cf {t}{_RESET}\u2500"
            if not tags and not buf:
                tag_str = f" {_DIM}{placeholder}{_RESET}"
            input_part = "".join(buf)
            return f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}{tag_str} {input_part}{_HIDE_CURSOR}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
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
                        return tags
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
                        _clear()
                        return tags
        except (termios.error, OSError):
            return tags

    def spin_wait(
        self,
        message: str,
        check_fn: Callable[[], bool],
        interval: float = 0.1,
        timeout: float = 0,
    ) -> bool:
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

    def _spin_wait_raw(
        self,
        message: str,
        check_fn: Callable[[], bool],
        interval: float,
        timeout: float,
    ) -> bool:
        import time as _t

        frames = ["\u25cf", "\u25cf\u25cf", "\u25cf\u25cf\u25cf", "\u25cf"]
        idx = 0
        start = _t.monotonic()
        try:
            with self._session.raw():
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

    def hierarchical_menu(
        self, menu: dict[str, str | list[str] | dict], message: str = "Menu:"
    ) -> str | None:
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

    def _hierarchical_menu_raw(
        self, menu: dict[str, str | list[str] | dict], message: str
    ) -> str | None:
        path: list[dict] = [menu]
        idx = 0

        def _current() -> dict:
            return path[-1]

        def _items() -> list[str]:
            return list(_current().keys())

        prev_rows = 0  # rows the frame on screen occupies

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
                lines.append(f"  {prefix}{color}{icon} {item}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows, enter=open, backspace=back){_RESET}\n"
                + breadcrumb_str
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
                    items = _items()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return None
                    if key.kind == _KEY_ENTER:
                        val = _current()[items[idx]]
                        if isinstance(val, dict):
                            path.append(val)
                            idx = 0
                        else:
                            _clear()
                            return str(val)
                    if key.kind == _KEY_BACKSPACE:
                        if len(path) > 1:
                            path.pop()
                            idx = 0
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % len(items)
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % len(items)
        except (termios.error, OSError):
            return None

    def spin_until(
        self,
        message: str,
        async_fn: Callable[[], Any],
        check: Callable[[Any], bool],
        interval: float = 0.1,
        timeout: float = 0,
    ) -> Any:
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

    def _spin_until_raw(
        self,
        message: str,
        async_fn: Callable[[], Any],
        check: Callable[[Any], bool],
        interval: float,
        timeout: float,
    ) -> Any:
        import time as _t

        frames = ["\u25cf", "\u25cf\u25cf", "\u25cf\u25cf\u25cf", "\u25cf"]
        idx = 0
        start = _t.monotonic()
        try:
            with self._session.raw():
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

    def multi_confirm(
        self, message: str, items: list[str], default: bool = True
    ) -> dict[str, bool]:
        """Confirm multiple items with toggle.

        Args:
            message: prompt text
            items: list of items to confirm
            default: default state for all items

        Returns:
            dict of item → confirmed status
        """
        if not self._is_tty:
            return dict.fromkeys(items, default)
        return self._multi_confirm_raw(message, items, default)

    def _multi_confirm_raw(self, message: str, items: list[str], default: bool) -> dict[str, bool]:
        idx = 0
        states = dict.fromkeys(items, default)

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            lines = []
            for i, item in enumerate(items):
                check = f"{_GREEN}\u2714{_RESET}" if states[item] else f"{_RED}\u2718{_RESET}"
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                lines.append(f"  {prefix}{check} {color}{item}{_RESET}")
            count = sum(1 for v in states.values() if v)
            count_str = f"  {_DIM}{count}/{len(items)} confirmed{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(space=toggle, enter=confirm){_RESET}\n"
                + "\n".join(lines)
                + f"\n{count_str}{_HIDE_CURSOR}"
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
                        return states
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return states
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % len(items)
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % len(items)
                    elif key.kind == _KEY_CHAR and key.char == " ":
                        states[items[idx]] = not states[items[idx]]
        except (termios.error, OSError):
            return states
