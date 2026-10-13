"""The select_* family. — InteractivePrompt mixin.

Every select_* variant: tree, search, filter, icons, pagination, grouping, tags, countdown, the table-backed forms and the multi-select combinators, all layered on the core select.
"""

from __future__ import annotations

import sys
import termios
import time
from collections.abc import Callable

from domain.shell._internal.interactive_base import InteractiveBase
from domain.shell._internal.interactive_confirm import _ConfirmPrompts
from domain.shell._internal.interactive_term import (  # noqa: F401
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
    _KEY_PAGE_DOWN,
    _KEY_PAGE_UP,
    _KEY_TAB,
    _KEY_UP,
    _RED,
    _RESET,
    _RESET_REVERSE,
    _REVERSE,
    _SHOW_CURSOR,
    _YELLOW,
    _rewind,
    _terminal_width,
    _truncate,
)


class _SelectPrompts(_ConfirmPrompts, InteractiveBase):
    """The select_* family..

    Split out of InteractivePrompt by .splitwork/split_interactive.py.
    """

    def select_tree(
        self,
        title: str,
        tree: dict[str, list[str] | dict],
        expanded: set[str] | None = None,
    ) -> str | None:
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

    def _select_tree_raw(
        self, title: str, tree: dict[str, list[str] | dict], expanded: set[str]
    ) -> str | None:
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
        qcur = 0  # caret into `query`, starts at the end
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
                lines.append(
                    f"{_ERASE_LINE}\r  Filter: {_DIM}{self._filter_view(query, qcur)}{_RESET}{_HIDE_CURSOR}"
                )
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
                        lines.append(
                            f"{_ERASE_LINE}\r{_REVERSE} {display_text}{_RESET_REVERSE}{_RESET}"
                        )
                    else:
                        lines.append(f"{_ERASE_LINE}\r  {display_text}")

            lines.append(
                f"{_ERASE_LINE}\r  {_DIM}Enter: select/expand  Esc: cancel  Type to filter{_RESET}{_HIDE_CURSOR}"
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
                    if key.matches(_KEY_UP):
                        cursor = max(0, cursor - 1)
                    elif key.matches(_KEY_DOWN):
                        cursor = min(len(filtered) - 1, cursor + 1)
                    else:
                        # One editor owns every caret key. `edited` is
                        # False when only the caret moved, so arrows
                        # reposition without re-running the site's filter work.
                        query, qcur, edited = self._filter_key(query, qcur, key)
                        if edited:
                            cursor = 0
        except (termios.error, OSError):
            leaves = self._tree_leaves(tree)
            return leaves[0] if leaves else None

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
        query = ""
        qcur = 0  # caret into `query`, starts at the end
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
            lines.append(
                f"{_ERASE_LINE}\r  {_BOLD}\u26b2 Search:{_RESET} {_BOLD}{self._filter_view(query, qcur, _RESET)}{_HIDE_CURSOR}"
            )
            if not filtered:
                lines.append(f"{_ERASE_LINE}\r  {_DIM}No matching options{_RESET}")
            else:
                for i in range(min(max_visible, len(filtered))):
                    idx = scroll + i
                    if idx >= len(filtered):
                        break
                    text = _truncate(filtered[idx], 50)
                    if idx == cursor:
                        lines.append(f"{_ERASE_LINE}\r{_REVERSE} > {text}{_RESET_REVERSE}{_RESET}")
                    else:
                        lines.append(f"{_ERASE_LINE}\r   {text}")
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
            _rewind(line_count)
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
                        return filtered[cursor] if 0 <= cursor < len(filtered) else options[0]
                    if key.matches(_KEY_UP):
                        cursor = max(0, cursor - 1)
                        scroll = min(scroll, cursor)
                    elif key.matches(_KEY_DOWN):
                        cursor = min(len(filtered) - 1, cursor + 1)
                        if cursor >= scroll + max_visible:
                            scroll = cursor - max_visible + 1
                    else:
                        # One editor owns every caret key. `edited` is
                        # False when only the caret moved, so arrows
                        # reposition without re-running the site's filter work.
                        query, qcur, edited = self._filter_key(query, qcur, key)
                        if edited:
                            cursor = 0
                            scroll = 0
        except (termios.error, OSError):
            return self._select_fallback(title, options)

    def select_with_filter(self, message: str, options: list[str], default: str = "") -> str:
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

    def _select_with_filter_raw(self, message: str, options: list[str], default: str) -> str:
        idx = 0
        query = ""
        qcur = 0  # caret into `query`, starts at the end
        if default in options:
            idx = options.index(default)

        def _filtered() -> list[int]:
            if not query:
                return list(range(len(options)))
            return [i for i, o in enumerate(options) if query.lower() in o.lower()]

        prev_rows = 0  # rows the frame on screen occupies

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
            filter_str = (
                f"  {_DIM}filter: {self._filter_view(query, qcur)}{_RESET}" if query else ""
            )
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(type to filter, arrows){_RESET}\n"
                f"{filter_str}\n" + "\n".join(vis) + f"{_HIDE_CURSOR}"
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
                        return options[idx]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return options[idx]
                    if key.matches(_KEY_UP):
                        filtered = _filtered()
                        if filtered:
                            cur = filtered.index(idx) if idx in filtered else 0
                            idx = filtered[(cur - 1) % len(filtered)]
                    elif key.matches(_KEY_DOWN):
                        filtered = _filtered()
                        if filtered:
                            cur = filtered.index(idx) if idx in filtered else 0
                            idx = filtered[(cur + 1) % len(filtered)]
                    else:
                        # One editor owns every caret key. `edited` is
                        # False when only the caret moved, so arrows
                        # reposition without re-running the site's filter work.
                        query, qcur, edited = self._filter_key(query, qcur, key)
                        if edited:
                            filtered = _filtered()
                            if filtered and idx not in filtered:
                                idx = filtered[0]
        except (termios.error, OSError):
            return options[idx] if idx < len(options) else default

    def select_with_icons(
        self, message: str, options: list[tuple[str, str]], default: str = ""
    ) -> str:
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

    def _select_with_icons_raw(
        self, message: str, options: list[tuple[str, str]], default: str
    ) -> str:
        labels = [o[1] for o in options]
        icons = [o[0] for o in options]
        idx = 0
        if default in labels:
            idx = labels.index(default)

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            lines = []
            for i, (icon, label) in enumerate(zip(icons, labels, strict=False)):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                lines.append(f"  {prefix}{color}{icon} {label}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows){_RESET}\n"
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
                        return labels[idx]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return labels[idx]
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % len(options)
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % len(options)
        except (termios.error, OSError):
            return labels[idx]

    def select_with_pagination(
        self, message: str, options: list[str], page_size: int = 10, default: str = ""
    ) -> str:
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

    def _select_with_pagination_raw(
        self, message: str, options: list[str], page_size: int, default: str
    ) -> str:
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

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            items = _page_items()
            lines = []
            for i, label in items:
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                lines.append(f"  {prefix}{color}{label}{_RESET}")
            page_info = f"  {_DIM}page {page + 1}/{_total_pages()} (PgUp/PgDn){_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows, enter){_RESET}\n"
                + "\n".join(lines)
                + f"\n{page_info}{_HIDE_CURSOR}"
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
                        return options[idx]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return options[idx]
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % len(options)
                        page = idx // page_size
                    elif key.matches(_KEY_DOWN):
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

    def select_with_search_and_preview(
        self, message: str, options: list[str], preview_fn: Callable[[str], str]
    ) -> str:
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

    def _select_with_search_and_preview_raw(
        self, message: str, options: list[str], preview_fn: Callable[[str], str]
    ) -> str:
        query = ""
        qcur = 0  # caret into `query`, starts at the end
        idx = 0

        def _filtered() -> list[int]:
            if not query:
                return list(range(len(options)))
            return [i for i, o in enumerate(options) if query.lower() in o.lower()]

        prev_rows = 0  # rows the frame on screen occupies

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
            filter_str = (
                f"  {_DIM}filter: {self._filter_view(query, qcur)}{_RESET}" if query else ""
            )
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(type to filter, arrows){_RESET}\n"
                f"{filter_str}\n" + "\n".join(vis) + f"\n{preview_str}{_HIDE_CURSOR}"
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
                        return options[idx]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return options[idx]
                    if key.matches(_KEY_UP):
                        filtered = _filtered()
                        if filtered:
                            cur = filtered.index(idx) if idx in filtered else 0
                            idx = filtered[(cur - 1) % len(filtered)]
                    elif key.matches(_KEY_DOWN):
                        filtered = _filtered()
                        if filtered:
                            cur = filtered.index(idx) if idx in filtered else 0
                            idx = filtered[(cur + 1) % len(filtered)]
                    else:
                        # One editor owns every caret key. `edited` is
                        # False when only the caret moved, so arrows
                        # reposition without re-running the site's filter work.
                        query, qcur, edited = self._filter_key(query, qcur, key)
                        if edited:
                            filtered = _filtered()
                            if filtered and idx not in filtered:
                                idx = filtered[0]
        except (termios.error, OSError):
            return options[idx] if idx < len(options) else options[0]

    def select_with_grouping(
        self, message: str, groups: dict[str, list[str]], default: str = ""
    ) -> str:
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

    def _select_with_grouping_raw(
        self, message: str, groups: dict[str, list[str]], default: str
    ) -> str:
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

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            lines = []
            for i, (group, item) in enumerate(flat):
                if group is None:
                    lines.append(f"  {_BOLD}{_YELLOW}{item}{_RESET}")
                else:
                    prefix = ">> " if i == idx else "   "
                    color = _CYAN if i == idx else ""
                    lines.append(f"  {prefix}{color}{item}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows){_RESET}\n"
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
                        return flat[idx][1]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return flat[idx][1]
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % len(flat)
                        while flat[idx][0] is None and idx > 0:
                            idx -= 1
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % len(flat)
                        while flat[idx][0] is None and idx < len(flat) - 1:
                            idx += 1
        except (termios.error, OSError):
            return flat[idx][1]

    def multi_select_with_preview(
        self, message: str, options: list[str], preview_fn: Callable[[str], str]
    ) -> list[str]:
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

    def _multi_select_with_preview_raw(
        self, message: str, options: list[str], preview_fn: Callable[[str], str]
    ) -> list[str]:
        idx = 0
        selected: set[int] = set()

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            preview_text = preview_fn(options[idx])
            preview_lines = preview_text.split("\n")[:6]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            lines = []
            for i, opt in enumerate(options):
                check = f"{_GREEN}\u2714{_RESET}" if i in selected else " "
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                lines.append(f"  {prefix}{check} {color}{opt}{_RESET}")
            count_str = f"  {_DIM}{len(selected)} selected{_RESET}"
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(space=toggle, enter=confirm){_RESET}\n"
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
                        return [options[i] for i in sorted(selected)]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return [options[i] for i in sorted(selected)]
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % len(options)
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % len(options)
                    elif key.kind == _KEY_CHAR and key.char == " ":
                        if idx in selected:
                            selected.discard(idx)
                        else:
                            selected.add(idx)
        except (termios.error, OSError):
            return [options[i] for i in sorted(selected)]

    def select_with_countdown(
        self, message: str, options: list[str], timeout: int = 10, default: int = 0
    ) -> str:
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

    def _select_with_countdown_raw(
        self, message: str, options: list[str], timeout: int, default: int
    ) -> str:
        idx = default
        remaining = timeout

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            lines = []
            for i, opt in enumerate(options):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                lines.append(f"  {prefix}{color}{opt}{_RESET}")
            countdown = (
                f"  {_YELLOW}auto-select in {remaining}s{_RESET}"
                if remaining > 0
                else f"  {_GREEN}selected!{_RESET}"
            )
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows, enter){_RESET}\n"
                + "\n".join(lines)
                + f"\n{countdown}{_HIDE_CURSOR}"
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
                        return options[idx]
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
                        return options[idx]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return options[idx]
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % len(options)
                        start = time.time()
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % len(options)
                        start = time.time()
        except (termios.error, OSError):
            return options[idx]

    def select_with_preview_and_icons(
        self,
        message: str,
        options: list[tuple[str, str]],
        preview_fn: Callable[[str], str],
    ) -> str:
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

    def _select_with_preview_and_icons_raw(
        self,
        message: str,
        options: list[tuple[str, str]],
        preview_fn: Callable[[str], str],
    ) -> str:
        labels = [o[1] for o in options]
        icons = [o[0] for o in options]
        idx = 0

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            preview_text = preview_fn(labels[idx])
            preview_lines = preview_text.split("\n")[:6]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            lines = []
            for i, (icon, label) in enumerate(zip(icons, labels, strict=False)):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                lines.append(f"  {prefix}{color}{icon} {label}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows){_RESET}\n"
                + "\n".join(lines)
                + f"\n{preview_str}{_HIDE_CURSOR}"
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
                        return labels[idx]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return labels[idx]
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % len(options)
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % len(options)
        except (termios.error, OSError):
            return labels[idx]

    def select_with_confirm(self, message: str, options: list[str], default: str = "") -> str:
        """Select an option and confirm with y/n."""
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_confirm_raw(message, options, default)

    def _select_with_confirm_raw(self, message: str, options: list[str], default: str) -> str:
        idx = 0
        if default in options:
            idx = options.index(default)

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            lines = []
            for i, opt in enumerate(options):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                lines.append(f"  {prefix}{color}{opt}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows, enter to confirm){_RESET}\n"
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
                        return options[idx]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return options[idx]
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % len(options)
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % len(options)
        except (termios.error, OSError):
            return options[idx]

    def select_with_preview_and_confirm(
        self,
        message: str,
        options: list[str],
        preview_fn: Callable[[str], str],
        default: str = "",
    ) -> str:
        """Select with preview and confirm."""
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_preview_and_confirm_raw(message, options, preview_fn, default)

    def _select_with_preview_and_confirm_raw(
        self,
        message: str,
        options: list[str],
        preview_fn: Callable[[str], str],
        default: str,
    ) -> str:
        idx = 0
        if default in options:
            idx = options.index(default)

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            preview_text = preview_fn(options[idx])
            preview_lines = preview_text.split("\n")[:5]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            lines = []
            for i, opt in enumerate(options):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                lines.append(f"  {prefix}{color}{opt}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows, enter=confirm){_RESET}\n"
                + "\n".join(lines)
                + f"\n{preview_str}{_HIDE_CURSOR}"
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
                        return options[idx]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return options[idx]
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % len(options)
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % len(options)
        except (termios.error, OSError):
            return options[idx]

    def select_with_preview_and_countdown(
        self,
        message: str,
        options: list[str],
        preview_fn: Callable[[str], str],
        timeout: int = 10,
        default: str = "",
    ) -> str:
        """Select with preview and auto-confirm countdown."""
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_preview_and_countdown_raw(
            message, options, preview_fn, timeout, default
        )

    def _select_with_preview_and_countdown_raw(
        self,
        message: str,
        options: list[str],
        preview_fn: Callable[[str], str],
        timeout: int,
        default: str,
    ) -> str:
        idx = 0
        if default in options:
            idx = options.index(default)

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            preview_text = preview_fn(options[idx])
            preview_lines = preview_text.split("\n")[:5]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            lines = []
            for i, opt in enumerate(options):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                lines.append(f"  {prefix}{color}{opt}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows, enter=select){_RESET}\n"
                + "\n".join(lines)
                + f"\n{preview_str}{_HIDE_CURSOR}"
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
                        return options[idx]
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
                        return options[idx]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return options[idx]
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % len(options)
                        start = time.time()
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % len(options)
                        start = time.time()
        except (termios.error, OSError):
            return options[idx]

    def multi_select_with_filter(
        self, message: str, options: list[str], default: list[str] | None = None
    ) -> list[str]:
        """Multi-select with type-to-filter."""
        if not self._is_tty:
            return (default or [])[:1]
        return self._multi_select_with_filter_raw(message, options, default or [])

    def _multi_select_with_filter_raw(
        self, message: str, options: list[str], default: list[str]
    ) -> list[str]:
        query = ""
        qcur = 0  # caret into `query`, starts at the end
        idx = 0
        selected = set(default)

        def _filtered() -> list[str]:
            if not query:
                return options
            q = query.lower()
            return [o for o in options if q in o.lower()]

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            flt = _filtered()
            lines = []
            for i, opt in enumerate(flt):
                check = f"{_GREEN}\u2714{_RESET}" if opt in selected else f"{_RED}\u2718{_RESET}"
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                lines.append(f"  {prefix}{check} {color}{opt}{_RESET}")
            if not lines:
                lines = [f"  {_DIM}(no matches){_RESET}"]
            count = f"  {_DIM}{len(selected)}/{len(options)} selected{_RESET}"
            query_display = (
                f"  {_BOLD}Filter:{_RESET} {self._filter_view(query, qcur, _CYAN)}{_RESET}"
                if query
                else f"  {_DIM}Type to filter{_RESET}"
            )
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}  {_DIM}(arrows=move, space=toggle, enter=confirm){_RESET}\n"
                + query_display
                + "\n"
                + "\n".join(lines)
                + f"\n{count}{_HIDE_CURSOR}"
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
                        return sorted(selected)
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return sorted(selected)
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % max(len(flt), 1)
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % max(len(flt), 1)
                    elif key.kind == _KEY_CHAR and key.char == " ":
                        if flt[idx] in selected:
                            selected.discard(flt[idx])
                        else:
                            selected.add(flt[idx])
                    else:
                        # One editor owns every caret key. `edited` is
                        # False when only the caret moved, so arrows
                        # reposition without re-running the site's filter work.
                        query, qcur, edited = self._filter_key(query, qcur, key)
                        if edited:
                            idx = 0
        except (termios.error, OSError):
            return sorted(selected)

    def select_with_filter_and_confirm(
        self, message: str, options: list[str], default: str = ""
    ) -> str:
        """Select with type-to-filter and confirm."""
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_filter_and_confirm_raw(message, options, default)

    def _select_with_filter_and_confirm_raw(
        self, message: str, options: list[str], default: str
    ) -> str:
        query = ""
        qcur = 0  # caret into `query`, starts at the end
        idx = 0
        if default in options:
            idx = options.index(default)

        def _filtered() -> list[str]:
            if not query:
                return options
            q = query.lower()
            return [o for o in options if q in o.lower()]

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            flt = _filtered()
            lines = []
            for i, opt in enumerate(flt):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                lines.append(f"  {prefix}{color}{opt}{_RESET}")
            if not lines:
                lines = [f"  {_DIM}(no matches){_RESET}"]
            query_display = (
                f"  {_BOLD}Filter:{_RESET} {self._filter_view(query, qcur, _CYAN)}{_RESET}"
                if query
                else f"  {_DIM}Type to filter{_RESET}"
            )
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows, type=filter, enter=confirm){_RESET}\n"
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
                        return flt[idx] if flt else options[0]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return flt[idx] if flt else options[0]
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
            return flt[0] if flt else options[0]

    def select_with_filter_and_preview(
        self, message: str, options: list[str], preview_fn: Callable[[str], str]
    ) -> str:
        """Select from options with type-to-filter and live preview."""
        if not self._is_tty:
            return self._select_fallback(message, options)
        return self._select_with_filter_and_preview_raw(message, options, preview_fn)

    def _select_with_filter_and_preview_raw(
        self, message: str, options: list[str], preview_fn: Callable[[str], str]
    ) -> str:
        query = ""
        qcur = 0  # caret into `query`, starts at the end
        idx = 0

        def _filtered() -> list[str]:
            if not query:
                return options
            q = query.lower()
            return [o for o in options if q in o.lower()]

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            flt = _filtered()
            preview_text = preview_fn(flt[idx]) if flt else ""
            preview_lines = preview_text.split("\n")[:5]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            lines = []
            for i, opt in enumerate(flt):
                prefix = ">> " if i == idx else "   "
                color = _CYAN if i == idx else ""
                lines.append(f"  {prefix}{color}{opt}{_RESET}")
            if not lines:
                lines = [f"  {_DIM}(no matches){_RESET}"]
            query_display = (
                f"  {_BOLD}Filter:{_RESET} {self._filter_view(query, qcur, _CYAN)}{_RESET}"
                if query
                else f"  {_DIM}Type to filter{_RESET}"
            )
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows, type=filter){_RESET}\n"
                + query_display
                + "\n"
                + "\n".join(lines)
                + f"\n{preview_str}{_HIDE_CURSOR}"
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
                        return flt[idx] if flt else options[0]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return flt[idx] if flt else options[0]
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
            return flt[0] if flt else options[0]

    def select_with_tags(self, message: str, options: list[str], tags: dict[str, list[str]]) -> str:
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

    def _select_with_tags_raw(
        self, message: str, options: list[str], tags: dict[str, list[str]]
    ) -> str:
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

        prev_rows = 0  # rows the frame on screen occupies

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
                f"{tag_str}\n" + "\n".join(vis) + f"{_HIDE_CURSOR}"
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
                    filtered = _filtered()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return filtered[idx % len(filtered)] if filtered else options[0]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return filtered[idx % len(filtered)] if filtered else options[0]
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % max(len(filtered), 1)
                    elif key.matches(_KEY_DOWN):
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

    def select_with_preview_and_grouping(
        self,
        message: str,
        groups: dict[str, list[str]],
        preview_fn: Callable[[str], str],
    ) -> str:
        """Select from categorized options with group headers and live preview."""
        if not self._is_tty:
            all_options = [o for opts in groups.values() for o in opts]
            return self._select_fallback(message, all_options)
        return self._select_with_preview_and_grouping_raw(message, groups, preview_fn)

    def _select_with_preview_and_grouping_raw(
        self,
        message: str,
        groups: dict[str, list[str]],
        preview_fn: Callable[[str], str],
    ) -> str:
        flat: list[tuple[str | None, str]] = []
        for group_name, items in groups.items():
            flat.append((None, group_name))
            for item in items:
                flat.append((group_name, item))
        idx = 1
        selectable = [i for i, (g, _) in enumerate(flat) if g is not None]

        prev_rows = 0  # rows the frame on screen occupies

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
                    lines.append(f"  {prefix}{color}{item}{_RESET}")
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows){_RESET}\n"
                + "\n".join(lines)
                + f"\n{preview_str}{_HIDE_CURSOR}"
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
                        return flat[idx][1]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return flat[idx][1]
                    if key.matches(_KEY_UP):
                        pos = selectable.index(idx) if idx in selectable else 0
                        idx = selectable[(pos - 1) % len(selectable)]
                    elif key.matches(_KEY_DOWN):
                        pos = selectable.index(idx) if idx in selectable else 0
                        idx = selectable[(pos + 1) % len(selectable)]
        except (termios.error, OSError):
            return flat[idx][1]

    def select_with_details(self, title: str, options: list[str], details: list[str]) -> str:
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

    def _select_with_details_raw(self, title: str, options: list[str], details: list[str]) -> str:
        """Interactive select with details pane using raw terminal input."""
        query = ""
        qcur = 0  # caret into `query`, starts at the end
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
                lines.append(
                    f"{_ERASE_LINE}\r  Filter: {_DIM}{self._filter_view(query, qcur)}{_RESET}{_HIDE_CURSOR}"
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
                    orig_idx, text = filtered[idx]
                    display = _truncate(text, width - 5)
                    if idx == cursor:
                        lines.append(
                            f"{_ERASE_LINE}\r{_REVERSE} > {display}{_RESET_REVERSE}{_RESET}"
                        )
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
            _rewind(line_count)
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
                            return filtered[cursor][1]
                        return options[0]

                    if key.matches(_KEY_UP):
                        cursor = max(0, cursor - 1)
                        scroll = min(scroll, cursor)

                    elif key.matches(_KEY_DOWN):
                        cursor = min(len(filtered) - 1, cursor + 1)
                        if cursor >= scroll + max_visible:
                            scroll = cursor - max_visible + 1

                    else:
                        # One editor owns every caret key. `edited` is
                        # False when only the caret moved, so arrows
                        # reposition without re-running the site's filter work.
                        query, qcur, edited = self._filter_key(query, qcur, key)
                        if edited:
                            cursor = 0
                            scroll = 0
        except (termios.error, OSError):
            return self._select_fallback(title, options)

    def select_with_preview(
        self, title: str, options: list[str], preview_fn: Callable[[str], str]
    ) -> str:
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

    def _select_with_preview_raw(
        self, title: str, options: list[str], preview_fn: Callable[[str], str]
    ) -> str:
        """Interactive select with live preview using raw terminal input."""
        query = ""
        qcur = 0  # caret into `query`, starts at the end
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
                lines.append(
                    f"{_ERASE_LINE}\r  Filter: {_DIM}{self._filter_view(query, qcur)}{_RESET}{_HIDE_CURSOR}"
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
                    for i, pline in enumerate(preview_text.split("\n")[: max_visible + 2]):
                        t = _truncate(pline, preview_w)
                        if i < len(lines):
                            lines[i] = f"{lines[i]}{_DIM}\u2502{_RESET} {t}"
                        else:
                            lines.append(f"{' ' * (list_w + 2)}{_DIM}\u2502{_RESET} {t}")

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
            _rewind(line_count)
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
                        # One editor owns every caret key. `edited` is
                        # False when only the caret moved, so arrows
                        # reposition without re-running the site's filter work.
                        query, qcur, edited = self._filter_key(query, qcur, key)
                        if edited:
                            cursor = 0
                            scroll = 0
        except (termios.error, OSError):
            return self._select_fallback(title, options)
