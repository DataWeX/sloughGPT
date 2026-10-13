"""Table, spreadsheet and view surfaces. — InteractivePrompt mixin.

table, table_select, table_edit, table_sort, table_with_search, table_with_search_and_preview, spreadsheet_editor, kanban_board, calendar_view, diff and diff_viewer.
"""

from __future__ import annotations

import sys
import termios
from collections.abc import Callable

from domain.shell._internal.interactive_base import InteractiveBase
from domain.shell._internal.interactive_picker import _PickerPrompts
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
    _KEY_DELETE,
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
    _dash_row,
    _rewind,
    _terminal_width,
)


class _TablePrompts(_PickerPrompts, InteractiveBase):
    """Table, spreadsheet and view surfaces..

    Split out of InteractivePrompt by .splitwork/split_interactive.py.
    """

    def spreadsheet_editor(
        self, headers: list[str], rows: list[list[str]], message: str = "Spreadsheet:"
    ) -> list[list[str]]:
        """Interactive spreadsheet editor with cell navigation."""
        if not self._is_tty:
            return rows
        return self._spreadsheet_editor_raw(headers, rows, message)

    def _spreadsheet_editor_raw(
        self, headers: list[str], rows: list[list[str]], message: str
    ) -> list[list[str]]:
        col = 0
        row = 0
        editing = False
        edit_buf = ""
        edit_cur = 0  # caret into edit_buf, at the end like any fresh input
        col_widths = [
            max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)
        ]

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            lines = []
            header = "  " + "  ".join(
                f"{_BOLD}{h:<{col_widths[i]}}{_RESET}" for i, h in enumerate(headers)
            )
            lines.append(header)
            for r_idx, r in enumerate(rows):
                cells = []
                for c_idx, c in enumerate(r):
                    display = c[: col_widths[c_idx]]
                    if r_idx == row and c_idx == col and editing:
                        # Already inside a reverse run, so the bar is the bare
                        # glyph. The editing cell has always shown the whole
                        # buffer rather than a width-cut slice — only the
                        # *other* cells use `display` — and that stays true.
                        cells.append(
                            f"{_REVERSE}{_CYAN}"
                            f"{self._filter_view(edit_buf, edit_cur, '')}"
                            f"{_RESET_REVERSE}"
                        )
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
            help_text = (
                f"  {_DIM}(arrows=move, enter=edit, tab=new row, esc=done){_RESET}"
                if not editing
                else f"  {_DIM}(type=value, enter=confirm, esc=cancel){_RESET}"
            )
            return (
                f"{_ERASE_LINE}\r  {_BOLD}{message}{_RESET}\n"
                + "\n".join(lines)
                + f"\n{pos}\n{help_text}{_HIDE_CURSOR}"
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
                            rows[row][col] = edit_buf
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
                            return rows
                        if key.kind == _KEY_ENTER:
                            editing = True
                            edit_buf = rows[row][col]
                            edit_cur = len(edit_buf)
                        if key.matches(_KEY_UP):
                            row = (row - 1) % len(rows)
                        elif key.matches(_KEY_DOWN):
                            row = (row + 1) % len(rows)
                        elif key.matches(_KEY_LEFT):
                            col = (col - 1) % len(headers)
                        elif key.matches(_KEY_RIGHT):
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

    def kanban_board(
        self, columns: dict[str, list[str]], message: str = "Kanban"
    ) -> dict[str, list[str]]:
        """Interactive kanban board with move between columns."""
        if not self._is_tty:
            return columns
        return self._kanban_board_raw(columns, message)

    def _kanban_board_raw(
        self, columns: dict[str, list[str]], message: str
    ) -> dict[str, list[str]]:
        col_idx = 0
        item_idx = 0
        col_names = list(columns.keys())

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            lines = []
            for c, col_name in enumerate(col_names):
                items = columns[col_name]
                col_header = f"  {_BOLD}{col_name}{_RESET}"
                lines.append(col_header)
                for i, item in enumerate(items):
                    prefix = ">>" if c == col_idx and i == item_idx else "  "
                    color = _CYAN if c == col_idx and i == item_idx else ""
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
                        return columns
                    col_items = columns[col_names[col_idx]]
                    if key.matches(_KEY_LEFT) or (key.kind == _KEY_CHAR and key.char == "h"):
                        col_idx = (col_idx - 1) % len(col_names)
                        item_idx = min(item_idx, len(columns[col_names[col_idx]]) - 1)
                        item_idx = max(0, item_idx)
                    elif key.matches(_KEY_RIGHT) or (key.kind == _KEY_CHAR and key.char == "l"):
                        col_idx = (col_idx + 1) % len(col_names)
                        item_idx = min(item_idx, len(columns[col_names[col_idx]]) - 1)
                        item_idx = max(0, item_idx)
                    elif key.matches(_KEY_UP) or (key.kind == _KEY_CHAR and key.char == "k"):
                        item_idx = (item_idx - 1) % max(len(col_items), 1)
                    elif key.matches(_KEY_DOWN) or (key.kind == _KEY_CHAR and key.char == "j"):
                        item_idx = (item_idx + 1) % max(len(col_items), 1)
                    elif key.matches(_KEY_LEFT):
                        if col_idx > 0 and col_items:
                            item = col_items.pop(item_idx)
                            columns[col_names[col_idx - 1]].append(item)
                            item_idx = 0
                            col_idx -= 1
                    elif key.matches(_KEY_RIGHT):
                        if col_idx < len(col_names) - 1 and col_items:
                            item = col_items.pop(item_idx)
                            columns[col_names[col_idx + 1]].append(item)
                            item_idx = 0
                            col_idx += 1
        except (termios.error, OSError):
            return columns

    def calendar_view(
        self,
        year: int,
        month: int,
        events: dict[int, str] | None = None,
        message: str = "Calendar",
    ) -> int | None:
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

    def _calendar_view_raw(
        self, year: int, month: int, events: dict[int, str], message: str
    ) -> int | None:
        import calendar as _cal

        day = 1
        cal = _cal.monthcalendar(year, month)
        month_name = _cal.month_name[month]

        prev_rows = 0  # rows the frame on screen occupies

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
                        return day
                    if key.matches(_KEY_UP):
                        day = max(1, day - 7)
                    elif key.matches(_KEY_DOWN):
                        day = min(31, day + 7)
                    elif key.matches(_KEY_LEFT):
                        day = max(1, day - 1)
                    elif key.matches(_KEY_RIGHT):
                        day = min(31, day + 1)
        except (termios.error, OSError):
            return None

    def table_select(
        self, headers: list[str], rows: list[list[str]], title: str = "Select row"
    ) -> int | None:
        """Interactive table with row selection. Returns selected row index."""
        if not rows:
            return None
        if not self._is_tty:
            return 0
        return self._table_select_raw(headers, rows, title)

    def _table_select_raw(
        self, headers: list[str], rows: list[list[str]], title: str
    ) -> int | None:
        cursor = 0
        scroll = 0
        max_visible = min(len(rows), 12)
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        def _render() -> list[str]:
            lines: list[str] = []
            lines.append(f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}")
            lines.append(
                f"{_ERASE_LINE}\r  {'  '.join(f'{_BOLD}{h.ljust(w)}{_RESET}' for h, w in zip(headers, widths, strict=False))}"
            )
            lines.append(f"{_ERASE_LINE}\r  {_DIM}{_dash_row(widths)}{_RESET}")
            for i in range(min(max_visible, len(rows))):
                idx = scroll + i
                if idx >= len(rows):
                    break
                cells = "  ".join(
                    str(rows[idx][j]).ljust(widths[j])
                    for j in range(min(len(widths), len(rows[idx])))
                )
                if idx == cursor:
                    lines.append(f"{_ERASE_LINE}\r{_REVERSE} > {cells}{_RESET_REVERSE}{_RESET}")
                else:
                    lines.append(f"{_ERASE_LINE}\r   {cells}")
            lines.append(f"{_ERASE_LINE}\r  {_DIM}Enter: select  Esc: cancel{_RESET}{_HIDE_CURSOR}")
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
                        return None
                    if key.kind == _KEY_ENTER:
                        _clear(prev_count)
                        return cursor if 0 <= cursor < len(rows) else None
                    if key.matches(_KEY_UP):
                        cursor = max(0, cursor - 1)
                        scroll = min(scroll, cursor)
                    elif key.matches(_KEY_DOWN):
                        cursor = min(len(rows) - 1, cursor + 1)
                        if cursor >= scroll + max_visible:
                            scroll = cursor - max_visible + 1
        except (termios.error, OSError):
            return 0

    def table_edit(
        self, headers: list[str], rows: list[list[str]], title: str = "Edit table"
    ) -> list[list[str]]:
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

    def _table_edit_raw(
        self, headers: list[str], rows: list[list[str]], title: str
    ) -> list[list[str]]:
        cursor_row = 0
        cursor_col = 0
        editing = False
        buf: list[str] = []
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        def _render() -> list[str]:
            lines: list[str] = []
            lines.append(
                f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}  {_DIM}(Tab: cell  Enter: edit/save  Esc: done){_RESET}"
            )
            lines.append(
                f"{_ERASE_LINE}\r  {'  '.join(f'{_BOLD}{h.ljust(w)}{_RESET}' for h, w in zip(headers, widths, strict=False))}"
            )
            lines.append(f"{_ERASE_LINE}\r  {_DIM}{_dash_row(widths)}{_RESET}")
            for r_idx, row in enumerate(rows):
                cells = []
                for c_idx in range(min(len(widths), len(row))):
                    val = (
                        "".join(buf)
                        if editing and r_idx == cursor_row and c_idx == cursor_col
                        else str(row[c_idx])
                    )
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
                        if editing:
                            editing = False
                            buf.clear()
                        else:
                            _clear(prev_count)
                            return rows
                    elif key.kind == _KEY_ENTER:
                        if editing:
                            rows[cursor_row][cursor_col] = "".join(buf)
                            editing = False
                            buf.clear()
                        else:
                            editing = True
                            buf = list(str(rows[cursor_row][cursor_col]))
                    elif not editing:
                        if key.kind == _KEY_TAB:
                            cursor_col = (cursor_col + 1) % len(widths)
                        elif key.matches(_KEY_UP):
                            cursor_row = max(0, cursor_row - 1)
                        elif key.matches(_KEY_DOWN):
                            cursor_row = min(len(rows) - 1, cursor_row + 1)
                        elif key.matches(_KEY_LEFT):
                            cursor_col = max(0, cursor_col - 1)
                        elif key.matches(_KEY_RIGHT):
                            cursor_col = min(len(widths) - 1, cursor_col + 1)
                    else:
                        if key.kind == _KEY_BACKSPACE:
                            if buf:
                                buf.pop()
                        elif key.kind == _KEY_DELETE:
                            if buf:
                                buf.clear()
                        elif key.kind == _KEY_CHAR:
                            buf.append(key.char)
        except (termios.error, OSError):
            return rows

    def table_with_search(
        self, headers: list[str], rows: list[list[str]], title: str = ""
    ) -> list[list[str]]:
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

    def _table_with_search_raw(
        self, headers: list[str], rows: list[list[str]], title: str
    ) -> list[list[str]]:
        query = ""
        qcur = 0  # caret into `query`, starts at the end
        col_widths = [
            max(len(h), max((len(str(r[i])) for r in rows), default=0))
            for i, h in enumerate(headers)
        ]

        def _filtered() -> list[list[str]]:
            if not query:
                return rows
            q = query.lower()
            return [r for r in rows if any(q in str(c).lower() for c in r)]

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            filtered = _filtered()
            lines = []
            header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
            lines.append(f"  {_BOLD}{header_line}{_RESET}")
            lines.append(f"  {_DIM}{'─' * sum(col_widths + [3 * (len(headers) - 1)])}{_RESET}")
            for r in filtered[:20]:
                line = "  ".join(
                    str(r[i]).ljust(col_widths[i]) if i < len(r) else " " * col_widths[i]
                    for i in range(len(headers))
                )
                lines.append(f"  {line}")
            filter_str = (
                f"  {_DIM}filter: {self._filter_view(query, qcur)} ({len(filtered)} rows){_RESET}"
                if query
                else f"  {_DIM}{len(rows)} rows{_RESET}"
            )
            title_str = f"  {_BOLD}{title}{_RESET}\n" if title else ""
            return (
                f"{_ERASE_LINE}\r{title_str}" + "\n".join(lines) + f"\n{filter_str}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            filtered = _filtered()
            line_count = 2 + min(len(filtered), 20) + 1
            if title:
                line_count += 1
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
                        return _filtered()
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return _filtered()
                    else:
                        # One editor owns every caret key; this box has no
                        # selection to reset when the text changes.
                        query, qcur, _ = self._filter_key(query, qcur, key)
        except (termios.error, OSError):
            return rows

    def select_table_with_preview(
        self,
        headers: list[str],
        rows: list[list[str]],
        preview_fn: Callable[[list[str]], str],
    ) -> list[str]:
        """Select a row from a table with live preview."""
        if not self._is_tty:
            return rows[0] if rows else []
        return self._select_table_with_preview_raw(headers, rows, preview_fn)

    def _select_table_with_preview_raw(
        self,
        headers: list[str],
        rows: list[list[str]],
        preview_fn: Callable[[list[str]], str],
    ) -> list[str]:
        idx = 0

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            col_widths = [
                max(len(h), max((len(r[i]) for r in rows), default=0))
                for i, h in enumerate(headers)
            ]
            preview_text = preview_fn(rows[idx]) if rows else ""
            preview_lines = preview_text.split("\n")[:5]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            header = "  " + "  ".join(
                f"{_BOLD}{h:<{col_widths[i]}}{_RESET}" for i, h in enumerate(headers)
            )
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
                        return rows[idx]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return rows[idx]
                    if key.matches(_KEY_UP):
                        idx = (idx - 1) % len(rows)
                    elif key.matches(_KEY_DOWN):
                        idx = (idx + 1) % len(rows)
        except (termios.error, OSError):
            return rows[idx] if rows else []

    def table_with_search_and_preview(
        self,
        headers: list[str],
        rows: list[list[str]],
        preview_fn: Callable[[list[str]], str],
    ) -> list[str]:
        """Searchable table with live preview."""
        if not self._is_tty:
            return rows[0] if rows else []
        return self._table_with_search_and_preview_raw(headers, rows, preview_fn)

    def _table_with_search_and_preview_raw(
        self,
        headers: list[str],
        rows: list[list[str]],
        preview_fn: Callable[[list[str]], str],
    ) -> list[str]:
        query = ""
        qcur = 0  # caret into `query`, starts at the end
        idx = 0

        def _filtered() -> list[list[str]]:
            if not query:
                return rows
            q = query.lower()
            return [r for r in rows if any(q in c.lower() for c in r)]

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            flt = _filtered()
            col_widths = [
                max(len(h), max((len(r[i]) for r in flt), default=0)) for i, h in enumerate(headers)
            ]
            preview_text = preview_fn(flt[idx]) if flt else ""
            preview_lines = preview_text.split("\n")[:4]
            preview_str = "\n".join(f"  {_DIM}{line}{_RESET}" for line in preview_lines)
            header = "  " + "  ".join(
                f"{_BOLD}{h:<{col_widths[i]}}{_RESET}" for i, h in enumerate(headers)
            )
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
            query_display = (
                f"  {_BOLD}Search:{_RESET} {self._filter_view(query, qcur, _CYAN)}{_RESET}"
                if query
                else f"  {_DIM}Type to search{_RESET}"
            )
            return (
                f"{_ERASE_LINE}\r  {_BOLD}Table{_RESET}  {_DIM}(arrows, type=search){_RESET}\n"
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
                        return flt[idx] if flt else rows[0]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return flt[idx] if flt else rows[0]
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
            return flt[0] if flt else rows[0]

    def table_sort(
        self, headers: list[str], rows: list[list[str]], title: str = "Sort table"
    ) -> list[list[str]]:
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

    def _table_sort_raw(
        self, headers: list[str], rows: list[list[str]], title: str
    ) -> list[list[str]]:
        sort_col = 0
        sort_asc = True
        sorted_rows = list(rows)
        cursor = 0
        scroll = 0
        max_visible = min(len(rows), 12)
        widths = [len(h) for h in headers]
        for row in sorted_rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        def _do_sort() -> None:
            nonlocal sorted_rows, cursor, scroll

            def _key(row):
                val = row[sort_col] if sort_col < len(row) else ""
                try:
                    return (0, float(val))
                except ValueError:
                    return (1, val.lower())

            sorted_rows.sort(key=_key, reverse=not sort_asc)
            cursor = 0
            scroll = 0

        _do_sort()

        def _render() -> list[str]:
            lines: list[str] = []
            lines.append(
                f"{_ERASE_LINE}\r  {_BOLD}{_CYAN}{title}{_RESET}  {_DIM}(Tab: column  Space: asc/desc  Enter: confirm){_RESET}"
            )
            hdr_parts = []
            for i, (h, w) in enumerate(zip(headers, widths, strict=False)):
                arrow = (
                    " \u25b2"
                    if sort_asc and i == sort_col
                    else (" \u25bc" if not sort_asc and i == sort_col else "")
                )
                if i == sort_col:
                    hdr_parts.append(f"{_BOLD}{_CYAN}{h.ljust(w)}{arrow}{_RESET}")
                else:
                    hdr_parts.append(f"{_BOLD}{h.ljust(w)}{_RESET}")
            lines.append(f"{_ERASE_LINE}\r  {'  '.join(hdr_parts)}")
            lines.append(f"{_ERASE_LINE}\r  {_DIM}{_dash_row(widths)}{_RESET}")
            for i in range(min(max_visible, len(sorted_rows))):
                idx = scroll + i
                if idx >= len(sorted_rows):
                    break
                cells = "  ".join(
                    str(sorted_rows[idx][j]).ljust(widths[j])
                    for j in range(min(len(widths), len(sorted_rows[idx])))
                )
                if idx == cursor:
                    lines.append(f"{_ERASE_LINE}\r{_REVERSE} > {cells}{_RESET_REVERSE}{_RESET}")
                else:
                    lines.append(f"{_ERASE_LINE}\r   {cells}")
            lines.append(f"{_ERASE_LINE}\r  {_DIM}{len(sorted_rows)} rows{_RESET}{_HIDE_CURSOR}")
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
                        return sorted_rows
                    if key.kind == _KEY_ENTER:
                        _clear(prev_count)
                        return sorted_rows
                    if key.kind == _KEY_TAB:
                        sort_col = (sort_col + 1) % len(headers)
                        _do_sort()
                    elif key.kind == _KEY_SPACE:
                        sort_asc = not sort_asc
                        _do_sort()
                    elif key.matches(_KEY_UP):
                        cursor = max(0, cursor - 1)
                        scroll = min(scroll, cursor)
                    elif key.matches(_KEY_DOWN):
                        cursor = min(len(sorted_rows) - 1, cursor + 1)
                        if cursor >= scroll + max_visible:
                            scroll = cursor - max_visible + 1
        except (termios.error, OSError):
            return sorted_rows

    def table(self, headers: list[str], rows: list[list[str]], title: str = "") -> None:
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

        header_str = "  ".join(
            f"{_BOLD}{h.ljust(w)}{_RESET}" for h, w in zip(headers, widths, strict=False)
        )
        lines.append(f"  {_BOLD}{header_str}{_RESET}")
        lines.append(f"  {_DIM}{_dash_row(widths)}{_RESET}")

        for idx, row in enumerate(rows):
            color = _DIM if idx % 2 == 1 else ""
            lines.append(f"  {_fmt_row(row, color)}")

        for line in lines:
            self._io.write(line)

    def diff(
        self,
        left_label: str,
        left_lines: list[str],
        right_label: str,
        right_lines: list[str],
        title: str = "",
    ) -> None:
        """Show a side-by-side diff with colored additions/removals.

        Args:
            left_label: label for left pane
            left_lines: content lines of left side
            right_label: label for right pane
            right_lines: content lines of right side
            title: optional title above the diff
        """
        width = _terminal_width()
        # header = "  " + left + "  |  " + right = 2 + col_w + 2 + 1 + 2 +
        # col_w, so it fits only while 2 * col_w + 7 <= width. The old
        # `(width - 5) // 2` put that first row one column past the terminal;
        # `(width - 7) // 2` lands one short, which is the honest side for a
        # floor to round to.
        col_w = max(1, (width - 7) // 2)

        def _trunc(s: str, w: int) -> str:
            if len(s) > w:
                return s[: w - 1] + "\u2026"
            return s

        def _center(s: str) -> str:
            """A header cell, centred in `col_w`.

            Truncated first: `:^{col_w}` widens for an over-long label
            instead of shortening it, and the header is the row that sets
            the width of everything under it.
            """
            return f"{_trunc(s, col_w):^{col_w}}"

        if title:
            self._io.write(f"  {_BOLD}{_CYAN}{title}{_RESET}")

        header = (
            f"  {_DIM}{_center(left_label)}{_RESET}  \u2502  {_DIM}{_center(right_label)}{_RESET}"
        )
        self._io.write(header)
        # One dash per column of each half, and the same two-space gutters
        # the header uses, so the separator is exactly as wide as the header
        # above it and its cross sits on the header's divider.
        rule = "\u2500" * col_w
        sep = f"  {rule}  \u253c  {rule}"
        self._io.write(f"{_DIM}{sep}{_RESET}")

        max_len = max(len(left_lines), len(right_lines))
        left_set = set(left_lines)
        right_set = set(right_lines)

        for i in range(max_len):
            l = left_lines[i] if i < len(left_lines) else ""
            r = right_lines[i] if i < len(right_lines) else ""

            # Pad before colouring. Padding `f"{cell:<N}"` on a cell that
            # already carries ESC[31m/ESC[0m measures those nine bytes as
            # content, so the coloured row came back nine visible columns
            # short and its divider sat somewhere else from the plain row
            # above it. The gutter was ten wide here against two in the
            # header, which put this divider ten columns right of that one.
            left_cell = f"{_trunc(l, col_w):<{col_w}}"
            right_cell = _trunc(r, col_w)

            if l and l not in right_set:
                l_display = f"{_RED}{left_cell}{_RESET}"
            else:
                l_display = left_cell

            if r and r not in left_set:
                r_display = f"{_GREEN}{right_cell}{_RESET}"
            else:
                r_display = right_cell

            self._io.write(f"  {l_display}  \u2502  {r_display}")
