"""
kanban_engine — standalone terminal rendering engine for kanban boards.

Own rendering pipeline. Reads board.jsonl, renders with ANSI,
supports interactive navigation, card details, search, trash.

Usage::

    from planner.engine import KanbanEngine
    engine = KanbanEngine.from_file("board.jsonl")
    engine.render()          # static render
    engine.interactive()     # interactive TUI

CLI::

    planner render              # static render
    planner render --interactive
    planner render --column todo --search "bug"
"""

from __future__ import annotations

import json
import os
import sys
import tty
import termios
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ── ANSI ───────────────────────────────────────────────────────────────

class ANSI:
    reset      = "\033[0m"
    bold       = "\033[1m"
    dim        = "\033[2m"
    italic     = "\033[3m"
    underline  = "\033[4m"
    inverse    = "\033[7m"

    black      = "\033[30m"
    red        = "\033[31m"
    green      = "\033[32m"
    yellow     = "\033[33m"
    blue       = "\033[34m"
    magenta    = "\033[35m"
    cyan       = "\033[36m"
    white      = "\033[37m"
    gray       = "\033[90m"

    bg_black   = "\033[40m"
    bg_red     = "\033[41m"
    bg_green   = "\033[42m"
    bg_yellow  = "\033[43m"
    bg_blue    = "\033[44m"
    bg_magenta = "\033[45m"
    bg_cyan    = "\033[46m"
    bg_white   = "\033[47m"
    bg_gray    = "\033[100m"

    @staticmethod
    def rgb_fg(r: int, g: int, b: int) -> str:
        return f"\033[38;2;{r};{g};{b}m"

    @staticmethod
    def rgb_bg(r: int, g: int, b: int) -> str:
        return f"\033[48;2;{r};{g};{b}m"

    @staticmethod
    def cursor_up(n: int = 1) -> str:
        return f"\033[{n}A"

    @staticmethod
    def cursor_down(n: int = 1) -> str:
        return f"\033[{n}B"

    @staticmethod
    def cursor_forward(n: int = 1) -> str:
        return f"\033[{n}C"

    @staticmethod
    def cursor_back(n: int = 1) -> str:
        return f"\033[{n}D"

    @staticmethod
    def cursor_to(row: int, col: int) -> str:
        return f"\033[{row};{col}H"

    @staticmethod
    def clear_line() -> str:
        return "\033[2K"

    @staticmethod
    def clear_screen() -> str:
        return "\033[2J\033[H"

    @staticmethod
    def save_cursor() -> str:
        return "\033[s"

    @staticmethod
    def restore_cursor() -> str:
        return "\033[u"

    @staticmethod
    def hide_cursor() -> str:
        return "\033[?25l"

    @staticmethod
    def show_cursor() -> str:
        return "\033[?25h"


# ── Box Drawing ────────────────────────────────────────────────────────

class Box:
    TL = "┌"
    TR = "┐"
    BL = "└"
    BR = "┘"
    H  = "─"
    V  = "│"
    T_DOWN = "┬"
    T_UP   = "┴"
    T_RIGHT = "├"
    T_LEFT  = "┤"
    CROSS   = "┼"

    DOUBLE_H = "═"
    DOUBLE_V = "║"
    DOUBLE_TL = "╔"
    DOUBLE_TR = "╗"
    DOUBLE_BL = "╚"
    DOUBLE_BR = "╝"


# ── Data ───────────────────────────────────────────────────────────────

PRIORITY_COLORS = {
    "high":   ANSI.red,
    "medium": ANSI.yellow,
    "low":    ANSI.green,
    "none":   ANSI.gray,
}

PRIORITY_ICONS = {
    "high":   "!!!",
    "medium": "!!",
    "low":    "!",
    "none":   " ",
}

COLUMN_COLORS = [
    ANSI.cyan,
    ANSI.yellow,
    ANSI.magenta,
    ANSI.green,
    ANSI.blue,
    ANSI.red,
]


@dataclass
class Card:
    id: str
    title: str
    description: str = ""
    column: str = "todo"
    priority: str = "none"
    tags: list[str] = field(default_factory=list)
    due_date: str = ""
    assignee: str = ""
    sprint: str = ""
    gh: str = ""
    notes: list[dict] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    trashed: bool = False
    trashed_at: str = ""

    def short_id(self) -> str:
        return self.id[:8] if len(self.id) > 8 else self.id

    def priority_icon(self) -> str:
        return PRIORITY_ICONS.get(self.priority, " ")

    def priority_color(self) -> str:
        return PRIORITY_COLORS.get(self.priority, ANSI.gray)


@dataclass
class Column:
    name: str
    wip_limit: int = 0
    order: int = 0


@dataclass
class Board:
    name: str = "Kanban"
    columns: list[Column] = field(default_factory=list)
    cards: list[Card] = field(default_factory=list)

    def cards_in_column(self, name: str, include_trashed: bool = False) -> list[Card]:
        cards = [c for c in self.cards if c.column == name]
        if not include_trashed:
            cards = [c for c in cards if not c.trashed]
        return sorted(cards, key=lambda c: (
            {"high": 0, "medium": 1, "low": 2, "none": 3}.get(c.priority, 99),
            c.title,
        ))

    def trashed_cards(self) -> list[Card]:
        return [c for c in self.cards if c.trashed]

    def all_tags(self) -> list[str]:
        tags: set[str] = set()
        for c in self.cards:
            tags.update(c.tags)
        return sorted(tags)

    def stats(self) -> dict[str, Any]:
        active = [c for c in self.cards if not c.trashed]
        by_col: dict[str, int] = {}
        for c in active:
            by_col[c.column] = by_col.get(c.column, 0) + 1
        return {
            "total": len(active),
            "trashed": len(self.cards) - len(active),
            "by_column": by_col,
            "tags": len(self.all_tags()),
        }


# ── Store ──────────────────────────────────────────────────────────────

class Store:
    """JSONL-backed board persistence."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Board:
        if not self.path.exists():
            return Board()
        lines = [l.strip() for l in self.path.read_text().splitlines() if l.strip()]
        if not lines:
            return Board()

        board_def = json.loads(lines[0])
        columns = [Column(**c) for c in board_def.get("columns", [])]
        cards = []
        for line in lines[1:]:
            try:
                d = json.loads(line)
                cards.append(Card(
                    id=d.get("id", ""),
                    title=d.get("title", ""),
                    description=d.get("description", ""),
                    column=d.get("column", "todo"),
                    priority=d.get("priority", "none"),
                    tags=d.get("tags", []),
                    due_date=d.get("due_date", ""),
                    assignee=d.get("assignee", ""),
                    sprint=d.get("sprint", ""),
                    gh=d.get("gh", ""),
                    notes=d.get("notes", []),
                    created_at=d.get("created_at", ""),
                    updated_at=d.get("updated_at", ""),
                    trashed=d.get("trashed", False),
                    trashed_at=d.get("trashed_at", ""),
                ))
            except (json.JSONDecodeError, KeyError):
                continue
        return Board(
            name=board_def.get("name", "Kanban"),
            columns=columns,
            cards=cards,
        )

    def save(self, board: Board) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        board_def = {
            "schema": "planner/1",
            "name": board.name,
            "columns": [
                {"name": c.name, "wip_limit": c.wip_limit, "order": c.order}
                for c in sorted(board.columns, key=lambda c: c.order)
            ],
        }
        lines = [json.dumps(board_def)]
        for card in board.cards:
            d = {
                "id": card.id,
                "title": card.title,
                "description": card.description,
                "column": card.column,
                "priority": card.priority,
                "tags": card.tags,
                "due_date": card.due_date,
                "assignee": card.assignee,
                "sprint": card.sprint,
                "gh": card.gh,
                "notes": card.notes,
                "created_at": card.created_at,
                "updated_at": card.updated_at,
                "trashed": card.trashed,
                "trashed_at": card.trashed_at,
            }
            lines.append(json.dumps(d))
        self.path.write_text("\n".join(lines) + "\n")


# ── Renderer ───────────────────────────────────────────────────────────

class Renderer:
    """Standalone terminal renderer for kanban boards."""

    def __init__(self, width: int = 0, color: bool = True) -> None:
        if width <= 0:
            try:
                self.width = os.get_terminal_size().columns
            except (AttributeError, ValueError, OSError):
                self.width = 80
        else:
            self.width = width
        self.color = color and sys.stdout.isatty()

    def _c(self, text: str, color_code: str) -> str:
        if not self.color:
            return text
        return f"{color_code}{text}{ANSI.reset}"

    def _bold(self, text: str) -> str:
        return self._c(text, ANSI.bold)

    def _dim(self, text: str) -> str:
        return self._c(text, ANSI.dim)

    def _truncate(self, text: str, max_width: int) -> str:
        if len(text) <= max_width:
            return text
        if max_width <= 3:
            return text[:max_width]
        return text[:max_width - 1] + "…"

    def _pad(self, text: str, width: int, align: str = "left") -> str:
        if align == "center":
            pad = max(width - len(text), 0)
            left = pad // 2
            return " " * left + text + " " * (pad - left)
        elif align == "right":
            return text.rjust(width)
        return text.ljust(width)

    def render_board(self, board: Board, column_filter: str = "",
                     search: str = "", show_trash: bool = False) -> str:
        if not board.columns:
            return self._dim("  No columns defined.")

        cols = sorted(board.columns, key=lambda c: c.order)

        if show_trash:
            return self._render_trash(board)

        if column_filter:
            cols = [c for c in cols if c.name == column_filter]
            if not cols:
                return self._dim(f"  Column '{column_filter}' not found.")

        col_width = max(self.width // max(len(cols), 1), 20)
        lines: list[str] = []

        # Title bar
        title = f" {board.name} "
        stats = board.stats()
        subtitle = f" {stats['total']} card(s) "
        if stats["trashed"]:
            subtitle += f"  {stats['trashed']} trashed "
        pad = self.width - len(title) - len(subtitle)
        lines.append(
            self._c(title, ANSI.bold + ANSI.bg_blue + ANSI.white)
            + " " * max(pad, 0)
            + self._c(subtitle, ANSI.dim)
        )
        lines.append("")

        # Column headers
        headers = []
        for col in cols:
            count = len(board.cards_in_column(col.name))
            hdr = col.name.upper().replace("_", " ")
            if col.wip_limit:
                over = count > col.wip_limit
                hdr += f" ({count}/{col.wip_limit})"
                if over:
                    hdr += " OVER"
            else:
                hdr += f" ({count})"
            headers.append(hdr)

        # Box top
        sep_parts = []
        for i, col in enumerate(cols):
            sep_parts.append(Box.TL + Box.H * (col_width - 2) + Box.TR)
        sep = Box.T_DOWN.join(sep_parts)
        lines.append(self._dim(sep))

        # Headers
        hdr_cells = []
        for i, (hdr, col) in enumerate(zip(headers, cols)):
            color = COLUMN_COLORS[i % len(COLUMN_COLORS)]
            cell = self._c(self._pad(hdr, col_width - 2, "center"), ANSI.bold + color)
            hdr_cells.append(Box.V + cell + Box.V)
        lines.append("".join(hdr_cells))

        # Header separator
        sep_parts = []
        for i in range(len(cols)):
            sep_parts.append(Box.T_RIGHT + Box.H * (col_width - 2) + Box.T_LEFT)
        sep = Box.CROSS.join(sep_parts)
        lines.append(self._dim(sep))

        # Cards
        by_col: dict[str, list[Card]] = {}
        for col in cols:
            by_col[col.name] = board.cards_in_column(col.name)

        max_rows = max((len(cards) for cards in by_col.values()), default=0)
        if max_rows == 0:
            max_rows = 1

        for row_idx in range(max_rows):
            cells = []
            for i, col in enumerate(cols):
                cards = by_col.get(col.name, [])
                if row_idx < len(cards):
                    card = cards[row_idx]
                    cell = self._render_card(card, col_width - 2, search)
                else:
                    cell = " " * (col_width - 2)
                cells.append(Box.V + cell + Box.V)
            lines.append("".join(cells))

        # Box bottom
        sep_parts = []
        for i in range(len(cols)):
            sep_parts.append(Box.BL + Box.H * (col_width - 2) + Box.BR)
        sep = Box.T_UP.join(sep_parts)
        lines.append(self._dim(sep))

        # Footer
        tag_str = ""
        if board.all_tags():
            tag_str = f"  tags: {', '.join(board.all_tags()[:5])}"
        lines.append(self._dim(f"  {stats['total']} card(s), {len(cols)} column(s){tag_str}"))

        return "\n".join(lines)

    def _render_card(self, card: Card, width: int, search: str = "") -> str:
        parts: list[str] = []

        # Priority icon
        pri_color = card.priority_color()
        parts.append(self._c(card.priority_icon(), pri_color + ANSI.bold))

        # Short ID
        parts.append(self._c(card.short_id(), ANSI.dim))

        # Title
        title = self._truncate(card.title, max(width - 14, 5))
        if search and search.lower() in title.lower():
            title = self._c(title, ANSI.inverse)
        parts.append(title)

        line1 = " ".join(parts)

        # Second line: tags, due, assignee
        meta: list[str] = []
        if card.tags:
            tag_str = " ".join(f"#{t}" for t in card.tags[:2])
            if len(card.tags) > 2:
                tag_str += "+"
            meta.append(self._c(tag_str, ANSI.cyan))
        if card.due_date:
            meta.append(self._c(f"[{card.due_date}]", ANSI.yellow))
        if card.assignee:
            meta.append(self._c(f"@{card.assignee}", ANSI.magenta))
        if card.notes:
            meta.append(self._c(f"({len(card.notes)})", ANSI.dim))

        line2 = " ".join(meta) if meta else ""

        # Truncate both lines to width
        # We need visible-width handling for ANSI codes
        result = line1
        if line2:
            result += "\n" + line2

        return self._pad(self._strip_ansi(line1), width) if not self.color else self._pad_raw(line1, width)

    def _pad_raw(self, text: str, width: int) -> str:
        """Pad text to width, accounting for ANSI escape sequences."""
        visible_len = len(self._strip_ansi(text))
        pad = max(width - visible_len, 0)
        return text + " " * pad

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape sequences for length calculation."""
        import re
        return re.sub(r'\033\[[0-9;]*m', '', text)

    def _render_trash(self, board: Board) -> str:
        trashed = board.trashed_cards()
        lines: list[str] = []
        lines.append(self._c(" TRASH ", ANSI.bold + ANSI.bg_red + ANSI.white))
        lines.append("")

        if not trashed:
            lines.append(self._dim("  Trash is empty."))
            return "\n".join(lines)

        for card in trashed:
            pri = self._c(card.priority_icon(), card.priority_color() + ANSI.bold)
            title = self._truncate(card.title, self.width - 20)
            trashed_at = card.trashed_at[:10] if card.trashed_at else "?"
            lines.append(f"  {pri} {self._c(card.short_id(), ANSI.dim)} {title} {self._c(f'(trashed {trashed_at})', ANSI.dim)}")

        lines.append("")
        lines.append(self._dim(f"  {len(trashed)} card(s) in trash"))
        return "\n".join(lines)


# ── Engine ─────────────────────────────────────────────────────────────

class KanbanEngine:
    """Standalone kanban rendering engine.

    Reads board.jsonl, renders to terminal, supports interactive navigation.
    """

    def __init__(self, board: Board, store: Store | None = None) -> None:
        self.board = board
        self.store = store
        self.renderer = Renderer()
        self._cursor_col = 0
        self._cursor_row = 0
        self._scroll_offset = 0
        self._selected_card: Card | None = None
        self._search_query = ""
        self._column_filter = ""
        self._show_trash = False
        self._dirty = True
        self._running = False

    @classmethod
    def from_file(cls, path: str | Path) -> KanbanEngine:
        store = Store(path)
        board = store.load()
        return cls(board, store)

    # ── Static render ──────────────────────────────────────────────

    def render(self) -> str:
        return self.renderer.render_board(
            self.board,
            column_filter=self._column_filter,
            search=self._search_query,
            show_trash=self._show_trash,
        )

    def render_card_detail(self, card: Card) -> str:
        lines: list[str] = []
        w = min(self.renderer.width, 60)

        # Header
        pri = self._renderer()._c(card.priority_icon(), card.priority_color() + ANSI.bold)
        lines.append(self._renderer()._c(f" {card.title} ", ANSI.bold + ANSI.bg_blue + ANSI.white))
        lines.append("")

        # Fields
        fields = [
            ("ID", card.short_id()),
            ("Column", card.column),
            ("Priority", f"{card.priority} {card.priority_icon()}"),
            ("Tags", ", ".join(card.tags) if card.tags else "—"),
            ("Due", card.due_date or "—"),
            ("Assignee", card.assignee or "—"),
            ("Sprint", card.sprint or "—"),
            ("GitHub", card.gh or "—"),
            ("Created", card.created_at[:19] if card.created_at else "—"),
            ("Updated", card.updated_at[:19] if card.updated_at else "—"),
        ]

        max_label = max(len(f[0]) for f in fields)
        for label, value in fields:
            lines.append(f"  {self._renderer()._c(label.rjust(max_label), ANSI.dim)}  {value}")

        # Description
        if card.description:
            lines.append("")
            lines.append(self._renderer()._c("  Description:", ANSI.bold))
            for desc_line in card.description.split("\n"):
                lines.append(f"    {desc_line}")

        # Notes
        if card.notes:
            lines.append("")
            lines.append(self._renderer()._c(f"  Notes ({len(card.notes)}):", ANSI.bold))
            for note in card.notes[-5:]:
                body = note.get("body", note.get("text", ""))
                ts = note.get("created_at", "")[:10]
                lines.append(f"    {self._renderer()._c(ts, ANSI.dim)} {body[:80]}")

        # Box
        border = Box.TL + Box.H * (w - 2) + Box.TR
        inner = []
        for line in lines:
            stripped = self.renderer._strip_ansi(line)
            padding = max(w - 2 - len(stripped), 0)
            inner.append(Box.V + line + " " * padding + Box.V)
        bottom = Box.BL + Box.H * (w - 2) + Box.BR

        return "\n".join([self.renderer._dim(border)] + inner + [self.renderer._dim(bottom)])

    def _renderer(self) -> Renderer:
        return self.renderer

    # ── Interactive ────────────────────────────────────────────────

    def interactive(self) -> None:
        """Run interactive TUI with keyboard navigation."""
        old_settings = None
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._running = True

            out = sys.stdout.write
            flush = sys.stdout.flush

            out(ANSI.hide_cursor())
            out(ANSI.clear_screen())
            flush()

            while self._running:
                if self._dirty:
                    out(ANSI.clear_screen())
                    out(ANSI.cursor_to(1, 1))
                    out(self.render())
                    if self._selected_card:
                        out("\n")
                        out(self.render_card_detail(self._selected_card))
                    out(ANSI.cursor_to(self.renderer.width and 999, 1))
                    flush()
                    self._dirty = False

                key = self._read_key()
                self._handle_key(key)

        finally:
            if old_settings is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            sys.stdout.write(ANSI.show_cursor())
            sys.stdout.write(ANSI.clear_screen())
            sys.stdout.flush()

    def _read_key(self) -> str:
        fd = sys.stdin.fileno()
        ch = os.read(fd, 1)
        if ch == b"\x1b":
            ch2 = os.read(fd, 1)
            if ch2 == b"[":
                ch3 = os.read(fd, 1)
                return f"\033[{ch3.decode()}"
            return "\x1b"
        return ch.decode(errors="replace")

    def _handle_key(self, key: str) -> None:
        cols = sorted(self.board.columns, key=lambda c: c.order)
        if not cols:
            return

        if key in ("q", "\x03"):  # q or Ctrl+C
            self._running = False
        elif key == "\033[A":  # up
            self._cursor_row = max(0, self._cursor_row - 1)
            self._selected_card = None
            self._dirty = True
        elif key == "\033[B":  # down
            col_name = cols[min(self._cursor_col, len(cols) - 1)].name
            max_row = len(self.board.cards_in_column(col_name)) - 1
            self._cursor_row = min(self._cursor_row, max_row)
            self._cursor_row += 1
            self._selected_card = None
            self._dirty = True
        elif key == "\033[D":  # left
            self._cursor_col = max(0, self._cursor_col - 1)
            self._cursor_row = 0
            self._selected_card = None
            self._dirty = True
        elif key == "\033[C":  # right
            self._cursor_col = min(len(cols) - 1, self._cursor_col + 1)
            self._cursor_row = 0
            self._selected_card = None
            self._dirty = True
        elif key in ("\n", "\r", " "):  # enter/space — select card
            col_name = cols[min(self._cursor_col, len(cols) - 1)].name
            cards = self.board.cards_in_column(col_name)
            if 0 <= self._cursor_row < len(cards):
                card = cards[self._cursor_row]
                if self._selected_card and self._selected_card.id == card.id:
                    self._selected_card = None
                else:
                    self._selected_card = card
            self._dirty = True
        elif key == "/":  # search
            self._search_query = self._prompt("Search: ")
            self._dirty = True
        elif key == "t":  # trash view
            self._show_trash = not self._show_trash
            self._dirty = True
        elif key == "c":  # clear filter
            self._column_filter = ""
            self._search_query = ""
            self._show_trash = False
            self._dirty = True
        elif key == "r":  # refresh
            if self.store:
                self.board = self.store.load()
            self._dirty = True
        elif key.isdigit() and int(key) <= len(cols):
            idx = int(key) - 1
            if idx == self._cursor_col:
                self._column_filter = ""
            else:
                self._column_filter = cols[idx].name
            self._dirty = True

    def _prompt(self, message: str) -> str:
        """Simple input prompt during interactive mode."""
        sys.stdout.write(ANSI.show_cursor())
        sys.stdout.write(f"\n{message}")
        sys.stdout.flush()
        # Read line from stdin in raw mode is complex, just return empty for now
        # In production, implement proper line editing
        sys.stdout.write(ANSI.hide_cursor())
        return ""


# ── CLI entry point ────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="planner render",
        description="Standalone kanban board rendering engine",
    )
    parser.add_argument("board", nargs="?", default=".kanban/board.jsonl",
                        help="Path to board.jsonl")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Run interactive TUI")
    parser.add_argument("-c", "--column", default="",
                        help="Show only this column")
    parser.add_argument("-s", "--search", default="",
                        help="Highlight cards matching search")
    parser.add_argument("-t", "--trash", action="store_true",
                        help="Show trash bin")
    parser.add_argument("-w", "--width", type=int, default=0,
                        help="Terminal width (0 = auto)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colors")

    args = parser.parse_args(argv)

    engine = KanbanEngine.from_file(args.board)
    engine.renderer = Renderer(width=args.width, color=not args.no_color)
    engine._column_filter = args.column
    engine._search_query = args.search
    engine._show_trash = args.trash

    if args.interactive:
        engine.interactive()
    else:
        print(engine.render())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
