"""
kanban — kanban board with notes, built into the planner package.

JSON-backed board with columns, cards, and per-card notes (comments).
Every operation is non-interactive and scriptable.

Usage as a module::

    from planner import KanbanStore
    store = KanbanStore()
    card = store.add_card("Fix boot order", column="todo", priority="high")
    store.move_card(card.id, "in_progress")
    store.add_note(card.id, "Started debugging the init script")

Usage as a CLI::

    planner kanban init
    planner kanban add "Fix boot order" --priority high
    planner kanban list --column todo
    planner kanban board
    planner kanban move <id> in_progress
    planner kanban note add <id> "Looking into it"
"""

from __future__ import annotations

import re
import sys
import json
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

from . import config

logger = logging.getLogger("planner.kanban")

_BOARD_FILE = "board.json"
_MAX_ID_SLUG = 60

DEFAULT_COLUMNS = [
    {"name": "todo", "wip_limit": 0, "order": 0},
    {"name": "in_progress", "wip_limit": 3, "order": 1},
    {"name": "review", "wip_limit": 0, "order": 2},
    {"name": "done", "wip_limit": 0, "order": 3},
]

PRIORITIES = ("low", "medium", "high", "critical")
PRIORITY_ICONS = {"low": " ", "medium": "!", "high": "!!", "critical": "!!!"}
PRIORITY_SORT = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _make_id(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug[:_MAX_ID_SLUG].rstrip("-")
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{slug}"


def _abbrev(s: str, max_len: int = 50) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s


@dataclass
class Note:
    """A comment attached to a card."""
    id: str = ""
    text: str = ""
    author: str = ""
    created_at: str = ""


@dataclass
class Card:
    """A single kanban card (task)."""
    id: str = ""
    title: str = ""
    description: str = ""
    column: str = "todo"
    priority: str = "medium"
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    due_date: str = ""
    assignee: str = ""
    notes: list[Note] = field(default_factory=list)

    @property
    def short_id(self) -> str:
        return self.id[:8] if self.id else ""

    @property
    def priority_icon(self) -> str:
        return PRIORITY_ICONS.get(self.priority, " ")

    def to_dict(self) -> dict:
        return {**asdict(self), "notes": [asdict(n) for n in self.notes]}

    @classmethod
    def from_dict(cls, d: dict) -> Card:
        notes = [Note(**n) for n in d.pop("notes", [])]
        return cls(notes=notes, **d)


@dataclass
class ColumnDef:
    name: str
    wip_limit: int = 0
    order: int = 0


@dataclass
class Board:
    name: str = "board"
    columns: list[ColumnDef] = field(default_factory=lambda: [ColumnDef(**c) for c in DEFAULT_COLUMNS])
    cards: list[Card] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "columns": [asdict(c) for c in self.columns], "cards": [c.to_dict() for c in self.cards]}

    @classmethod
    def from_dict(cls, d: dict) -> Board:
        return cls(
            name=d.get("name", "board"),
            columns=[ColumnDef(**c) for c in d.get("columns", DEFAULT_COLUMNS)],
            cards=[Card.from_dict(c) for c in d.get("cards", [])],
        )


class _JSONBackend:
    def __init__(self, board_dir: Path):
        self._dir = board_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / _BOARD_FILE

    def load(self) -> Board:
        if not self._path.exists():
            return Board()
        try:
            return Board.from_dict(json.loads(self._path.read_text()))
        except (json.JSONDecodeError, Exception):
            logger.warning("Corrupt board file, starting fresh")
            return Board()

    def save(self, board: Board) -> None:
        self._path.write_text(json.dumps(board.to_dict(), indent=2, default=str))


class KanbanStore:
    """Kanban board store — all operations are non-interactive."""

    def __init__(self, board_dir: str | Path | None = None):
        self._dir = Path(board_dir) if board_dir else config.default_board_dir()
        self._bk = _JSONBackend(self._dir)

    def init_board(self, name: str = "board", force: bool = False) -> Board:
        path = self._dir / _BOARD_FILE
        if path.exists() and not force:
            return self._bk.load()
        board = Board(name=name)
        self._bk.save(board)
        return board

    def load_board(self) -> Board:
        return self._bk.load()

    def _find_one(self, board: Board, prefix: str) -> Card | None:
        lower = prefix.lower()
        matches = [c for c in board.cards if c.id.lower().startswith(lower)]
        if len(matches) != 1:
            if len(matches) > 1:
                logger.warning("Ambiguous id '%s': %s", prefix, [m.short_id for m in matches])
            return None
        return matches[0]

    def add_card(self, title: str, column: str = "", priority: str = "medium",
                 description: str = "", tags: list[str] | None = None,
                 due_date: str = "", assignee: str = "") -> Card:
        board = self.load_board()
        column = column or (board.columns[0].name if board.columns else "todo")
        now = datetime.now(timezone.utc).isoformat()
        card = Card(
            id=_make_id(title), title=title, description=description,
            column=column, priority=priority if priority in PRIORITIES else "medium",
            tags=tags or [], created_at=now, updated_at=now,
            due_date=due_date, assignee=assignee,
        )
        board.cards.insert(0, card)
        self._bk.save(board)
        return card

    def get_card(self, card_id: str) -> Card | None:
        board = self.load_board()
        return self._find_one(board, card_id)

    def update_card(self, card_id: str, **kwargs: Any) -> Card | None:
        board = self.load_board()
        card = self._find_one(board, card_id)
        if card is None:
            return None
        for key in ("title", "description", "priority", "tags", "due_date", "assignee"):
            if key in kwargs and kwargs[key] is not None:
                setattr(card, key, kwargs[key])
        card.updated_at = datetime.now(timezone.utc).isoformat()
        self._bk.save(board)
        return card

    def delete_card(self, card_id: str) -> bool:
        board = self.load_board()
        card = self._find_one(board, card_id)
        if card is None:
            return False
        board.cards.remove(card)
        self._bk.save(board)
        return True

    def move_card(self, card_id: str, column: str) -> Card | None:
        board = self.load_board()
        col_names = {c.name for c in board.columns}
        if column not in col_names:
            logger.warning("Unknown column '%s'. Known: %s", column, sorted(col_names))
            return None
        card = self._find_one(board, card_id)
        if card is None:
            return None
        card.column = column
        card.updated_at = datetime.now(timezone.utc).isoformat()
        self._bk.save(board)
        return card

    def list_cards(self, column: str | None = None, priority: str | None = None,
                   tag: str | None = None, assignee: str | None = None,
                   limit: int = 100) -> list[Card]:
        board = self.load_board()
        results: list[Card] = []
        for card in board.cards:
            if column and card.column != column:
                continue
            if priority and card.priority != priority:
                continue
            if tag and tag not in card.tags:
                continue
            if assignee and card.assignee != assignee:
                continue
            results.append(card)
            if len(results) >= limit:
                break
        return results

    def search_cards(self, query: str, limit: int = 30) -> list[Card]:
        q = query.lower()
        board = self.load_board()
        return [c for c in board.cards
                if q in c.title.lower() or q in c.description.lower()
                or q in " ".join(c.tags).lower() or q in c.assignee.lower()][:limit]

    def archive_done(self) -> int:
        board = self.load_board()
        before = len(board.cards)
        board.cards = [c for c in board.cards if c.column != "done"]
        removed = before - len(board.cards)
        if removed:
            self._bk.save(board)
        return removed

    def add_note(self, card_id: str, text: str, author: str = "") -> Note | None:
        board = self.load_board()
        card = self._find_one(board, card_id)
        if card is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        note = Note(id=_make_id(f"note-{card_id}"), text=text, author=author, created_at=now)
        card.notes.append(note)
        card.updated_at = now
        self._bk.save(board)
        return note

    def list_notes(self, card_id: str) -> list[Note] | None:
        board = self.load_board()
        card = self._find_one(board, card_id)
        return card.notes if card else None

    def delete_note(self, card_id: str, note_id: str) -> bool:
        board = self.load_board()
        card = self._find_one(board, card_id)
        if card is None:
            return False
        before = len(card.notes)
        card.notes = [n for n in card.notes if n.id[:8] != note_id and n.id != note_id]
        if len(card.notes) == before:
            return False
        card.updated_at = datetime.now(timezone.utc).isoformat()
        self._bk.save(board)
        return True

    def list_columns(self) -> list[ColumnDef]:
        return self.load_board().columns

    def add_column(self, name: str, wip_limit: int = 0) -> ColumnDef | None:
        board = self.load_board()
        if any(c.name == name for c in board.columns):
            logger.warning("Column '%s' already exists", name)
            return None
        col = ColumnDef(name=name, wip_limit=wip_limit, order=len(board.columns))
        board.columns.append(col)
        self._bk.save(board)
        return col

    def rename_column(self, name: str, new_name: str) -> bool:
        board = self.load_board()
        for col in board.columns:
            if col.name == name:
                col.name = new_name
                for card in board.cards:
                    if card.column == name:
                        card.column = new_name
                self._bk.save(board)
                return True
        return False

    def remove_column(self, name: str, move_to: str = "todo") -> bool:
        board = self.load_board()
        before = len(board.columns)
        board.columns = [c for c in board.columns if c.name != name]
        if len(board.columns) == before:
            return False
        for card in board.cards:
            if card.column == name:
                card.column = move_to
        self._bk.save(board)
        return True

    def stats(self) -> dict:
        board = self.load_board()
        by_col: dict[str, int] = {}
        by_pri: dict[str, int] = {}
        for card in board.cards:
            by_col[card.column] = by_col.get(card.column, 0) + 1
            by_pri[card.priority] = by_pri.get(card.priority, 0) + 1
        return {
            "total": len(board.cards),
            "columns": sorted(by_col.items()),
            "priorities": sorted(by_pri.items()),
            "column_count": len(board.columns),
        }


_store: KanbanStore | None = None


def get_kanban_store(board_dir: str | Path | None = None) -> KanbanStore:
    global _store
    if _store is None or board_dir is not None:
        _store = KanbanStore(board_dir=board_dir)
    return _store


def reset_kanban_store() -> None:
    global _store
    _store = None


def _render_board(board: Board, width: int = 78) -> str:
    if not board.columns:
        return "  No columns."
    col_width = max(width // len(board.columns), 20)
    lines: list[str] = [f"  Kanban: {board.name}", ""]

    cols_sorted = sorted(board.columns, key=lambda c: c.order)
    headers = [f"  {c.name.upper()}" + (f" (WIP:{c.wip_limit})" if c.wip_limit else "") for c in cols_sorted]

    by_col: dict[str, list[Card]] = {c.name: [] for c in board.columns}
    for card in board.cards:
        by_col.setdefault(card.column, []).append(card)

    for col_name in by_col:
        by_col[col_name].sort(key=lambda c: PRIORITY_SORT.get(c.priority, 99))

    sep = "─" * (sum(col_width for _ in cols_sorted) + len(cols_sorted) + 1)
    lines.append(f"┌{sep}┐")
    lines.append(f"│{'│'.join(f'{h:^{col_width}}' for h, cw in zip(headers, [col_width]*len(cols_sorted)))}│")
    lines.append(f"├{sep}┤")

    max_rows = max((len(cards) for cards in by_col.values()), default=1)
    for row_idx in range(max_rows):
        cells = []
        for col in cols_sorted:
            cards = by_col.get(col.name, [])
            if row_idx < len(cards):
                c = cards[row_idx]
                due = f" [{c.due_date}]" if c.due_date else ""
                assign = f" @{c.assignee}" if c.assignee else ""
                tags = ""
                if c.tags:
                    tags = " #" + ",".join(c.tags[:2])
                    if len(c.tags) > 2:
                        tags += "+"
                notes = f" ({len(c.notes)})" if c.notes else ""
                label = f"{c.priority_icon} {c.short_id} {_abbrev(c.title, col_width-13)}{due}{assign}{tags}{notes}"
                cells.append(f" {label:<{col_width-1}}")
            else:
                cells.append(" " * col_width)
        lines.append(f"│{'│'.join(f'{c:<{col_width}}' for c, cw in zip(cells, [col_width]*len(cols_sorted)))}│")

    lines.append(f"└{sep}┘")
    lines.append(f"  {len(board.cards)} card(s) across {len(board.columns)} column(s)")
    return "\n".join(lines)


def cli_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="kanban", description="Kanban board with notes")
    parser.add_argument("--dir", default=None, help="Board directory (default: config/env)")
    sub = parser.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init", help="Initialize a new board")
    p_init.add_argument("--name", default="board", help="Board name")
    p_init.add_argument("--force", action="store_true", help="Recreate if exists")

    p_add = sub.add_parser("add", help="Add a card")
    p_add.add_argument("title", help="Card title")
    p_add.add_argument("--column", default="", help="Target column")
    p_add.add_argument("--priority", default="medium", choices=PRIORITIES)
    p_add.add_argument("--desc", default="", help="Description")
    p_add.add_argument("--tags", default="", help="Comma-separated tags")
    p_add.add_argument("--due", default="", help="Due date (ISO)")
    p_add.add_argument("--assignee", default="", help="Assignee")

    p_list = sub.add_parser("list", help="List cards")
    p_list.add_argument("--column", default=None, help="Filter by column")
    p_list.add_argument("--priority", default=None, choices=PRIORITIES)
    p_list.add_argument("--tag", default=None, help="Filter by tag")
    p_list.add_argument("--assignee", default=None, help="Filter by assignee")
    p_list.add_argument("--limit", type=int, default=50)

    p_show = sub.add_parser("show", help="Show card details")
    p_show.add_argument("card_id", help="Card id or prefix")

    p_edit = sub.add_parser("edit", help="Edit a card")
    p_edit.add_argument("card_id", help="Card id or prefix")
    p_edit.add_argument("--title", default=None)
    p_edit.add_argument("--desc", default=None)
    p_edit.add_argument("--priority", default=None, choices=PRIORITIES)
    p_edit.add_argument("--tags", default=None)
    p_edit.add_argument("--due", default=None)
    p_edit.add_argument("--assignee", default=None)

    p_move = sub.add_parser("move", help="Move card to another column")
    p_move.add_argument("card_id", help="Card id or prefix")
    p_move.add_argument("column", help="Target column")

    p_del = sub.add_parser("delete", aliases=["rm"], help="Delete a card")
    p_del.add_argument("card_id", help="Card id or prefix")

    sub.add_parser("board", help="Show ASCII kanban board")

    p_note = sub.add_parser("note", help="Manage notes on a card")
    p_note_sub = p_note.add_subparsers(dest="note_cmd")
    p_na = p_note_sub.add_parser("add", help="Add note")
    p_na.add_argument("card_id")
    p_na.add_argument("text", help="Note text")
    p_na.add_argument("--author", default="")
    p_nl = p_note_sub.add_parser("list", help="List notes")
    p_nl.add_argument("card_id")
    p_nd = p_note_sub.add_parser("delete", aliases=["rm"], help="Delete note")
    p_nd.add_argument("card_id")
    p_nd.add_argument("note_id")

    sub.add_parser("columns", help="List columns")
    p_ca = sub.add_parser("column-add", help="Add a column")
    p_ca.add_argument("name")
    p_ca.add_argument("--wip", type=int, default=0)
    p_cr = sub.add_parser("column-rename", help="Rename a column")
    p_cr.add_argument("name")
    p_cr.add_argument("new_name")
    p_crm = sub.add_parser("column-rm", help="Remove a column")
    p_crm.add_argument("name")
    p_crm.add_argument("--move-to", default="todo")

    sub.add_parser("archive", help="Archive (delete) all done cards")
    p_search = sub.add_parser("search", help="Search cards")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=30)
    sub.add_parser("stats", help="Board statistics")

    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 1

    store = get_kanban_store(board_dir=args.dir)

    if args.cmd == "init":
        board = store.init_board(name=args.name, force=args.force)
        print(f"Board '{board.name}' at {store._dir / _BOARD_FILE}")
        cols = ", ".join(c.name for c in board.columns)
        print(f"  {len(board.columns)} columns: {cols}")
        return 0

    if args.cmd == "add":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        card = store.add_card(args.title, column=args.column, priority=args.priority,
                              description=args.desc, tags=tags, due_date=args.due, assignee=args.assignee)
        print(f"Added: {card.short_id}  {card.title}  [{card.column}]  {card.priority_icon}")
        return 0

    if args.cmd == "list":
        cards = store.list_cards(column=args.column, priority=args.priority,
                                 tag=args.tag, assignee=args.assignee, limit=args.limit)
        if not cards:
            print("No cards found.")
            return 0
        for c in cards:
            tags_s = f"  [{', '.join(c.tags)}]" if c.tags else ""
            due_s = f"  [{c.due_date}]" if c.due_date else ""
            assign_s = f"  @{c.assignee}" if c.assignee else ""
            notes_s = f"  ({len(c.notes)})" if c.notes else ""
            print(f"  {c.priority_icon} {c.short_id}  {c.title}  [{c.column}]{tags_s}{due_s}{assign_s}{notes_s}")
        print(f"\n  {len(cards)} card(s)")
        return 0

    if args.cmd == "show":
        card = store.get_card(args.card_id)
        if card is None:
            print(f"Card not found: {args.card_id}")
            return 1
        tags_s = ", ".join(card.tags) if card.tags else "none"
        print(f"  {card.priority_icon}  {card.title}")
        print(f"  id:       {card.id}")
        print(f"  column:   {card.column}")
        print(f"  priority: {card.priority}")
        print(f"  tags:     {tags_s}")
        print(f"  assignee: {card.assignee or 'unassigned'}")
        print(f"  due:      {card.due_date or 'none'}")
        print(f"  created:  {card.created_at}")
        print(f"  updated:  {card.updated_at}")
        if card.description:
            print(f"\n  {card.description}")
        if card.notes:
            print(f"\n  Notes ({len(card.notes)}):")
            for n in card.notes:
                author_s = f"  [{n.author}]" if n.author else ""
                print(f"    [{n.id[:8]}]  {n.created_at[:10]}{author_s}")
                for line in n.text.split("\n"):
                    print(f"      {line}")
        return 0

    if args.cmd == "edit":
        kwargs: dict[str, Any] = {}
        if args.title is not None: kwargs["title"] = args.title
        if args.desc is not None: kwargs["description"] = args.desc
        if args.priority is not None: kwargs["priority"] = args.priority
        if args.tags is not None: kwargs["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
        if args.due is not None: kwargs["due_date"] = args.due
        if args.assignee is not None: kwargs["assignee"] = args.assignee
        if not kwargs:
            print("No changes specified.")
            return 1
        card = store.update_card(args.card_id, **kwargs)
        if card is None:
            print(f"Card not found: {args.card_id}")
            return 1
        print(f"Updated: {card.short_id}  {card.title}")
        return 0

    if args.cmd == "move":
        card = store.move_card(args.card_id, args.column)
        if card is None:
            print(f"Card not found: {args.card_id}")
            return 1
        print(f"Moved: {card.short_id}  {card.title}  ->  {card.column}")
        return 0

    if args.cmd in ("delete", "rm"):
        print("Deleted." if store.delete_card(args.card_id) else "Card not found.")
        return 0

    if args.cmd == "board":
        board = store.load_board()
        tw = shutil.get_terminal_size(fallback=(80, 24)).columns
        print(_render_board(board, width=tw - 4))
        return 0

    if args.cmd == "note":
        if args.note_cmd is None:
            print("Usage: kanban note add|list|delete ...")
            return 1
        if args.note_cmd == "add":
            note = store.add_note(args.card_id, args.text, author=args.author)
            print(f"Note added: [{note.id[:8]}]" if note else "Card not found.")
            return 0 if note else 1
        if args.note_cmd == "list":
            notes = store.list_notes(args.card_id)
            if notes is None:
                print("Card not found.")
                return 1
            if not notes:
                print("No notes on this card.")
                return 0
            for n in notes:
                author_s = f"  [{n.author}]" if n.author else ""
                print(f"  [{n.id[:8]}]  {n.created_at[:10]}{author_s}")
                print(f"    {n.text}")
            return 0
        if args.note_cmd in ("delete", "rm"):
            print("Note deleted." if store.delete_note(args.card_id, args.note_id) else "Note not found.")
            return 0

    if args.cmd == "columns":
        for c in sorted(store.list_columns(), key=lambda x: x.order):
            wip = f"  (wip: {c.wip_limit})" if c.wip_limit else ""
            print(f"  {c.name}{wip}")
        return 0

    if args.cmd == "column-add":
        col = store.add_column(args.name, wip_limit=args.wip)
        print(f"Column added: {col.name}" if col else "Failed.")
        return 0 if col else 1

    if args.cmd == "column-rename":
        print("Renamed." if store.rename_column(args.name, args.new_name) else "Column not found.")
        return 0

    if args.cmd == "column-rm":
        print("Removed." if store.remove_column(args.name, move_to=args.move_to) else "Column not found.")
        return 0

    if args.cmd == "archive":
        print(f"Archived {store.archive_done()} done card(s).")
        return 0

    if args.cmd == "search":
        results = store.search_cards(args.query, limit=args.limit)
        if not results:
            print(f"No cards matching '{args.query}'")
            return 0
        for c in results:
            tags_s = f"  [{', '.join(c.tags)}]" if c.tags else ""
            print(f"  {c.priority_icon} {c.short_id}  {c.title}  [{c.column}]{tags_s}")
        print(f"\n  {len(results)} card(s)")
        return 0

    if args.cmd == "stats":
        s = store.stats()
        print(f"  Total cards: {s['total']}")
        print(f"  Columns:     {s['column_count']}")
        for name, count in s["columns"]:
            print(f"    {name:15s}  {count:3d}  {'█' * count}")
        print(f"  Priorities:")
        for pri, count in s["priorities"]:
            print(f"    {pri:10s}  {count}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
