"""
planner.kanban — Kanban board with cards, columns, notes, and search.

Thin wrapper around planner.store that adds:
- ColumnDef with WIP limits
- Card notes (simple text annotations)
- Board name and column management
- Search, archive, and render helpers
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .store import Store, Card as _BaseCard, Board as _BaseBoard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(title: str) -> str:
    slug = title.lower().strip()
    slug = unicodedata.normalize("NFKD", slug)
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")[:80]


def _abbrev(text: str, maxlen: int) -> str:
    if len(text) <= maxlen:
        return text
    return text[:maxlen] + "..."


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Note:
    """A lightweight note attached to a card."""
    id: str = ""
    text: str = ""
    author: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Note:
        return cls(
            id=d.get("id", ""),
            text=d.get("text", ""),
            author=d.get("author", ""),
            created_at=d.get("created_at", ""),
        )


@dataclass
class ColumnDef:
    """Column definition with optional WIP limit."""
    name: str = ""
    wip_limit: int = 0
    order: int = 0


@dataclass
class Card:
    """A kanban card with notes."""
    id: str = ""
    title: str = ""
    description: str = ""
    column: str = "todo"
    priority: str = "medium"
    tags: list[str] = field(default_factory=list)
    assignee: str = ""
    due_date: str = ""
    created_at: str = ""
    updated_at: str = ""
    notes: list[Note] = field(default_factory=list)

    @property
    def short_id(self) -> str:
        return self.id[:8] if self.id else ""

    @property
    def priority_icon(self) -> str:
        icons = {"critical": "!!!", "high": "!!", "medium": "!", "low": " "}
        return icons.get(self.priority, "?")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["notes"] = [n.to_dict() for n in self.notes]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Card:
        notes_raw = d.get("notes", [])
        notes = [Note.from_dict(n) if isinstance(n, dict) else n for n in notes_raw]
        return cls(
            id=d.get("id", ""),
            title=d.get("title", ""),
            description=d.get("description", ""),
            column=d.get("column", "todo"),
            priority=d.get("priority", "medium"),
            tags=d.get("tags", []),
            assignee=d.get("assignee", ""),
            due_date=d.get("dueDate", d.get("due_date", "")),
            created_at=d.get("createdAt", d.get("created_at", "")),
            updated_at=d.get("updatedAt", d.get("updated_at", "")),
            notes=notes,
        )


@dataclass
class Board:
    """The kanban board state."""
    name: str = "board"
    columns: list[ColumnDef] = field(default_factory=lambda: [
        ColumnDef(name="todo", wip_limit=0, order=0),
        ColumnDef(name="in_progress", wip_limit=0, order=1),
        ColumnDef(name="review", wip_limit=0, order=2),
        ColumnDef(name="done", wip_limit=0, order=3),
    ])
    cards: list[Card] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": [asdict(c) for c in self.columns],
            "cards": [c.to_dict() for c in self.cards],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Board:
        cols_raw = d.get("columns", [])
        if cols_raw and isinstance(cols_raw[0], str):
            columns = [ColumnDef(name=c, order=i) for i, c in enumerate(cols_raw)]
        elif cols_raw:
            columns = [ColumnDef(
                name=c.get("name", ""),
                wip_limit=c.get("wip_limit", c.get("wipLimit", 0)),
                order=c.get("order", i),
            ) for i, c in enumerate(cols_raw)]
        else:
            columns = [
                ColumnDef(name="todo", order=0),
                ColumnDef(name="in_progress", order=1),
                ColumnDef(name="review", order=2),
                ColumnDef(name="done", order=3),
            ]
        cards_raw = d.get("cards", [])
        cards = [Card.from_dict(c) if isinstance(c, dict) else c for c in cards_raw]
        return cls(
            name=d.get("name", "board"),
            columns=columns,
            cards=cards,
        )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

_DEFAULT_COLUMNS = ["TODO", "IN PROGRESS", "REVIEW", "DONE"]


def _render_board(board: Board) -> str:
    if not board.columns:
        return "  No columns defined."
    col_map: dict[str, list[Card]] = {}
    for card in board.cards:
        col_map.setdefault(card.column, []).append(card)
    lines: list[str] = []
    for col in board.columns:
        cards = col_map.get(col.name, [])
        header = col.name.upper()
        lines.append(f"  {header} ({len(cards)} card(s))")
        for card in cards:
            lines.append(f"    [{card.priority_icon}] {card.id[:8]}  {card.title}")
    total = len(board.cards)
    lines.append(f"\n  {total} card(s)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class KanbanStore:
    """High-level kanban store backed by JSONL."""

    def __init__(self, board_dir: Path | None = None):
        self._store = Store(board_dir)
        self._board: Board | None = None
        self._meta_file = self._store._dir / "board_meta.json"

    def _load_meta(self) -> dict[str, Any]:
        import json
        if self._meta_file.exists():
            try:
                return json.loads(self._meta_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_meta(self, board: Board) -> None:
        import json
        meta = {"name": board.name, "columns": [asdict(c) for c in board.columns]}
        self._meta_file.write_text(json.dumps(meta, indent=2) + "\n")

    def _load(self) -> Board:
        if self._board is None:
            raw = self._store.load_board()
            raw_dict = {"cards": [c.to_dict() for c in raw.cards]}
            self._board = Board.from_dict(raw_dict)
            meta = self._load_meta()
            if meta.get("name"):
                self._board.name = meta["name"]
            if meta.get("columns"):
                self._board.columns = [ColumnDef(
                    name=col.get("name", ""),
                    wip_limit=col.get("wip_limit", col.get("wipLimit", 0)),
                    order=col.get("order", i),
                ) for i, col in enumerate(meta["columns"])]
        return self._board

    def _save(self) -> None:
        if self._board is not None:
            self._save_meta(self._board)
            from .store import Board as RawBoard, Card as RawCard
            raw_cards = []
            for c in self._board.cards:
                d = c.to_dict()
                d["dueDate"] = c.due_date
                d["createdAt"] = c.created_at
                d["updatedAt"] = c.updated_at
                raw_cards.append(RawCard.from_dict(d))
            raw = RawBoard(cards=raw_cards)
            self._store.save_board(raw)

    def init_board(self, name: str, force: bool = False) -> Board:
        meta = self._load_meta()
        if meta.get("name") and not force:
            self._board = None  # force reload from disk
            return self._load()
        self._board = Board(name=name)
        self._save()
        return self._board

    def load_board(self) -> Board:
        return self._load()

    def add_card(self, title: str, column: str = "todo", priority: str = "medium",
                 description: str = "", tags: list[str] | None = None,
                 assignee: str = "", due_date: str = "") -> Card:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        card = Card(
            id=_make_id(title) or f"card-{int(datetime.now().timestamp())}",
            title=title,
            description=description,
            column=column,
            priority=priority,
            tags=tags or [],
            assignee=assignee,
            due_date=due_date,
            created_at=now,
            updated_at=now,
        )
        self._load().cards.append(card)
        self._save()
        return card

    def get_card(self, card_id: str) -> Card | None:
        board = self._load()
        matches = [c for c in board.cards if c.id == card_id or c.id.startswith(card_id)]
        if len(matches) == 1:
            return matches[0]
        return None

    def update_card(self, card_id: str, **kwargs: Any) -> Card | None:
        board = self._load()
        for card in board.cards:
            if card.id == card_id:
                for key, value in kwargs.items():
                    if hasattr(card, key):
                        setattr(card, key, value)
                self._save()
                return card
        return None

    def delete_card(self, card_id: str) -> bool:
        board = self._load()
        exact = [c for c in board.cards if c.id == card_id]
        if exact:
            board.cards = [c for c in board.cards if c.id != card_id]
            self._save()
            return True
        prefix_matches = [c for c in board.cards if c.id.startswith(card_id)]
        if len(prefix_matches) == 1:
            board.cards = [c for c in board.cards if c.id != prefix_matches[0].id]
            self._save()
            return True
        return False

    def move_card(self, card_id: str, to_column: str) -> Card | None:
        board = self._load()
        valid_cols = {c.name for c in board.columns}
        if to_column not in valid_cols:
            return None
        for card in board.cards:
            if card.id == card_id:
                card.column = to_column
                self._save()
                return card
        return None

    def list_cards(self, column: str | None = None, priority: str | None = None,
                   tag: str | None = None, assignee: str | None = None,
                   limit: int | None = None) -> list[Card]:
        board = self._load()
        result = board.cards
        if column:
            result = [c for c in result if c.column == column]
        if priority:
            result = [c for c in result if c.priority == priority]
        if tag:
            result = [c for c in result if tag in c.tags]
        if assignee:
            result = [c for c in result if c.assignee == assignee]
        if limit:
            result = result[:limit]
        return result

    def search_cards(self, query: str, limit: int = 20) -> list[Card]:
        q = query.lower()
        board = self._load()
        results: list[Card] = []
        for card in board.cards:
            if (q in card.title.lower()
                    or q in card.description.lower()
                    or any(q in t.lower() for t in card.tags)):
                results.append(card)
                if len(results) >= limit:
                    break
        return results

    def add_note(self, card_id: str, text: str, author: str = "") -> Note | None:
        from datetime import datetime, timezone
        board = self._load()
        for card in board.cards:
            if card.id == card_id:
                note = Note(
                    id=f"note-{int(datetime.now().timestamp())}",
                    text=text,
                    author=author,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                card.notes.append(note)
                self._save()
                return note
        return None

    def list_notes(self, card_id: str) -> list[Note] | None:
        board = self._load()
        for card in board.cards:
            if card.id == card_id:
                return card.notes
        return None

    def delete_note(self, card_id: str, note_id: str) -> bool:
        board = self._load()
        for card in board.cards:
            if card.id == card_id:
                exact = [n for n in card.notes if n.id == note_id]
                if exact:
                    card.notes = [n for n in card.notes if n.id != note_id]
                    self._save()
                    return True
                prefix_matches = [n for n in card.notes if n.id.startswith(note_id)]
                if len(prefix_matches) == 1:
                    card.notes = [n for n in card.notes if n.id != prefix_matches[0].id]
                    self._save()
                    return True
        return False

    def archive_done(self) -> int:
        board = self._load()
        done_cards = [c for c in board.cards if c.column == "done"]
        board.cards = [c for c in board.cards if c.column != "done"]
        count = len(done_cards)
        if count:
            self._save()
        return count

    def list_columns(self) -> list[ColumnDef]:
        return self._load().columns

    def add_column(self, name: str, wip_limit: int = 0) -> ColumnDef | None:
        board = self._load()
        if any(c.name == name for c in board.columns):
            return None
        col = ColumnDef(name=name, wip_limit=wip_limit, order=len(board.columns))
        board.columns.append(col)
        self._save()
        return col

    def rename_column(self, old_name: str, new_name: str) -> bool:
        board = self._load()
        for col in board.columns:
            if col.name == old_name:
                col.name = new_name
                for card in board.cards:
                    if card.column == old_name:
                        card.column = new_name
                self._save()
                return True
        return False

    def remove_column(self, name: str, move_to: str | None = None) -> bool:
        board = self._load()
        orig_len = len(board.columns)
        board.columns = [c for c in board.columns if c.name != name]
        if len(board.columns) < orig_len:
            if move_to:
                for card in board.cards:
                    if card.column == name:
                        card.column = move_to
            self._save()
            return True
        return False

    def stats(self) -> dict[str, Any]:
        board = self._load()
        by_col: dict[str, int] = {}
        by_pri: dict[str, int] = {}
        for card in board.cards:
            by_col[card.column] = by_col.get(card.column, 0) + 1
            by_pri[card.priority] = by_pri.get(card.priority, 0) + 1
        return {
            "total": len(board.cards),
            "columns": by_col,
            "priorities": by_pri,
            "column_count": len(board.columns),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_kanban_store: KanbanStore | None = None


def get_kanban_store(board_dir: Path | None = None) -> KanbanStore:
    global _kanban_store
    if _kanban_store is None:
        _kanban_store = KanbanStore(board_dir)
    return _kanban_store


def reset_kanban_store() -> None:
    global _kanban_store
    _kanban_store = None
