"""
planner.store — unified JSONL store for board cards and notes.

Single source of truth for both kanban cards and dev journal notes,
stored as JSONL files that match the web API format.

Board: <repo>/.kanban/board.jsonl
Notes: <repo>/.dev-notes/store/notes.journal.jsonl

Usage::

    from app_planner.store import PlannerStore

    store = PlannerStore()
    card = store.add_card("Fix boot order", column="todo", priority="high")
    note = store.create_note("Debugged init script", status="done")
    store.sync()  # reconcile notes ↔ board
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import config

# ── Data Models ──────────────────────────────────────────────────────────


@dataclass
class Card:
    """A single kanban card."""

    id: str = ""
    title: str = ""
    description: str = ""
    column: str = "todo"
    priority: str = "medium"
    tags: list[str] = field(default_factory=list)
    assignee: str = ""
    due_date: str = ""
    sprint: str = ""
    gh: str = ""
    card_type: str = ""
    blocked_by: list[str] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Card:
        return cls(
            id=d.get("id", ""),
            title=d.get("title", ""),
            description=d.get("description", ""),
            column=d.get("column", "todo"),
            priority=d.get("priority", "medium"),
            tags=coerce_tags(d.get("tags")),
            assignee=d.get("assignee", ""),
            due_date=d.get("due_date", d.get("dueDate", "")),
            sprint=d.get("sprint", ""),
            gh=d.get("gh", ""),
            card_type=d.get("card_type", d.get("type", "")),
            blocked_by=d.get("blocked_by", []),
            notes=d.get("notes", []),
            created_at=d.get("created_at", d.get("createdAt", "")),
            updated_at=d.get("updated_at", d.get("updatedAt", "")),
        )


@dataclass
class Note:
    """A dev journal note."""

    id: str = ""
    title: str = ""
    body: str = ""
    status: str = "open"
    tags: list[str] = field(default_factory=list)
    sprint: str = ""
    gh: str = ""
    assignee: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Note:
        return cls(
            id=d.get("id", ""),
            title=d.get("title", ""),
            body=d.get("body", ""),
            status=d.get("status", "open"),
            tags=coerce_tags(d.get("tags")),
            sprint=d.get("sprint", ""),
            gh=d.get("gh", ""),
            assignee=d.get("assignee", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


@dataclass
class Board:
    """The kanban board state."""

    name: str = "Main"
    columns: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "todo", "wip_limit": 5, "order": 0},
            {"name": "in_progress", "wip_limit": 3, "order": 1},
            {"name": "review", "wip_limit": 2, "order": 2},
            {"name": "done", "wip_limit": 0, "order": 3},
        ]
    )
    cards: list[Card] = field(default_factory=list)


def coerce_tags(value: Any) -> list[str]:
    """Normalize a tags field to a list of strings.

    Journal records written before hardening carry comma-separated
    strings; ``list("a,b")`` would explode them into single characters.
    """
    if not value:
        return []
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    if isinstance(value, (list, tuple)):
        return [str(t) for t in value]
    return [str(value)]


@dataclass(frozen=True)
class SyncReport:
    """Outcome of the most recent PlannerStore.sync() run (card 3ccc38b7)."""

    added: int = 0
    updated: int = 0
    total: int = 0
    reverted: tuple[str, ...] = ()
    repaired: bool = False
    dry_run: bool = False


# ── Store ────────────────────────────────────────────────────────────────


class PlannerStore:
    """Unified JSONL store for board cards and notes."""

    def __init__(
        self,
        board_dir: Path | None = None,
        notes_dir: Path | None = None,
    ) -> None:
        self._board_dir = Path(board_dir) if board_dir else config.default_board_dir()
        self._notes_dir = Path(notes_dir) if notes_dir else config.default_notes_dir()
        self._board_dir.mkdir(parents=True, exist_ok=True)
        self._notes_dir.mkdir(parents=True, exist_ok=True)
        self._board_file = self._board_dir / "board.jsonl"
        self._notes_file = self._notes_dir / "notes.journal.jsonl"
        self.last_sync_report: SyncReport | None = None

    # ── Board ───────────────────────────────────────────────────────────

    @staticmethod
    def _card_line(card: Card) -> str:
        """Canonical, byte-stable JSONL serialization of a card line."""
        return json.dumps(card.to_dict(), sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _schema_line(name: str, columns: list[dict[str, Any]]) -> str:
        """Canonical serialization of the board header line."""
        return json.dumps(
            {"schema": "planner/1", "name": name, "columns": columns},
            sort_keys=True,
            ensure_ascii=False,
        )

    def _atomic_write(self, text: str) -> None:
        """Write the board atomically (temp file + rename) to avoid torn writes."""
        fd, tmp = tempfile.mkstemp(dir=self._board_dir, prefix=".board-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, self._board_file)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _has_schema_header(lines: list[str]) -> bool:
        for raw in lines:
            try:
                if json.loads(raw).get("schema") == "planner/1":
                    return True
            except json.JSONDecodeError:
                continue
        return False

    def _surgical_rewrite(self, replacements: dict[str, str | None]) -> int:
        """Rewrite only the JSONL lines for the given card ids.

        *replacements* maps card id -> canonical next line (None deletes the
        line). Every other line — header, unrelated/note-schema cards, even
        unparsable lines — is preserved byte-for-byte. Writes happen only when
        the file actually changed, atomically.

        Returns:
            Number of lines matched and replaced.
        """
        if not self._board_file.exists():
            return 0
        old_text = self._board_file.read_text()
        out: list[str] = []
        matched = 0
        for raw in old_text.splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                out.append(raw)
                continue
            if isinstance(obj, dict) and obj.get("id") in replacements:
                new_line = replacements[obj["id"]]
                matched += 1
                if new_line is not None:
                    out.append(new_line)
                continue
            out.append(raw)
        new_text = "\n".join(out) + "\n" if out else ""
        if new_text != old_text:
            self._atomic_write(new_text)
        return matched

    def _append_card(self, card: Card) -> None:
        """Append a single card line, adding the schema header only if missing.

        Existing lines (including old note-schema cards) are preserved
        byte-for-byte, so add operations produce a one-line diff.
        """
        existing = self._board_file.read_text().splitlines() if self._board_file.exists() else []
        body = [raw for raw in existing if raw.strip()]
        out = list(body)
        if not self._has_schema_header(body):
            board = self.load_board()
            out.insert(0, self._schema_line(board.name, board.columns))
        out.append(self._card_line(card))
        self._atomic_write("\n".join(out) + "\n")

    def _read_board_lines(self) -> list[dict[str, Any]]:
        if not self._board_file.exists():
            return []
        lines: list[dict[str, Any]] = []
        for raw in self._board_file.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return lines

    def _write_board(self, board: Board) -> None:
        lines: list[str] = []
        lines.append(self._schema_line(board.name, board.columns))
        for card in board.cards:
            lines.append(self._card_line(card))
        self._atomic_write("\n".join(lines) + "\n" if lines else "")

    def load_board(self) -> Board:
        lines = self._read_board_lines()
        board = Board()
        for obj in lines:
            if obj.get("schema") == "planner/1" and obj.get("columns"):
                board.name = obj.get("name", "Main")
                board.columns = obj["columns"]
            elif obj.get("id") and obj.get("title"):
                board.cards.append(Card.from_dict(obj))
        return board

    def save_board(self, board: Board) -> None:
        self._write_board(board)

    def get_card(self, card_id: str) -> Card | None:
        for obj in self._read_board_lines():
            if obj.get("id") == card_id and obj.get("title"):
                return Card.from_dict(obj)
        return None

    def add_card(
        self,
        title: str,
        column: str = "todo",
        priority: str = "medium",
        description: str = "",
        tags: list[str] | None = None,
        assignee: str = "",
        due_date: str = "",
        sprint: str = "",
        gh: str = "",
        card_type: str = "",
    ) -> Card:
        now = datetime.now(UTC).isoformat()
        card = Card(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            column=column,
            priority=priority,
            tags=tags or [],
            assignee=assignee,
            due_date=due_date,
            sprint=sprint,
            gh=gh,
            card_type=card_type,
            created_at=now,
            updated_at=now,
        )
        self._append_card(card)
        return card

    def update_card(self, card_id: str, **kwargs: Any) -> Card | None:
        board = self.load_board()
        for card in board.cards:
            if card.id == card_id:
                for key, value in kwargs.items():
                    if hasattr(card, key):
                        setattr(card, key, value)
                card.updated_at = datetime.now(UTC).isoformat()
                self._surgical_rewrite({card_id: self._card_line(card)})
                return card
        return None

    def delete_card(self, card_id: str) -> bool:
        return self._surgical_rewrite({card_id: None}) > 0

    def move_card(self, card_id: str, to_column: str) -> bool:
        board = self.load_board()
        for card in board.cards:
            if card.id == card_id:
                card.column = to_column
                card.updated_at = datetime.now(UTC).isoformat()
                self._surgical_rewrite({card_id: self._card_line(card)})
                return True
        return False

    def list_cards(self, column: str | None = None, assignee: str | None = None) -> list[Card]:
        board = self.load_board()
        cards = board.cards
        if column:
            cards = [c for c in cards if c.column == column]
        if assignee:
            cards = [c for c in cards if c.assignee == assignee]
        return cards

    def search_cards(self, query: str) -> list[Card]:
        q = query.lower()
        return [
            c
            for c in self.load_board().cards
            if q in c.title.lower()
            or q in c.description.lower()
            or any(q in t.lower() for t in c.tags)
        ]

    def archive_done(self) -> int:
        """Remove all cards in 'done' column. Returns count archived."""
        board = self.load_board()
        done_ids = [c.id for c in board.cards if c.column == "done"]
        archived = len(done_ids)
        if archived:
            self._surgical_rewrite(dict.fromkeys(done_ids))
        return archived

    def block_card(self, card_id: str, blocker_id: str) -> Card | None:
        board = self.load_board()
        for card in board.cards:
            if card.id == card_id:
                if blocker_id not in card.blocked_by:
                    card.blocked_by.append(blocker_id)
                    card.updated_at = datetime.now(UTC).isoformat()
                    self._surgical_rewrite({card_id: self._card_line(card)})
                return card
        return None

    def unblock_card(self, card_id: str, blocker_id: str) -> Card | None:
        board = self.load_board()
        for card in board.cards:
            if card.id == card_id:
                if blocker_id in card.blocked_by:
                    card.blocked_by.remove(blocker_id)
                    card.updated_at = datetime.now(UTC).isoformat()
                    self._surgical_rewrite({card_id: self._card_line(card)})
                return card
        return None

    def is_blocked(self, card_id: str) -> bool:
        card = self.get_card(card_id)
        return bool(card.blocked_by) if card else False

    def get_tags(self) -> dict[str, int]:
        tag_counts: dict[str, int] = {}
        for card in self.load_board().cards:
            for tag in card.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts

    def get_stats(self) -> dict[str, Any]:
        board = self.load_board()
        by_column: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        for card in board.cards:
            by_column[card.column] = by_column.get(card.column, 0) + 1
            by_priority[card.priority] = by_priority.get(card.priority, 0) + 1
        return {
            "total": len(board.cards),
            "byColumn": by_column,
            "byPriority": by_priority,
            "columns": len(board.columns),
        }

    # ── Notes ───────────────────────────────────────────────────────────

    def _read_notes(self) -> list[Note]:
        if not self._notes_file.exists():
            return []
        notes: list[Note] = []
        for raw in self._notes_file.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                notes.append(Note.from_dict(json.loads(raw)))
            except json.JSONDecodeError:
                continue
        return notes

    def _write_notes(self, notes: list[Note]) -> None:
        lines = [json.dumps(n.to_dict(), ensure_ascii=False) for n in notes]
        self._notes_file.write_text("\n".join(lines) + "\n" if lines else "")

    def list_notes(
        self,
        tag: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Note]:
        notes = self._read_notes()
        if tag:
            notes = [n for n in notes if tag in n.tags]
        if status:
            notes = [n for n in notes if n.status == status]
        return notes[:limit]

    def get_note(self, note_id: str) -> Note | None:
        for note in self._read_notes():
            if note.id == note_id:
                return note
        return None

    def create_note(
        self,
        title: str,
        body: str = "",
        status: str = "open",
        tags: list[str] | None = None,
        sprint: str = "",
        gh: str = "",
        assignee: str = "",
    ) -> Note:
        now = datetime.now(UTC).isoformat()
        note = Note(
            id=str(uuid.uuid4()),
            title=title,
            body=body,
            status=status,
            tags=tags or [],
            sprint=sprint,
            gh=gh,
            assignee=assignee,
            created_at=now,
            updated_at=now,
        )
        notes = self._read_notes()
        notes.append(note)
        self._write_notes(notes)
        return note

    def update_note(self, note_id: str, **kwargs: Any) -> Note | None:
        notes = self._read_notes()
        for i, note in enumerate(notes):
            if note.id == note_id:
                for key, value in kwargs.items():
                    if hasattr(note, key):
                        setattr(note, key, value)
                note.updated_at = datetime.now(UTC).isoformat()
                notes[i] = note
                self._write_notes(notes)
                return note
        return None

    def delete_note(self, note_id: str) -> bool:
        notes = self._read_notes()
        original_len = len(notes)
        notes = [n for n in notes if n.id != note_id]
        if len(notes) < original_len:
            self._write_notes(notes)
            return True
        return False

    def search_notes(self, query: str) -> list[Note]:
        q = query.lower()
        return [
            n
            for n in self._read_notes()
            if q in n.title.lower() or q in n.body.lower() or any(q in t.lower() for t in n.tags)
        ]

    # ── Sync ────────────────────────────────────────────────────────────

    def sync(
        self,
        *,
        repair: bool = False,
        dry_run: bool = False,
    ) -> tuple[int, int, int]:
        """Reconcile notes → board. Returns (added, updated, total).

        Unmappable note statuses (null / empty / unknown) carry no column
        information: no card is created for them and existing cards are
        left unchanged — never a silent revert to todo. With
        ``repair=True`` such cards are pessimistically reset to ``todo``
        instead: the explicit disaster-recovery path. ``dry_run=True``
        plans and reports without writing anything. The full outcome of
        the latest run is exposed as ``self.last_sync_report``.
        """
        notes = self.list_notes(limit=9999)
        board = self.load_board()
        existing = {c.title: c for c in board.cards}
        added = 0
        updated = 0
        reverted: list[str] = []

        for note in notes:
            col = config.resolve_column(note.status)
            title = note.title or "(untitled)"
            card = existing.get(title)

            if col is None:
                # No column information: never create, never silently move.
                if card is None:
                    continue
                if repair and card.column != "todo":
                    if not dry_run:
                        self.move_card(card.id, "todo")
                    reverted.append(card.id)
                continue

            if card is None:
                if not dry_run:
                    self.add_card(
                        title=title,
                        column=col,
                        tags=coerce_tags(note.tags),
                        description=note.body or "",
                        assignee=note.assignee or "",
                        sprint=note.sprint or "",
                        gh=note.gh or "",
                    )
                added += 1
                continue

            if card.column != col:
                if not dry_run:
                    self.move_card(card.id, col)
                updated += 1

            if note.assignee and card.assignee != note.assignee:
                if not dry_run:
                    self.update_card(card.id, assignee=note.assignee)
                updated += 1

        total = len(board.cards) + added if dry_run else len(self.load_board().cards)
        self.last_sync_report = SyncReport(
            added=added,
            updated=updated,
            total=total,
            reverted=tuple(reverted),
            repaired=repair,
            dry_run=dry_run,
        )
        return added, updated, total


# ── Module-level singleton ───────────────────────────────────────────────

_store: PlannerStore | None = None


def get_store(
    board_dir: Path | None = None,
    notes_dir: Path | None = None,
) -> PlannerStore:
    global _store
    if _store is None:
        _store = PlannerStore(board_dir=board_dir, notes_dir=notes_dir)
    return _store


def reset_store() -> None:
    global _store
    _store = None
