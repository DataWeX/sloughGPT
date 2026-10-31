"""
kanban — kanban board with notes, built into the app-planner package.

JSONL-backed board (the canonical ``.kanban/board.jsonl`` written by
PlannerStore, with the legacy whole-file ``.kanban/board.json`` as a
read-only fallback) with columns, cards, and per-card notes (comments).
Every operation is non-interactive and scriptable.

Usage as a module::

    from app_planner import KanbanStore

    store = KanbanStore()
    card = store.add_card("Fix boot order", column="todo", priority="high")
    store.move_card(card.id, "wip")
    store.add_note(card.id, "Started debugging the init script")

Usage as a CLI::

    app-planner init
    app-planner add "Fix boot order" --priority high
    app-planner cards --column todo
    app-planner board
    app-planner move <id> wip
    app-planner note add <id> "Looking into it"
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from . import config
from .ids import display_id, is_uuid_id, make_readable_id
from .store import PlannerStore, _token_of

logger = logging.getLogger("app_planner.kanban")

_BOARD_FILE = "board.json"  # legacy whole-board file — read-only fallback
_LIVE_BOARD_FILE = "board.jsonl"  # canonical JSONL board (PlannerStore format)

DEFAULT_COLUMNS: list[dict[str, Any]] = [
    {"name": "todo", "wip_limit": 0, "order": 0},
    {"name": "wip", "wip_limit": 3, "order": 1},
    {"name": "review", "wip_limit": 0, "order": 2},
    {"name": "done", "wip_limit": 0, "order": 3},
]

PRIORITIES = ("low", "medium", "high", "critical")
PRIORITY_ICONS = {"low": " ", "medium": "!", "high": "!!", "critical": "!!!"}
PRIORITY_SORT = {"critical": 0, "high": 1, "medium": 2, "low": 3}

CARD_TYPES = ("task", "bug", "feature", "chore", "spike", "docs")
CARD_TYPE_ICONS = {
    "task": "☐",
    "bug": "🐛",
    "feature": "✨",
    "chore": "🔧",
    "spike": "🔍",
    "docs": "📝",
}


def _make_id(title: str, taken: set[str] | None = None) -> str:
    return make_readable_id(title, taken or ())


def _abbrev(s: str, max_len: int = 50) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s


@dataclass
class Note:
    """A comment attached to a card."""

    id: str = ""
    text: str = ""
    author: str = ""
    created_at: str = ""
    extra: dict = field(default_factory=dict)  # unknown keys, round-tripped losslessly

    def to_dict(self) -> dict:
        d = dict(self.extra)
        d.update(
            {
                "id": self.id,
                "text": self.text,
                "author": self.author,
                "created_at": self.created_at,
            }
        )
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Note:
        raw = dict(d)
        known: dict[str, Any] = {}
        for key in list(raw):
            target = key if key in _NOTE_FIELDS else _NOTE_ALIASES.get(key)
            if target is not None and target not in known:
                known[target] = raw.pop(key)
        return cls(
            id=str(known.get("id") or ""),
            text=str(known.get("text") or ""),
            author=str(known.get("author") or ""),
            created_at=str(known.get("created_at") or ""),
            extra=raw,
        )


_NOTE_FIELDS = ("id", "text", "author", "created_at")
_NOTE_ALIASES = {
    "body": "text",
    "user": "author",
    "ts": "created_at",
    "at": "created_at",
    "createdAt": "created_at",
}

_CARD_FIELDS = (
    "id",
    "title",
    "description",
    "column",
    "priority",
    "card_type",
    "effort",
    "tags",
    "created_at",
    "updated_at",
    "due_date",
    "assignee",
    "sprint",
    "blocked_by",
    "history",
)
_CARD_ALIASES = {
    "type": "card_type",
    "createdAt": "created_at",
    "updatedAt": "updated_at",
    "dueDate": "due_date",
}
_CARD_STR_FIELDS = (
    "id",
    "title",
    "description",
    "column",
    "priority",
    "card_type",
    "created_at",
    "updated_at",
    "due_date",
    "assignee",
    "sprint",
)


@dataclass
class Card:
    """A single kanban card (task)."""

    id: str = ""
    title: str = ""
    description: str = ""
    column: str = "todo"
    priority: str = "medium"
    card_type: str = "task"
    effort: float = 0.0
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    due_date: str = ""
    assignee: str = ""
    sprint: str = ""
    blocked_by: list[str] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    # Fields this legacy model doesn't know (gh, chain_*, root_hash, ...) are
    # kept here and written back on save, so GUI writes never destroy data
    # the live web board carries.
    extra: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return display_id(self.id)

    @property
    def priority_icon(self) -> str:
        return PRIORITY_ICONS.get(self.priority, " ")

    def to_dict(self) -> dict:
        d = dict(self.extra)
        d.update(
            {
                "id": self.id,
                "title": self.title,
                "description": self.description,
                "column": self.column,
                "priority": self.priority,
                "card_type": self.card_type,
                "effort": self.effort,
                "tags": list(self.tags),
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "due_date": self.due_date,
                "assignee": self.assignee,
                "sprint": self.sprint,
                "blocked_by": list(self.blocked_by),
                "notes": [n.to_dict() for n in self.notes],
                "history": list(self.history),
            }
        )
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Card:
        raw = dict(d)  # never mutate the caller's dict
        notes_raw = raw.pop("notes", None)
        if not isinstance(notes_raw, list):
            notes_raw = []
        known: dict[str, Any] = {}
        for key in list(raw):
            target = key if key in _CARD_FIELDS else _CARD_ALIASES.get(key)
            if target is not None and target not in known:
                known[target] = raw.pop(key)
        clean: dict[str, Any] = {}
        for key in _CARD_STR_FIELDS:
            if known.get(key) is not None:
                clean[key] = str(known[key])
        if not clean.get("column"):
            clean.pop("column", None)  # empty/missing → dataclass default
        if "effort" in known:
            try:
                clean["effort"] = float(known["effort"] or 0.0)
            except (TypeError, ValueError):
                pass
        for key in ("tags", "blocked_by", "history"):
            if key in known:
                clean[key] = list(known[key]) if isinstance(known[key], list) else []
        notes = [Note.from_dict(n) for n in notes_raw if isinstance(n, dict)]
        return cls(notes=notes, extra=raw, **clean)


@dataclass
class ColumnDef:
    name: str
    wip_limit: int = 0
    order: int = 0


@dataclass
class Board:
    name: str = "board"
    columns: list[ColumnDef] = field(
        default_factory=lambda: [ColumnDef(**c) for c in DEFAULT_COLUMNS]
    )
    cards: list[Card] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "columns": [asdict(c) for c in self.columns],
            "cards": [c.to_dict() for c in self.cards],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Board:
        columns: list[ColumnDef] = []
        for c in d.get("columns") or []:
            if not isinstance(c, dict) or not c.get("name"):
                continue
            try:
                wip_limit = int(c.get("wip_limit") or 0)
            except (TypeError, ValueError):
                wip_limit = 0
            try:
                order = int(c.get("order") or 0)
            except (TypeError, ValueError):
                order = 0
            columns.append(ColumnDef(name=str(c["name"]), wip_limit=wip_limit, order=order))
        if not columns:
            columns = [ColumnDef(**c) for c in DEFAULT_COLUMNS]
        cards = [
            Card.from_dict(c)
            for c in d.get("cards") or []
            if isinstance(c, dict) and c.get("id") and c.get("title")
        ]
        board = cls(name=str(d.get("name") or "board"), columns=columns, cards=cards)
        _coerce_columns(board)
        return board


def _coerce_columns(board: Board) -> None:
    """Downgrade undeclared column spellings to the fallback (read defense).

    Mirrors PlannerStore.load_board: the JSONL is hand-editable, so a retired
    or misspelled column must not yield a card that header-driven renders
    silently hide.
    """
    declared = [c.name for c in board.columns]
    if not declared:
        return
    declared_set = set(declared)
    fallback = "todo" if "todo" in declared_set else declared[0]
    for card in board.cards:
        if card.column not in declared_set:
            logger.warning(
                "Card %s has undeclared column %r — coercing to %r",
                card.id,
                card.column,
                fallback,
            )
            card.column = fallback


def _clone_board(board: Board) -> Board:
    """Snapshot of a board before mutation — save() diffs against it so it
    writes only the lines this caller changed (concurrent edits to other
    cards survive)."""
    return Board.from_dict(board.to_dict())


def _canonical_card(obj: dict, declared: set[str], fallback: str) -> Card:
    """Canonical parse of a raw card object under the given column vocabulary."""
    card = Card.from_dict(obj)
    if card.column not in declared:
        card.column = fallback
    return card


def _engine_speaks_board(board: Board) -> bool:
    """True when every column name is a KanbanEngine CardState.

    Mixed vocabularies must not be half-validated (a4-col board with a
    'review' column would otherwise enforce verify-gated transitions for
    review→done only, while todo→wip slipped through raw) — the engine
    applies to the whole board or not at all.
    """
    from planner.engine import CardState

    try:
        for col in board.columns:
            CardState(col.name)
    except ValueError:
        return False
    return True


def _card_line(card: Card) -> str:
    """Canonical, byte-stable JSONL line (same format PlannerStore writes)."""
    return json.dumps(card.to_dict(), sort_keys=True, ensure_ascii=False)


class _JSONBackend:
    """Board persistence over the canonical ``board.jsonl``.

    Reads the JSONL line-by-line (schema header → name/columns, card lines →
    tolerant :class:`Card` parse); falls back to the legacy whole-file
    ``board.json`` read-only, so the first save migrates a legacy board
    without ever touching the legacy file.

    ``save`` is a surgical, OCC-guarded diff built on PlannerStore's commit
    machinery: it writes only the lines the caller changed relative to
    *baseline* (or, when no baseline is given, every line that differs from
    the file), preserving every other line — header, foreign schema rows,
    even unparsable ones — byte-for-byte. Concurrent edits to other cards
    survive; same-card writes are last-writer-wins (the PlannerStore
    contract). With no baseline, an empty incoming board over a non-empty
    file is refused unless *allow_wipe* — that shape means a corrupt read or
    an unintended full-board replace, never a legitimate mutation.
    """

    def __init__(self, board_dir: Path):
        self._dir = board_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / _LIVE_BOARD_FILE
        self._legacy = self._dir / _BOARD_FILE
        self._planner = PlannerStore(board_dir=self._dir)

    @property
    def exists(self) -> bool:
        return self._path.exists() or self._legacy.exists()

    def load(self) -> Board:
        if self._path.exists():
            return self._load_jsonl()
        if self._legacy.exists():
            return self._load_legacy()
        return Board()

    def _load_jsonl(self) -> Board:
        name = "board"
        columns: list | None = None
        card_objs: list[dict] = []
        for raw in self._path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("Skipping unparsable line in %s", self._path.name)
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("schema") == "planner/1" and obj.get("columns"):
                name = obj.get("name") or "board"
                columns = obj["columns"]
            elif obj.get("id") and obj.get("title"):
                card_objs.append(obj)
        return Board.from_dict(
            {"name": name, "columns": columns or DEFAULT_COLUMNS, "cards": card_objs}
        )

    def _load_legacy(self) -> Board:
        try:
            return Board.from_dict(json.loads(self._legacy.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            logger.warning("Corrupt legacy board file %s: %s — starting fresh", self._legacy, exc)
            return Board()

    def save(
        self,
        board: Board,
        *,
        baseline: Board | None = None,
        allow_wipe: bool = False,
    ) -> None:
        self._planner._occ(lambda: self._attempt_save(board, baseline, allow_wipe))

    def _attempt_save(
        self,
        board: Board,
        baseline: Board | None,
        allow_wipe: bool,
    ) -> None:
        raw = self._path.read_text(encoding="utf-8") if self._path.exists() else ""
        token = _token_of(raw)
        declared = {c.name for c in board.columns}
        fallback = (
            "todo" if "todo" in declared else (board.columns[0].name if board.columns else "todo")
        )

        lines = raw.splitlines()
        parsed: list[dict | None] = []
        file_ids: set[str] = set()
        file_canon: dict[str, dict] = {}
        header_idx = -1
        for i, text in enumerate(lines):
            obj: dict | None = None
            stripped = text.strip()
            if stripped:
                try:
                    loaded = json.loads(stripped)
                    obj = loaded if isinstance(loaded, dict) else None
                except json.JSONDecodeError:
                    obj = None
            parsed.append(obj)
            if obj is None:
                continue
            if obj.get("schema") == "planner/1" and obj.get("columns") and header_idx < 0:
                header_idx = i
            elif obj.get("id") and obj.get("title"):
                fid = str(obj["id"])
                if fid not in file_ids:
                    file_ids.add(fid)
                    file_canon[fid] = _canonical_card(obj, declared, fallback).to_dict()

        incoming = {c.id: c for c in board.cards}
        if baseline is not None and not raw.strip():
            # First save into a missing/empty live file (legacy board.json
            # migration): there is no jsonl for "unchanged" to diff against —
            # every incoming card must land.
            baseline = None
        if baseline is None:
            if not incoming and file_ids and not allow_wipe:
                raise RuntimeError(
                    f"refusing to clear {len(file_ids)} card(s) from {self._path.name} "
                    "(pass allow_wipe for an explicit full-board replace)"
                )
            write_ids = {
                cid
                for cid, card in incoming.items()
                if cid not in file_canon or file_canon[cid] != card.to_dict()
            }
            delete_ids = file_ids - set(incoming)
        else:
            base_map = {c.id: c for c in baseline.cards}
            write_ids = {
                cid
                for cid, card in incoming.items()
                if cid not in base_map or base_map[cid].to_dict() != card.to_dict()
            }
            delete_ids = set(base_map) - set(incoming)

        header_line = PlannerStore._schema_line(board.name, [asdict(c) for c in board.columns])
        if header_idx < 0:
            new_header: str | None = header_line
        elif parsed[header_idx] != json.loads(header_line):
            new_header = header_line
        else:
            new_header = None

        out: list[str] = []
        for i, text in enumerate(lines):
            if i == header_idx:
                out.append(new_header or text)
                continue
            obj = parsed[i]
            if obj is None:
                out.append(text)
                continue
            if obj.get("id") and obj.get("title"):
                cid = str(obj["id"])
                if cid in delete_ids:
                    continue
                if cid in write_ids:
                    out.append(_card_line(incoming[cid]))
                    continue
            out.append(text)
        for cid in write_ids:
            if cid not in file_ids:
                out.append(_card_line(incoming[cid]))
        if header_idx < 0 and new_header is not None:
            out.insert(0, new_header)

        new_text = "\n".join(out) + "\n" if out else ""
        if new_text != raw:
            self._planner._atomic_write(new_text, expect=token)


class KanbanStore:
    """Kanban board store — all operations are non-interactive."""

    def __init__(self, board_dir: str | Path | None = None):
        self._dir = Path(board_dir) if board_dir else config.default_board_dir()
        self._bk = _JSONBackend(self._dir)

    def init_board(self, name: str = "board", force: bool = False) -> Board:
        if self._bk.exists and not force:
            return self._bk.load()
        board = Board(name=name)
        self._bk.save(board, allow_wipe=force)
        return board

    def load_board(self) -> Board:
        return self._bk.load()

    def _find_one(self, board: Board, prefix: str) -> Card | None:
        lower = prefix.lower()
        matches = [c for c in board.cards if c.id.lower().startswith(lower)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            matches.sort(
                key=lambda c: c.updated_at or c.created_at or c.id,
                reverse=True,
            )
            chosen = matches[0]
            logger.warning(
                "Ambiguous id '%s' (%d matches); using most recently updated: %s",
                prefix,
                len(matches),
                chosen.id,
            )
            return chosen
        return None

    def add_card(
        self,
        title: str,
        column: str = "",
        priority: str = "medium",
        card_type: str = "task",
        effort: float = 0.0,
        sprint: str = "",
        description: str = "",
        tags: list[str] | None = None,
        due_date: str = "",
        assignee: str = "",
    ) -> Card:
        board = self.load_board()
        baseline = _clone_board(board)
        column = column or (board.columns[0].name if board.columns else "todo")
        now = datetime.now(UTC).isoformat()
        card = Card(
            id=_make_id(title),
            title=title,
            description=description,
            column=column,
            priority=priority if priority in PRIORITIES else "medium",
            card_type=card_type if card_type in CARD_TYPES else "task",
            effort=effort,
            sprint=sprint,
            tags=tags or [],
            created_at=now,
            updated_at=now,
            due_date=due_date,
            assignee=assignee,
        )
        board.cards.insert(0, card)
        self._bk.save(board, baseline=baseline)
        return card

    def get_card(self, card_id: str) -> Card | None:
        board = self.load_board()
        return self._find_one(board, card_id)

    def update_card(self, card_id: str, **kwargs: Any) -> Card | None:
        board = self.load_board()
        baseline = _clone_board(board)
        card = self._find_one(board, card_id)
        if card is None:
            return None
        changes = {}
        for key in (
            "title",
            "description",
            "priority",
            "card_type",
            "effort",
            "sprint",
            "tags",
            "due_date",
            "assignee",
            "blocked_by",
        ):
            if key in kwargs and kwargs[key] is not None:
                old = getattr(card, key)
                new = kwargs[key]
                if old != new:
                    changes[key] = {"old": old, "new": new}
                    setattr(card, key, new)
        if changes:
            card.history.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "changes": changes,
                }
            )
            card.updated_at = datetime.now(UTC).isoformat()
            self._bk.save(board, baseline=baseline)
        return card

    def block_card(self, card_id: str, blocker_id: str) -> Card | None:
        board = self.load_board()
        baseline = _clone_board(board)
        card = self._find_one(board, card_id)
        if card is None:
            return None
        if blocker_id not in card.blocked_by:
            card.blocked_by.append(blocker_id)
            card.updated_at = datetime.now(UTC).isoformat()
            self._bk.save(board, baseline=baseline)
        return card

    def unblock_card(self, card_id: str, blocker_id: str) -> Card | None:
        board = self.load_board()
        baseline = _clone_board(board)
        card = self._find_one(board, card_id)
        if card is None:
            return None
        if blocker_id in card.blocked_by:
            card.blocked_by.remove(blocker_id)
            card.updated_at = datetime.now(UTC).isoformat()
            self._bk.save(board, baseline=baseline)
        return card

    def is_blocked(self, card_id: str) -> bool:
        board = self.load_board()
        card = self._find_one(board, card_id)
        return bool(card.blocked_by) if card else False

    def delete_card(self, card_id: str) -> bool:
        board = self.load_board()
        baseline = _clone_board(board)
        card = self._find_one(board, card_id)
        if card is None:
            return False
        board.cards.remove(card)
        self._bk.save(board, baseline=baseline)
        return True

    def move_card(
        self,
        card_id: str,
        column: str,
        validate: bool = True,
        actor: str = "",
        force: bool = False,
    ) -> Card | None:
        """Move *card_id* to *column*; return the card, or ``None``.

        When validate=True (default) AND the board speaks the KanbanEngine
        vocabulary (every column is a CardState: inbox/ready/doing/review/
        verify/done/blocked), the transition goes through the engine's state
        machine: legal-transition check + WIP enforcement (returns None when
        rejected). Boards on the legacy column vocabulary (todo/wip/...) fall
        back to raw column assignment — the engine cannot reason about
        states it does not know, and rejecting every move would break the
        running board; mapping the vocabularies is a product decision, not
        this store's.

        An unknown column always fails (board integrity), regardless of
        *validate* — that flag is reserved for engine-owned transition/WIP
        rules; the store itself enforces column existence only. *actor*
        records a history entry on the card; *force* is accepted for
        engine-path parity (nothing store-side to override).
        """
        board = self.load_board()
        baseline = _clone_board(board)
        col_names = {c.name for c in board.columns}
        if column not in col_names:
            logger.warning("Unknown column '%s'. Known: %s", column, sorted(col_names))
            return None
        card = self._find_one(board, card_id)
        if card is None:
            return None
        if validate and _engine_speaks_board(board):
            from planner.engine import KanbanEngine

            cols_dict = [
                {"name": c.name, "wip_limit": c.wip_limit, "order": c.order} for c in board.columns
            ]
            cards_dict = [{"id": c.id, "column": c.column, "title": c.title} for c in board.cards]
            result = KanbanEngine().transition(
                cols_dict,
                cards_dict,
                card.id,
                column,
                actor=actor,
                force=force,
            )
            if not result.ok:
                return None
        card.column = column
        card.updated_at = datetime.now(UTC).isoformat()
        if actor:
            card.history.append(
                {"actor": actor, "action": "move", "to": column, "at": card.updated_at}
            )
        self._bk.save(board, baseline=baseline)
        return card

    def list_cards(
        self,
        column: str | None = None,
        priority: str | None = None,
        tag: str | None = None,
        assignee: str | None = None,
        due_before: str = "",
        due_after: str = "",
        overdue: bool = False,
        limit: int = 100,
    ) -> list[Card]:
        board = self.load_board()
        results: list[Card] = []
        today = date.today().isoformat()
        for card in board.cards:
            if column and card.column != column:
                continue
            if priority and card.priority != priority:
                continue
            if tag and tag not in card.tags:
                continue
            if assignee and card.assignee != assignee:
                continue
            if due_before and (not card.due_date or card.due_date > due_before):
                continue
            if due_after and (not card.due_date or card.due_date < due_after):
                continue
            if overdue and (not card.due_date or card.due_date >= today):
                continue
            results.append(card)
            if len(results) >= limit:
                break
        return results

    def search_cards(self, query: str, limit: int = 30) -> list[Card]:
        q = query.lower()
        board = self.load_board()
        return [
            c
            for c in board.cards
            if q in c.title.lower()
            or q in c.description.lower()
            or q in " ".join(c.tags).lower()
            or q in c.assignee.lower()
        ][:limit]

    def archive_done(self) -> int:
        board = self.load_board()
        baseline = _clone_board(board)
        before = len(board.cards)
        board.cards = [c for c in board.cards if c.column != "done"]
        removed = before - len(board.cards)
        if removed:
            self._bk.save(board, baseline=baseline)
        return removed

    def add_note(self, card_id: str, text: str, author: str = "") -> Note | None:
        board = self.load_board()
        baseline = _clone_board(board)
        card = self._find_one(board, card_id)
        if card is None:
            return None
        now = datetime.now(UTC).isoformat()
        note = Note(
            id=_make_id(f"note-{text[:40]}", {n.id for n in card.notes}),
            text=text,
            author=author,
            created_at=now,
        )
        card.notes.append(note)
        card.updated_at = now
        self._bk.save(board, baseline=baseline)
        return note

    def list_notes(self, card_id: str) -> list[Note] | None:
        board = self.load_board()
        card = self._find_one(board, card_id)
        return card.notes if card else None

    def delete_note(self, card_id: str, note_id: str) -> bool:
        board = self.load_board()
        baseline = _clone_board(board)
        card = self._find_one(board, card_id)
        if card is None:
            return False
        before = len(card.notes)
        card.notes = [
            n
            for n in card.notes
            if not (n.id == note_id or (is_uuid_id(n.id) and n.id[:8] == note_id))
        ]
        if len(card.notes) == before:
            return False
        card.updated_at = datetime.now(UTC).isoformat()
        self._bk.save(board, baseline=baseline)
        return True

    def list_columns(self) -> list[ColumnDef]:
        return self.load_board().columns

    def add_column(self, name: str, wip_limit: int = 0) -> ColumnDef | None:
        board = self.load_board()
        baseline = _clone_board(board)
        if any(c.name == name for c in board.columns):
            logger.warning("Column '%s' already exists", name)
            return None
        col = ColumnDef(name=name, wip_limit=wip_limit, order=len(board.columns))
        board.columns.append(col)
        self._bk.save(board, baseline=baseline)
        return col

    def rename_column(self, name: str, new_name: str) -> bool:
        board = self.load_board()
        baseline = _clone_board(board)
        for col in board.columns:
            if col.name == name:
                col.name = new_name
                for card in board.cards:
                    if card.column == name:
                        card.column = new_name
                self._bk.save(board, baseline=baseline)
                return True
        return False

    def remove_column(self, name: str, move_to: str = "todo") -> bool:
        board = self.load_board()
        baseline = _clone_board(board)
        before = len(board.columns)
        board.columns = [c for c in board.columns if c.name != name]
        if len(board.columns) == before:
            return False
        for card in board.cards:
            if card.column == name:
                card.column = move_to
        self._bk.save(board, baseline=baseline)
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
    headers = [
        f"  {c.name.upper()}" + (f" (WIP:{c.wip_limit})" if c.wip_limit else "")
        for c in cols_sorted
    ]

    by_col: dict[str, list[Card]] = {c.name: [] for c in board.columns}
    for card in board.cards:
        by_col.setdefault(card.column, []).append(card)

    for col_name in by_col:
        by_col[col_name].sort(key=lambda c: PRIORITY_SORT.get(c.priority, 99))

    sep = "─" * (sum(col_width for _ in cols_sorted) + len(cols_sorted) + 1)
    lines.append(f"┌{sep}┐")
    lines.append(
        f"│{'│'.join(f'{h:^{col_width}}' for h, cw in zip(headers, [col_width] * len(cols_sorted)))}│"
    )
    lines.append(f"├{sep}┤")

    today = date.today().isoformat()
    max_rows = max((len(cards) for cards in by_col.values()), default=1)
    for row_idx in range(max_rows):
        cells = []
        for col in cols_sorted:
            cards = by_col.get(col.name, [])
            if row_idx < len(cards):
                c = cards[row_idx]
                type_icon = CARD_TYPE_ICONS.get(c.card_type, "☐")
                due = f" [{c.due_date}]" if c.due_date else ""
                overdue = (
                    " !OVERDUE" if c.due_date and c.due_date < today and c.column != "done" else ""
                )
                blocked = " BLOCKED" if c.blocked_by else ""
                assign = f" @{c.assignee}" if c.assignee else ""
                tags = ""
                if c.tags:
                    tags = " #" + ",".join(c.tags[:2])
                    if len(c.tags) > 2:
                        tags += "+"
                notes = f" ({len(c.notes)})" if c.notes else ""
                effort = f" {c.effort}pts" if c.effort else ""
                sprint = f" [{c.sprint}]" if c.sprint else ""
                label = f"{c.priority_icon} {type_icon} {c.short_id} {_abbrev(c.title, col_width - 17)}{due}{overdue}{blocked}{assign}{effort}{sprint}{tags}{notes}"
                cells.append(f" {label:<{col_width - 1}}")
            else:
                cells.append(" " * col_width)
        lines.append(
            f"│{'│'.join(f'{c:<{col_width}}' for c, cw in zip(cells, [col_width] * len(cols_sorted)))}│"
        )

    lines.append(f"└{sep}┘")
    lines.append(f"  {len(board.cards)} card(s) across {len(board.columns)} column(s)")
    return "\n".join(lines)


def cli_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="app-planner", description="Kanban board with notes")
    parser.add_argument("--dir", default=None, help="Board directory (default: config/env)")
    sub = parser.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init", help="Initialize a new board")
    p_init.add_argument("--name", default="board", help="Board name")
    p_init.add_argument("--force", action="store_true", help="Recreate if exists")

    p_add = sub.add_parser("add", help="Add a card")
    p_add.add_argument("title", help="Card title")
    p_add.add_argument("--column", default="", help="Target column")
    p_add.add_argument("--priority", default="medium", choices=PRIORITIES)
    p_add.add_argument("--type", default="task", choices=CARD_TYPES, help="Card type")
    p_add.add_argument("--effort", type=float, default=0.0, help="Effort estimate (story points)")
    p_add.add_argument("--sprint", default="", help="Sprint identifier")
    p_add.add_argument("--desc", default="", help="Description")
    p_add.add_argument("--tags", default="", help="Comma-separated tags")
    p_add.add_argument("--due", default="", help="Due date (ISO)")
    p_add.add_argument("--assignee", default="", help="Assignee")

    p_list = sub.add_parser("list", help="List cards")
    p_list.add_argument("--column", default=None, help="Filter by column")
    p_list.add_argument("--priority", default=None, choices=PRIORITIES)
    p_list.add_argument("--tag", default=None, help="Filter by tag")
    p_list.add_argument("--assignee", default=None, help="Filter by assignee")
    p_list.add_argument("--due-before", default="", help="Due before YYYY-MM-DD")
    p_list.add_argument("--due-after", default="", help="Due after YYYY-MM-DD")
    p_list.add_argument("--overdue", action="store_true", help="Only overdue cards")
    p_list.add_argument("--limit", type=int, default=50)

    p_show = sub.add_parser("show", help="Show card details")
    p_show.add_argument("card_id", help="Card id or prefix")

    p_edit = sub.add_parser("edit", help="Edit a card")
    p_edit.add_argument("card_id", help="Card id or prefix")
    p_edit.add_argument("--title", default=None)
    p_edit.add_argument("--desc", default=None)
    p_edit.add_argument("--priority", default=None, choices=PRIORITIES)
    p_edit.add_argument("--type", default=None, choices=CARD_TYPES)
    p_edit.add_argument("--effort", type=float, default=None, help="Effort estimate")
    p_edit.add_argument("--sprint", default=None, help="Sprint identifier")
    p_edit.add_argument("--tags", default=None)
    p_edit.add_argument("--due", default=None)
    p_edit.add_argument("--assignee", default=None)

    p_move = sub.add_parser("move", help="Move card to another column")
    p_move.add_argument("card_id", help="Card id or prefix")
    p_move.add_argument("column", help="Target column")

    p_block = sub.add_parser("block", help="Block a card by another card")
    p_block.add_argument("card_id", help="Card id or prefix to block")
    p_block.add_argument("blocker_id", help="Blocker card id or prefix")

    p_unblock = sub.add_parser("unblock", help="Unblock a card")
    p_unblock.add_argument("card_id", help="Card id or prefix")
    p_unblock.add_argument("blocker_id", help="Blocker card id or prefix")

    sub.add_parser("blocked", help="List blocked cards")

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

    p_export = sub.add_parser("export", help="Export board to JSON")
    p_export.add_argument("--output", default="", help="Output file path")

    p_import = sub.add_parser("import", help="Import board from JSON")
    p_import.add_argument("input", help="Input JSON file path")

    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 1

    store = get_kanban_store(board_dir=args.dir)

    if args.cmd == "init":
        board = store.init_board(name=args.name, force=args.force)
        print(f"Board '{board.name}' at {store._dir / _LIVE_BOARD_FILE}")
        cols = ", ".join(c.name for c in board.columns)
        print(f"  {len(board.columns)} columns: {cols}")
        return 0

    if args.cmd == "add":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        card = store.add_card(
            args.title,
            column=args.column,
            priority=args.priority,
            card_type=args.type,
            effort=args.effort,
            sprint=args.sprint,
            description=args.desc,
            tags=tags,
            due_date=args.due,
            assignee=args.assignee,
        )
        effort_s = f"  effort={args.effort}" if args.effort else ""
        sprint_s = f"  sprint={args.sprint}" if args.sprint else ""
        print(
            f"Added: {card.short_id}  {card.title}  [{card.column}]  {card.priority_icon}{effort_s}{sprint_s}"
        )
        return 0

    if args.cmd == "list":
        cards = store.list_cards(
            column=args.column,
            priority=args.priority,
            tag=args.tag,
            assignee=args.assignee,
            due_before=args.due_before,
            due_after=args.due_after,
            overdue=args.overdue,
            limit=args.limit,
        )
        if not cards:
            print("No cards found.")
            return 0
        today = date.today().isoformat()
        for c in cards:
            tags_s = f"  [{', '.join(c.tags)}]" if c.tags else ""
            due_s = f"  [{c.due_date}]" if c.due_date else ""
            overdue_s = (
                "  OVERDUE" if c.due_date and c.due_date < today and c.column != "done" else ""
            )
            assign_s = f"  @{c.assignee}" if c.assignee else ""
            effort_s = f"  {c.effort}pts" if c.effort else ""
            sprint_s = f"  [{c.sprint}]" if c.sprint else ""
            notes_s = f"  ({len(c.notes)})" if c.notes else ""
            print(
                f"  {c.priority_icon} {c.short_id}  {c.title}  [{c.column}]{tags_s}{due_s}{overdue_s}{assign_s}{effort_s}{sprint_s}{notes_s}"
            )
        print(f"\n  {len(cards)} card(s)")
        return 0

    if args.cmd == "show":
        card = store.get_card(args.card_id)
        if card is None:
            print(f"Card not found: {args.card_id}")
            return 1
        tags_s = ", ".join(card.tags) if card.tags else "none"
        blocked_s = ", ".join(card.blocked_by) if card.blocked_by else "none"
        type_icon = CARD_TYPE_ICONS.get(card.card_type, "☐")
        effort_s = f"  effort:   {card.effort} pts" if card.effort else "  effort:   0"
        sprint_s = f"  sprint:   {card.sprint}" if card.sprint else ""
        print(f"  {card.priority_icon} {type_icon}  {card.title}")
        print(f"  id:       {card.id}")
        print(f"  type:     {card.card_type}")
        print(f"  column:   {card.column}")
        print(f"  priority: {card.priority}")
        print(f"  tags:     {tags_s}")
        print(f"  assignee: {card.assignee or 'unassigned'}")
        print(f"  due:      {card.due_date or 'none'}")
        print(f"  blocked:  {blocked_s}")
        print(effort_s)
        if sprint_s:
            print(sprint_s)
        print(f"  created:  {card.created_at}")
        print(f"  updated:  {card.updated_at}")
        if card.history:
            print(f"\n  History ({len(card.history)} changes):")
            for h in card.history[-5:]:
                ts = h["timestamp"][:19].replace("T", " ")
                changes = ", ".join(f"{k}={v['new']}" for k, v in h["changes"].items())
                print(f"    [{ts}] {changes}")
        if card.description:
            print(f"\n  {card.description}")
        if card.notes:
            print(f"\n  Notes ({len(card.notes)}):")
            for n in card.notes:
                author_s = f"  [{n.author}]" if n.author else ""
                print(f"    [{display_id(n.id)}]  {n.created_at[:10]}{author_s}")
                for line in n.text.split("\n"):
                    print(f"      {line}")
        return 0

    if args.cmd == "edit":
        kwargs: dict[str, Any] = {}
        if args.title is not None:
            kwargs["title"] = args.title
        if args.desc is not None:
            kwargs["description"] = args.desc
        if args.priority is not None:
            kwargs["priority"] = args.priority
        if args.type is not None:
            kwargs["card_type"] = args.type
        if args.effort is not None:
            kwargs["effort"] = args.effort
        if args.sprint is not None:
            kwargs["sprint"] = args.sprint
        if args.tags is not None:
            kwargs["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
        if args.due is not None:
            kwargs["due_date"] = args.due
        if args.assignee is not None:
            kwargs["assignee"] = args.assignee
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

    if args.cmd == "block":
        card = store.block_card(args.card_id, args.blocker_id)
        if card is None:
            print(f"Card not found: {args.card_id}")
            return 1
        print(f"Blocked: {card.short_id}  {card.title}  by {display_id(args.blocker_id)}")
        return 0

    if args.cmd == "unblock":
        card = store.unblock_card(args.card_id, args.blocker_id)
        if card is None:
            print(f"Card not found: {args.card_id}")
            return 1
        print(f"Unblocked: {card.short_id}  {card.title}")
        return 0

    if args.cmd == "blocked":
        cards = store.list_cards(limit=9999)
        blocked = [c for c in cards if c.blocked_by]
        if not blocked:
            print("No blocked cards.")
            return 0
        for c in blocked:
            blockers = ", ".join(display_id(b) for b in c.blocked_by)
            print(f"  {c.short_id}  {c.title}  [{c.column}]  blocked by: {blockers}")
        print(f"\n  {len(blocked)} blocked card(s)")
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
            print(f"Note added: [{display_id(note.id)}]" if note else "Card not found.")
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
                print(f"  [{display_id(n.id)}]  {n.created_at[:10]}{author_s}")
                print(f"    {n.text}")
            return 0
        if args.note_cmd in ("delete", "rm"):
            print(
                "Note deleted."
                if store.delete_note(args.card_id, args.note_id)
                else "Note not found."
            )
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
        print(
            "Removed."
            if store.remove_column(args.name, move_to=args.move_to)
            else "Column not found."
        )
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
        print("  Priorities:")
        for pri, count in s["priorities"]:
            print(f"    {pri:10s}  {count}")
        return 0

    if args.cmd == "export":
        board = store.load_board()
        path = args.output or (store._dir / f"{board.name}_export.json")
        path.write_text(json.dumps(board.to_dict(), indent=2, default=str))
        print(f"Exported {len(board.cards)} cards to {path}")
        return 0

    if args.cmd == "import":
        path = Path(args.input)
        if not path.exists():
            print(f"File not found: {path}")
            return 1
        try:
            data = json.loads(path.read_text())
            board = Board.from_dict(data)
        except Exception as exc:
            print(f"Import failed: {exc}")
            return 1
        store._bk.save(board, allow_wipe=True)
        print(f"Imported {len(board.cards)} cards from {path}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
