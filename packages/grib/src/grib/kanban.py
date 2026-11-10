"""
kanban — kanban board with notes, built into the grib package.

JSONL-backed board (the canonical ``.kanban/board.jsonl`` written by
PlannerStore, with the legacy whole-file ``.kanban/board.json`` as a
read-only fallback) with columns, cards, and per-card notes (comments).
Every operation is non-interactive and scriptable.

Usage as a module::

    from grib import KanbanStore

    store = KanbanStore()
    card = store.add_card("Fix boot order", column="todo", priority="high")
    store.move_card(card.id, "wip")
    store.add_note(card.id, "Started debugging the init script")

Usage as a CLI::

    grib board show
    grib board add "Fix boot order" --priority high
    grib board move <id> wip
    grib board delete <id>
    grib sync
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from . import config
from .ids import display_id, is_uuid_id, make_readable_id
from .store import PlannerStore, _token_of

logger = logging.getLogger("grib.kanban")

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


def _make_id(title: str, taken: set[str] | None = None) -> str:
    return make_readable_id(title, taken or ())


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
        # Seal-at-write (card 20261109_112756): any changed line invalidates
        # the chain — re-chain here so the GUI write path leaves
        # verify_chain() intact instead of waiting for a lucky `grib sync`.
        self._planner.compute_chains()

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


def cli_main(argv: list[str] | None = None) -> int:
    """Backward-compatible entry point delegating to the unified CLI."""
    from grib.cli import cli_main as unified_cli_main

    return unified_cli_main(argv)


if __name__ == "__main__":
    sys.exit(cli_main())
