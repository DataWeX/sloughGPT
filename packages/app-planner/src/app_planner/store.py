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
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import config
from .atomic import atomic_write_text
from .slot_chain import apply_board

logger = logging.getLogger("app_planner.store")


# ── Board integrity ──────────────────────────────────────────────────────


class BoardIntegrityError(RuntimeError):
    """A board write would have introduced a duplicate card id."""


def _card_ids(text: str) -> list[str]:
    """Card ids present in *text*, in file order.

    Skips blank lines, the schema header (it carries no ``id``) and any line
    that is not a JSON object — so an unparsable line is never mistaken for a
    card.
    """
    ids: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("id"):
            ids.append(obj["id"])
    return ids


def _duplicate_ids(ids: list[str]) -> set[str]:
    """Ids seen more than once."""
    seen: set[str] = set()
    dups: set[str] = set()
    for card_id in ids:
        if card_id in seen:
            dups.add(card_id)
        seen.add(card_id)
    return dups


def duplicate_ids(board_file: Path) -> set[str]:
    """Ids carried by more than one line of *board_file*. Read-only.

    This is the check the live board fails: it carries 11 such groups, all
    pre-existing (see ``dedupe_board``).
    """
    if not board_file.exists():
        return set()
    return _duplicate_ids(_card_ids(board_file.read_text()))


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
            tags=d.get("tags", []),
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
            tags=d.get("tags", []),
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
        """Write the board durably and atomically (tmp -> fsync -> replace).

        Every board mutation funnels through here — ``_append_card``,
        ``_surgical_rewrite`` and ``_write_board`` alike — so this is the one
        place where card-id uniqueness can actually be enforced, and the one
        place a future regression has to get past.

        The rule is a **ratchet**, not a ban: a write may not introduce an id
        that is not *already* duplicated on disk. The live board carries 11
        pre-existing duplicate-id groups written by an older code path
        (``_surgical_rewrite`` once matched every line for an id; fixed in
        72188d8de, which also recorded the groups). Banning duplicates
        outright would refuse every board write on this machine and wedge all
        sessions — worse than the corruption it is meant to stop. Grandfathering
        what is already there lets those cards still be read, moved, repaired
        and rewritten while making it impossible to add a 12th group.

        The check re-reads the file, so it also catches a duplicate introduced
        between this call's caller reading and writing.
        """
        self._assert_no_new_duplicates(text)
        atomic_write_text(self._board_file, text)
        apply_board(self._board_file)  # one line: re-derive slot chain on every board write

    def _assert_no_new_duplicates(self, text: str) -> None:
        """Raise if *text* carries a duplicate id the file does not already have."""
        on_disk = self._board_file.read_text() if self._board_file.exists() else ""
        grandfathered = _duplicate_ids(_card_ids(on_disk))
        fresh = sorted(_duplicate_ids(_card_ids(text)) - grandfathered)
        if fresh:
            raise BoardIntegrityError(
                f"refusing to write {self._board_file}: {len(fresh)} card id(s) "
                f"would appear on more than one line: {', '.join(fresh)}. "
                "Board ids must be unique — this is a bug in the calling path, "
                "not a board the user can repair."
            )

    def duplicate_board_ids(self) -> set[str]:
        """Ids carried by more than one line of this board. Read-only."""
        return duplicate_ids(self._board_file)

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

        At most ONE line is rewritten per id: the first one encountered. The
        callers (``move_card``/``update_card``/``block_card``/``delete_card``)
        all select their target via ``load_board()``, which returns the first
        card line for a given id — so the write must land on that same line.
        Matching *every* line carrying the id makes a duplicate-id group
        collapse into N copies of one card, destroying the siblings. The board
        currently carries 13 such groups; see the dedupe card.

        Returns:
            Number of ids resolved to a line (at most ``len(replacements)``).
        """
        if not self._board_file.exists():
            return 0
        old_text = self._board_file.read_text()
        out: list[str] = []
        matched = 0
        seen: set[Any] = set()
        for raw in old_text.splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                out.append(raw)
                continue
            if not isinstance(obj, dict):
                out.append(raw)
                continue
            card_id = obj.get("id")
            # First-match only: keep the write on the line load_board() read.
            if card_id in replacements and card_id not in seen:
                seen.add(card_id)
                new_line = replacements[card_id]
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
        skipped = 0
        for raw in self._notes_file.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                notes.append(Note.from_dict(json.loads(raw)))
            except json.JSONDecodeError:
                # A torn/corrupt line is kept on disk (evidence) but must
                # never be dropped silently — report it every read.
                skipped += 1
        if skipped:
            logger.warning(
                "notes journal %s: %d corrupt line(s) skipped on read",
                self._notes_file,
                skipped,
            )
        return notes

    def _write_notes(self, notes: list[Note]) -> None:
        lines = [json.dumps(n.to_dict(), ensure_ascii=False) for n in notes]
        atomic_write_text(self._notes_file, "\n".join(lines) + "\n" if lines else "")

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

    def sync(self) -> tuple[int, int, int]:
        """Reconcile notes ↔ board. Returns (added, updated, total).

        Cards are matched to notes by title, but the match is **positional**:
        the Nth note with a given title maps to the Nth card with that title.
        Without that, two notes sharing a title (duplicate-title twins) both
        resolve to the same card and each issues move_card() against it, so the
        last one processed wins and every subsequent run reports the card as
        moved again — sync never converged.

        Positional matching also stops the first run from collapsing the twins
        onto a single card: the second twin finds no card left at its index and
        gets its own. The board is append-only, so card order is stable and the
        same note claims the same card run over run — that is what makes a
        converged board stay converged.

        The index is positional rather than keyed on card.id because legacy
        boards derive card ids from the title plus a seconds-resolution
        timestamp (`kanban._make_id`), so two same-title cards created in the
        same second share an id and could not be told apart.

        Cards are read via a snapshot taken at the start of each run; the
        returned Card objects must not be reused after move_card(), which
        reloads the board and would leave their column stale.
        """
        notes = self.list_notes(limit=9999)
        board = self.load_board()
        by_title: dict[str, list[Card]] = {}
        for c in board.cards:
            by_title.setdefault(c.title, []).append(c)

        added = 0
        updated = 0
        cursor: dict[str, int] = {}

        for note in notes:
            col = config.STATUS_TO_COLUMN.get((note.status or "").lower(), "todo")
            title = note.title or "(untitled)"
            pool = by_title.get(title, ())
            idx = cursor.get(title, 0)
            cursor[title] = idx + 1
            card = pool[idx] if idx < len(pool) else None

            if card is None:
                card = self.add_card(
                    title=title,
                    column=col,
                    tags=list(note.tags or []),
                    description=note.body or "",
                    assignee=note.assignee or "",
                    sprint=note.sprint or "",
                    gh=note.gh or "",
                )
                by_title.setdefault(title, []).append(card)
                added += 1
                continue

            if card.column != col:
                self.move_card(card.id, col)
                updated += 1

            if note.assignee and card.assignee != note.assignee:
                self.update_card(card.id, assignee=note.assignee)
                updated += 1

        total = len(self.load_board().cards)
        return added, updated, total


def dedupe_board(board_file: Path, apply: bool = False) -> dict[str, Any]:
    """Plan the removal of duplicate-id card lines; write only with ``apply=True``.

    Returns ``{"groups", "removed", "applied", "details", "text_before",
    "text_after"}``; each detail carries ``{"id", "lines", "keep_line",
    "keep_updated_at"}`` so a reviewer can see exactly what would go.

    Two properties are deliberate:

    * **Position** — the first occurrence keeps its slot. ``sync()`` maps the
      Nth note to the Nth card *by title position*, so deleting a line shifts
      every card after it. Keeping the first slot bounds the damage to
      everything below the removed copies; run ``app-planner sync`` afterwards
      and it re-converges (it is idempotent once positions settle).
    * **Bytes, not ``Card``** — the surviving line is copied raw. Legacy lines
      carry ``notes``/``note_hashes``/``slot_*`` fields that ``Card`` does not
      model, so round-tripping through the dataclass would silently strip note
      history and slot-chain metadata off the very cards being repaired.

    The newest copy wins on ``updated_at``, ties going to the first line.

    This is a data migration and is therefore never implicit: nothing runs
    without ``apply=True``, and the write goes through ``_atomic_write`` so it
    re-derives the slot chain and is checked by the same id ratchet.
    """
    if not board_file.exists():
        return {"groups": 0, "removed": 0, "applied": False, "details": []}

    raws = [raw for raw in board_file.read_text().splitlines() if raw.strip()]
    objs: dict[int, dict[str, Any]] = {}
    positions: dict[str, list[int]] = {}
    for i, raw in enumerate(raws):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not obj.get("id"):
            continue
        objs[i] = obj
        positions.setdefault(obj["id"], []).append(i)

    details: list[dict[str, Any]] = []
    drop: set[int] = set()
    for card_id, idxs in positions.items():
        if len(idxs) < 2:
            continue
        # newest updated_at wins; ties fall back to the first line (-i is
        # larger for the smaller index, so max() selects it).
        keep = max(idxs, key=lambda i: (str(objs[i].get("updated_at") or ""), -i))
        details.append(
            {
                "id": card_id,
                "lines": [i + 1 for i in idxs],
                "keep_line": keep + 1,
                "keep_updated_at": objs[keep].get("updated_at"),
            }
        )
        drop.update(i for i in idxs if i != keep)

    new_text = "\n".join(raw for i, raw in enumerate(raws) if i not in drop) + "\n" if raws else ""
    applied = False
    if drop and apply:
        PlannerStore(board_dir=board_file.parent, notes_dir=board_file.parent)._atomic_write(
            new_text
        )
        applied = True
    return {
        "groups": len(details),
        "removed": len(drop),
        "applied": applied,
        "details": details,
        "text_before": "\n".join(raws),
        "text_after": new_text,
    }


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
