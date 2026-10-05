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

import fcntl
import hashlib
import heapq
import json
import os
import random
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

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
    # Hash chain (canonical ordering + tamper evidence; -1 = not yet chained)
    chain_index: int = -1
    chain_prev: str = ""
    chain_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Card:
        chain_index = d.get("chain_index", -1)
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
            chain_index=chain_index if isinstance(chain_index, int) else -1,
            chain_prev=d.get("chain_prev", "") or "",
            chain_hash=d.get("chain_hash", "") or "",
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
            {"name": "wip", "wip_limit": 3, "order": 1},
            {"name": "review", "wip_limit": 2, "order": 2},
            {"name": "done", "wip_limit": 0, "order": 3},
        ]
    )
    cards: list[Card] = field(default_factory=list)


# ── Hash chain ────────────────────────────────────────────────────────────

CHAIN_GENESIS = "0" * 64
_CHAIN_FIELDS = ("chain_index", "chain_prev", "chain_hash")


def chain_hash_for(card: Card, prev_hash: str) -> str:
    """Canonical chain hash: sha256(prev ‖ chain-fields-stripped payload).

    The chain fields themselves are excluded so a card's hash depends only
    on its content and its predecessor — recomputation is well-founded.
    """
    payload = {k: v for k, v in card.to_dict().items() if k not in _CHAIN_FIELDS}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256((prev_hash + "\x00" + blob).encode()).hexdigest()


# Serialized key of chain_hash by _card_line (sort_keys=True) — the marker
# for "is this board sealed?" that needs no JSON parse (add path is hot).
_SEALED_MARKER = '"chain_hash": "'


def _sealed_in(raw: str) -> bool:
    """True when board-file text *raw* holds at least one non-empty chain_hash.

    A sealed board carries chain fields on (at least) one card; a legacy
    board has the key absent or empty everywhere and must stay unsealed —
    no store mutation ever introduces chain fields where they did not exist
    (retirement rule, card a8e408dc).
    """
    start = raw.find(_SEALED_MARKER)
    while start != -1:
        value = start + len(_SEALED_MARKER)
        if raw[value : value + 1] != '"':  # non-empty value → sealed
            return True
        start = raw.find(_SEALED_MARKER, value)
    return False


# ── Store ────────────────────────────────────────────────────────────────


# ── Optimistic concurrency (OCC) ─────────────────────────────────────────
# Lost-update guard for the shared JSONL files. Every write funnel reads the
# file, rebuilds full content, and replaces it atomically — without a
# generation check, a writer holding a stale snapshot erases whatever landed
# in between (the "vanishing card" bug, 2026-10-04). Protocol:
#
#   1. read once -> content AND its sha256 token come from THAT read
#   2. build the new content from it
#   3. commit: non-blocking flock (barrier) -> revalidate token adjacent to
#      os.replace -> mismatch/busy raises BoardWriteConflict
#   4. funnel catches, backs off, re-reads, rebuilds (bounded, loud failure)
#
# Reads never lock; the flock covers only the microsecond validate+replace
# and auto-releases if the holder dies. Retry budget: exponential backoff
# with per-call jitter — linear/no-jitter herding exhausted 10 attempts in
# 4/40 hammer runs under box load spikes (2026-10-04 evidence). POSIX-only.

_OCC_ATTEMPTS = 30
_OCC_BACKOFF_S = 0.005       # first retry delay; doubles per attempt
_OCC_BACKOFF_CAP_S = 0.1     # per-sleep ceiling (worst total ≈ 3s, then LOUD)


class BoardWriteConflict(RuntimeError):
    """File generation changed between read and commit (or barrier busy)."""


def _token_of(raw: str) -> str:
    """Generation token for text just read (empty/missing file == ABSENT)."""
    if not raw:
        return "ABSENT"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _current_token(path: Path) -> str:
    """Generation token of the file right now — the validation side."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return "ABSENT"
    if not data:
        return "ABSENT"
    return hashlib.sha256(data).hexdigest()


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

    # ── OCC write path ──────────────────────────────────────────────────

    @staticmethod
    def _lockfile_for(path: Path) -> Path:
        """Stable lockfile next to the protected file — never the data file
        itself (os.replace swaps inodes; a lock on the old inode would lie)."""
        return path.parent / ".commit.lock"

    def _validate_token(self, path: Path, expect: str) -> bool:
        """OCC validation seam (tests patch this): unchanged since *expect*?"""
        return _current_token(path) == expect

    def _commit(self, path: Path, text: str, expect: str) -> None:
        """Guarded commit: non-blocking flock -> revalidate token -> atomic
        replace. Raises BoardWriteConflict for the caller's _occ retry loop."""
        fd = os.open(self._lockfile_for(path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise BoardWriteConflict(f"{path.name}: commit barrier busy") from exc
            if not self._validate_token(path, expect):
                raise BoardWriteConflict(f"{path.name}: changed since read")
            fd2, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".tmp")
            try:
                with os.fdopen(fd2, "w", encoding="utf-8") as f:
                    f.write(text)
                os.replace(tmp, path)
            except BaseException:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _atomic_write(self, text: str, expect: str | None = None) -> None:
        """Atomic board write (temp + rename, no torn files). *expect* is the
        generation token of the read *text* was built from; None validates
        against a token taken at commit (content not derived from the file,
        so there is nothing to rebuild — conflict stays fatal/loud)."""
        token = expect if expect is not None else _current_token(self._board_file)
        self._commit(self._board_file, text, token)

    def _occ(self, fn: Callable[[], Any]) -> Any:
        """OCC runner: on BoardWriteConflict, back off and re-run *fn* so it
        re-reads and rebuilds. Backoff is exponential with per-call jitter so
        concurrent retryers desynchronize instead of herding (herding burned
        the whole budget in load spikes). Exhaustion raises — loudly."""
        last: BoardWriteConflict | None = None
        rng = random.Random()  # per-call + OS-seeded: distinct across forks
        for attempt in range(1, _OCC_ATTEMPTS + 1):
            try:
                return fn()
            except BoardWriteConflict as exc:
                last = exc
                if attempt < _OCC_ATTEMPTS:
                    delay = min(_OCC_BACKOFF_CAP_S, _OCC_BACKOFF_S * 2 ** (attempt - 1))
                    time.sleep(delay * rng.uniform(0.5, 1.5))
        raise BoardWriteConflict(
            f"unresolved after {_OCC_ATTEMPTS} OCC attempts: {last}"
        ) from last

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

        OCC: read + generation token come from one snapshot; on conflict the
        read/rebuild re-runs (replacements are idempotent by id), so a
        concurrent write landing between our read and our replace can never
        be erased. A concurrent change to the same card line is last-writer-
        wins (semantic conflict, not a lost update).

        RESEAL (chain seal): when the pre-write snapshot shows a sealed
        board (any non-empty chain_hash), a changed write is followed by
        ``compute_chains()`` — a second, independent OCC write. The
        two-write window is safe by construction: the reseal re-reads
        current state (``load_board``) and commits under its own generation
        token, so it seals whatever actually landed; a crash in between
        leaves a stale chain that ``verify_chain()`` / ``board verify``
        flag — detected, never silent. Unsealed legacy boards get no reseal.

        Returns:
            Number of lines matched and replaced.
        """
        if not self._board_file.exists():
            return 0

        sealed = False
        changed = False

        def attempt() -> int:
            nonlocal sealed, changed
            old_text = self._board_file.read_text(encoding="utf-8")
            token = _token_of(old_text)
            out: list[str] = []
            matched = 0
            sealed = False
            for raw in old_text.splitlines():
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    out.append(raw)
                    continue
                if isinstance(obj, dict):
                    if obj.get("chain_hash"):
                        sealed = True
                    if obj.get("id") in replacements:
                        new_line = replacements[obj["id"]]
                        matched += 1
                        if new_line is not None:
                            out.append(new_line)
                        continue
                out.append(raw)
            new_text = "\n".join(out) + "\n" if out else ""
            changed = new_text != old_text
            if changed:
                self._atomic_write(new_text, expect=token)
            return matched

        matched = self._occ(attempt)
        if sealed and changed:
            self.compute_chains()
        return matched

    def _append_card(self, card: Card) -> None:
        """Append a single card line, adding the schema header only if missing.

        Existing lines (including old note-schema cards) are preserved
        byte-for-byte, so add operations produce a one-line diff.

        OCC: this "append" is a full rebuild from a read snapshot — without
        the generation check, two concurrent adds erase each other's line
        (the original vanishing-card bug). Rebuild re-runs on conflict.

        RESEAL: if the read snapshot shows a sealed board (string scan, no
        JSON parse — this is the benchmarked hot path), the append is
        followed by ``compute_chains()`` folding the new card into the
        chain — the same two-write, OCC-protected window documented on
        ``_surgical_rewrite``. Unsealed legacy boards are appended to
        untouched (chain fields are never introduced). On a sealed board
        the ``add_card()`` return value keeps its pre-seal chain fields;
        read them back via ``get_card()``.
        """

        sealed = False

        def attempt() -> None:
            nonlocal sealed
            raw = (
                self._board_file.read_text(encoding="utf-8")
                if self._board_file.exists()
                else ""
            )
            sealed = _sealed_in(raw)
            token = _token_of(raw)
            body = [line for line in raw.splitlines() if line.strip()]
            out = list(body)
            if not self._has_schema_header(body):
                board = self.load_board()
                out.insert(0, self._schema_line(board.name, board.columns))
            out.append(self._card_line(card))
            self._atomic_write("\n".join(out) + "\n", expect=token)

        self._occ(attempt)
        if sealed:
            self.compute_chains()

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
        expect = _current_token(self._board_file)
        self._atomic_write("\n".join(lines) + "\n" if lines else "", expect=expect)

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

    def column_names(self) -> list[str]:
        """The board's column vocabulary, read from its schema header.

        Stops at the first ``planner/1`` header line (normally line 1) and
        skips other lines with a cheap substring test, so ``add_card`` — the
        hot path measured by ``scripts/benchmark_board_write.py`` — never
        pays a full-board decode just to check a spelling. A board with no
        header falls back to the defaults ``_append_card`` would insert.
        """
        if self._board_file.exists():
            with self._board_file.open(encoding="utf-8") as fh:
                for raw in fh:
                    if '"schema"' not in raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("schema") == "planner/1" and obj.get("columns"):
                        return [c.get("name") for c in obj["columns"]]
        return [c["name"] for c in Board().columns]

    def validate_column(self, column: str) -> str:
        """Return *column* when it names a real board column, else raise.

        The single source of truth for the column vocabulary: every write
        path (CLI add/move, ``update_card``, sync) funnels through here, so a
        retired or misspelled spelling — ``in_progress``, ``in-progress``,
        ``TODO`` — can never enter the board again. Reads are untouched:
        listing a stale column still filters, it just cannot be written.
        """
        valid = self.column_names()
        if column not in valid:
            raise ValueError(f"Invalid column: {column}. Valid: {', '.join(valid)}")
        return column

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
        self.validate_column(column)
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
        if "column" in kwargs:
            self.validate_column(kwargs["column"])
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
        self.validate_column(to_column)
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

    def _read_notes_with_token(self) -> tuple[list[Note], str]:
        """Read the journal AND its generation token from ONE snapshot (OCC)."""
        raw = (
            self._notes_file.read_text(encoding="utf-8")
            if self._notes_file.exists()
            else ""
        )
        notes: list[Note] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                notes.append(Note.from_dict(json.loads(stripped)))
            except json.JSONDecodeError:
                continue
        return notes, _token_of(raw)

    def _read_notes(self) -> list[Note]:
        return self._read_notes_with_token()[0]

    def _write_notes(self, notes: list[Note], expect: str | None = None) -> None:
        """Atomic + guarded notes write (was a bare write_text — the notes
        journal had BOTH lost-update and torn-file exposure)."""
        lines = [json.dumps(n.to_dict(), ensure_ascii=False) for n in notes]
        text = "\n".join(lines) + "\n" if lines else ""
        token = expect if expect is not None else _current_token(self._notes_file)
        self._commit(self._notes_file, text, token)

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

        def attempt() -> Note:
            notes, token = self._read_notes_with_token()
            notes.append(note)
            self._write_notes(notes, expect=token)
            return note

        return self._occ(attempt)

    def update_note(self, note_id: str, **kwargs: Any) -> Note | None:

        def attempt() -> Note | None:
            notes, token = self._read_notes_with_token()
            for i, note in enumerate(notes):
                if note.id == note_id:
                    for key, value in kwargs.items():
                        if hasattr(note, key):
                            setattr(note, key, value)
                    note.updated_at = datetime.now(UTC).isoformat()
                    notes[i] = note
                    self._write_notes(notes, expect=token)
                    return note
            return None

        return self._occ(attempt)

    def delete_note(self, note_id: str) -> bool:

        def attempt() -> bool:
            notes, token = self._read_notes_with_token()
            original_len = len(notes)
            notes = [n for n in notes if n.id != note_id]
            if len(notes) < original_len:
                self._write_notes(notes, expect=token)
                return True
            return False

        return self._occ(attempt)

    def search_notes(self, query: str) -> list[Note]:
        q = query.lower()
        return [
            n
            for n in self._read_notes()
            if q in n.title.lower() or q in n.body.lower() or any(q in t.lower() for t in n.tags)
        ]

    # ── Chain ───────────────────────────────────────────────────────────

    @staticmethod
    def _chain_order(cards: list[Card]) -> tuple[list[Card], list[str]]:
        """Topological order over ``blocked_by``, stable by (created_at, id).

        Returns ``(ordered cards, cycle-involved card ids)``. Cards caught in
        a dependency cycle (or downstream of one) cannot be topologically
        placed, so they are appended in stable order — every card still gets
        an index and the call never hangs.
        """
        known = {c.id for c in cards}
        dependents: dict[str, list[int]] = {}
        indegree = [0] * len(cards)
        for idx, card in enumerate(cards):
            seen: set[str] = set()
            for dep in card.blocked_by:
                if not isinstance(dep, str) or dep not in known or dep in seen:
                    continue
                seen.add(dep)
                dependents.setdefault(dep, []).append(idx)
                indegree[idx] += 1

        def key(idx: int) -> tuple[str, str]:
            card = cards[idx]
            return (str(card.created_at or ""), str(card.id))

        heap: list[tuple[tuple[str, str], int]] = [
            (key(i), i) for i in range(len(cards)) if indegree[i] == 0
        ]
        heapq.heapify(heap)
        emitted = [False] * len(cards)
        order: list[int] = []
        while heap:
            _, idx = heapq.heappop(heap)
            emitted[idx] = True
            order.append(idx)
            for nxt in dependents.get(cards[idx].id, ()):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heapq.heappush(heap, (key(nxt), nxt))
        leftover = [i for i in range(len(cards)) if not emitted[i]]
        if leftover:
            leftover.sort(key=key)
            order.extend(leftover)
        cycles = sorted({cards[i].id for i in leftover})
        return [cards[i] for i in order], cycles

    def _reorder_cards(self, ordered: list[Card]) -> None:
        """Rewrite managed card lines in *ordered* sequence.

        Schema headers, unparsable lines, and any other non-card content are
        preserved byte-for-byte in place; card lines are emitted as one block
        at the position of the first card line. No write when unchanged.
        """
        if not self._board_file.exists():
            return

        def attempt() -> None:
            old_text = self._board_file.read_text(encoding="utf-8")
            token = _token_of(old_text)
            id_set = {c.id for c in ordered}
            card_lines = [self._card_line(c) for c in ordered]
            out: list[str] = []
            emitted = False
            for raw in old_text.splitlines():
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    out.append(raw)
                    continue
                cid = obj.get("id") if isinstance(obj, dict) else None
                if isinstance(cid, str) and cid in id_set:
                    if not emitted:
                        out.extend(card_lines)
                        emitted = True
                    continue
                out.append(raw)
            if not emitted:
                out.extend(card_lines)
            new_text = "\n".join(out) + "\n" if out else ""
            if new_text != old_text:
                self._atomic_write(new_text, expect=token)

        self._occ(attempt)

    def compute_chains(self) -> list[str]:
        """Assign the canonical hash chain to every card; reorder the file.

        Chain order is topological over ``blocked_by`` (so a blocker always
        gets a lower ``chain_index`` than the cards it blocks), stable by
        ``(created_at, id)``. Idempotent: recomputing an unchanged board
        yields identical hashes. Returns cycle-involved card ids.
        """
        board = self.load_board()
        ordered, cycles = self._chain_order(board.cards)
        prev = CHAIN_GENESIS
        for index, card in enumerate(ordered):
            card.chain_index = index
            card.chain_prev = prev
            card.chain_hash = chain_hash_for(card, prev)
            prev = card.chain_hash
        self._reorder_cards(ordered)
        return cycles

    def verify_chain(self) -> list[int]:
        """Chain indices failing verification; empty list means intact.

        Three invariants: file sequence == chain sequence, each stored link
        matches its predecessor's stored hash, and every card is chained.
        A mutated card is flagged at its own index — untouched successors
        keep verifying against their stored (stale) predecessor hash.
        """
        cards = self.load_board().cards
        broken: set[int] = set()
        for position, card in enumerate(cards):
            if card.chain_index != position:
                broken.add(card.chain_index)
        prev = CHAIN_GENESIS
        for card in cards:
            if card.chain_index < 0:
                broken.add(-1)
                continue
            if card.chain_prev != prev or card.chain_hash != chain_hash_for(card, prev):
                broken.add(card.chain_index)
            prev = card.chain_hash
        return sorted(broken)

    # ── Sync ────────────────────────────────────────────────────────────

    def sync(self) -> tuple[int, int, int]:
        """Reconcile notes ↔ board. Returns (added, updated, total)."""
        notes = self.list_notes(limit=9999)
        board = self.load_board()
        existing = {c.title: c for c in board.cards}
        added = 0
        updated = 0

        for note in notes:
            col = config.STATUS_TO_COLUMN.get((note.status or "").lower(), "todo")
            title = note.title or "(untitled)"
            card = existing.get(title)

            if card is None:
                self.add_card(
                    title=title,
                    column=col,
                    tags=list(note.tags or []),
                    description=note.body or "",
                    assignee=note.assignee or "",
                    sprint=note.sprint or "",
                    gh=note.gh or "",
                )
                added += 1
                continue

            if card.column != col:
                self.move_card(card.id, col)
                updated += 1

            if note.assignee and card.assignee != note.assignee:
                self.update_card(card.id, assignee=note.assignee)
                updated += 1

        # Reconcile, then seal the board: chain recomputation is idempotent
        # and rewrites cards in canonical order (blockers before blocked).
        self.compute_chains()

        total = len(self.load_board().cards)
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
