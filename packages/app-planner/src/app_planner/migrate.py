"""
app_planner.migrate — import repo-local planner data into the canonical store.

The canonical-state flip moved the default board/notes dirs to the
user-config locations (``config.BOARD_FALLBACK`` / ``config.NOTES_FALLBACK``).
This module carries existing repo-local data (``<root>/.kanban``,
``<root>/.dev-notes``) across, once, safely:

- **Source chain verified first** — a tampered hash chain refuses to
  migrate (loud failure, never silent corruption of the canonical store).
- **Union by id, canonical wins** — existing canonical cards/notes are
  never overwritten or removed; source rows missing from the canonical
  store are appended.
- **Backup before write** — any canonical file about to change is copied
  to ``<dir>/backups/<utc-stamp>/`` first (preserve-the-original rule).
- **Originals untouched** — the source repo-local files are read-only
  input; nothing under the source root is written or deleted.

``dry_run=True`` performs every read + verification step and reports what
*would* be imported, writing nothing.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import config
from .store import Note, PlannerStore

BOARD_FILE = "board.jsonl"
NOTES_FILE = "notes.journal.jsonl"


class MigrationError(RuntimeError):
    """Raised when migration cannot proceed safely (e.g. broken source chain)."""


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _backup(path: Path, stamp: str) -> Path | None:
    """Copy *path* under ``<parent>/backups/<stamp>/`` before it changes."""
    if not path.exists():
        return None
    dest = path.parent / "backups" / stamp / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return dest


def _read_note_lines(path: Path) -> list[Note]:
    if not path.exists():
        return []
    notes: list[Note] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            notes.append(Note.from_dict(json.loads(raw)))
        except json.JSONDecodeError:
            continue
    return notes


def migrate(
    source_root: Path | None = None,
    *,
    dry_run: bool = False,
    board_dir: Path | None = None,
    notes_dir: Path | None = None,
) -> dict[str, Any]:
    """Import repo-local planner data into the canonical store.

    Args:
        source_root: Repo root holding ``.kanban``/``.dev-notes``. Defaults
            to repository detection (``config.find_project_root()``).
        dry_run: Verify + count only; write nothing.
        board_dir: Canonical board dir (default ``config.BOARD_FALLBACK``).
        notes_dir: Canonical notes dir (default ``config.NOTES_FALLBACK``).

    Returns:
        Summary dict: source, per-store imported/skipped counts, backup
        paths, dry_run flag.

    Raises:
        MigrationError: no source found, or the source board chain fails
        verification.
    """
    root = source_root if source_root is not None else config.find_project_root()
    if root is None:
        raise MigrationError(
            "no repo-local planner data found (no .kanban/board.json in any "
            "ancestor); pass --source-root explicitly"
        )
    root = Path(root)
    src_board_dir = root / ".kanban"
    src_notes_dir = root / ".dev-notes"
    dst_board_dir = Path(board_dir) if board_dir else config.BOARD_FALLBACK
    dst_notes_dir = Path(notes_dir) if notes_dir else config.NOTES_FALLBACK

    result: dict[str, Any] = {
        "source": str(root),
        "dry_run": dry_run,
        "board": {"imported": 0, "skipped": 0, "cycles": []},
        "notes": {"imported": 0, "skipped": 0},
        "backups": [],
    }

    # ── Board ─────────────────────────────────────────────────────────
    src_board_file = src_board_dir / BOARD_FILE
    if src_board_file.is_file():
        src_store = PlannerStore(board_dir=src_board_dir, notes_dir=src_notes_dir)
        src_cards = src_store.load_board().cards
        # Verify only a chained source: boards built by plain add_card are
        # legitimately unchained (chain_index=-1) until compute_chains runs;
        # they import with a fresh chain. A source that HAS chain links but
        # fails verification is tampered/inconsistent → refuse.
        if any(c.chain_index >= 0 for c in src_cards):
            broken = src_store.verify_chain()
            if broken:
                raise MigrationError(
                    f"source board chain broken at index/indices {broken[:10]} — "
                    "repair the source board before migrating (refusing to import "
                    "an unverifiable chain into the canonical store)"
                )
        dst_store = PlannerStore(board_dir=dst_board_dir, notes_dir=dst_notes_dir)
        existing_ids = {c.id for c in dst_store.load_board().cards}
        incoming = [c for c in src_cards if c.id not in existing_ids]
        result["board"]["imported"] = len(incoming)
        result["board"]["skipped"] = len(src_cards) - len(incoming)
        if not dry_run and incoming:
            backup = _backup(dst_board_dir / BOARD_FILE, _stamp())
            if backup:
                result["backups"].append(str(backup))
            board = dst_store.load_board()
            # Imported cards arrive with their source-side chain fields;
            # persist the union first, then let compute_chains() reload it,
            # re-derive ONE canonical chain over the union, and persist the
            # reordered board. blocked_by cycles across sources are reported
            # (every card still gets an index — no card is ever dropped).
            board.cards.extend(incoming)
            dst_store._write_board(board)
            result["board"]["cycles"] = dst_store.compute_chains()

    # ── Notes ─────────────────────────────────────────────────────────
    src_notes_file = src_notes_dir / NOTES_FILE
    if src_notes_file.is_file():
        src_notes = _read_note_lines(src_notes_file)
        dst_store = PlannerStore(board_dir=dst_board_dir, notes_dir=dst_notes_dir)
        existing_notes = dst_store._read_notes()
        existing_note_ids = {n.id for n in existing_notes}
        incoming_notes = [n for n in src_notes if n.id not in existing_note_ids]
        result["notes"]["imported"] = len(incoming_notes)
        result["notes"]["skipped"] = len(src_notes) - len(incoming_notes)
        if not dry_run and incoming_notes:
            backup = _backup(dst_notes_dir / NOTES_FILE, _stamp())
            if backup:
                result["backups"].append(str(backup))
            merged = existing_notes + incoming_notes
            dst_store._write_notes(merged)

    return result
