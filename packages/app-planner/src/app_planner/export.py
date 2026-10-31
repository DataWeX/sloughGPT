"""
app_planner.export — mirror the canonical store into the repo for GitHub sync.

The canonical-state flip keeps mutable planner state in ``~/.config`` (see
``config`` docstring); GitHub only ever sees repo copies. ``export()`` is the
producer half of the sync pipeline (``scripts/kanban_sync.sh`` under the
``kanban-sync.timer`` systemd unit drives it). It mirrors:

* canonical ``board.jsonl``  -> ``<repo>/.kanban/board.jsonl`` — the *live*
  copy the chain verifier and fresh clones read (NOT the dated
  ``.kanban/snapshots/`` directory, which ``snapshot()`` keeps appending to
  and which would explode at timer frequency). This mirror is a **union
  merge**, not a byte-copy: the repo copy can hold cards the canonical store
  never saw (sessions commit cards straight to it), so canonical decides the
  content of cards it owns while repo-only cards survive. The union is then
  re-sealed with :meth:`PlannerStore.compute_chains` so the board keeps one
  valid hash chain. A byte-copy deleted 1124 of 1136 cards on 2026-10-31;
* canonical ``*.md`` notes   -> ``<repo>/.dev-notes/*.md`` — byte-exact
  add/update; repo notes no longer present in canonical are removed (they
  are git-tracked, so history keeps them; canonical is source of truth);
* ``<repo>/.kanban/manifest.json`` — ``{exported_at, note_count,
  card_count, board_sha256, source}`` for the CI freshness alarm
  (``scripts/check_kanban_freshness.py``).

Legacy repo-only artifacts (``.dev-notes/store/``, ``notes.journal.jsonl``)
are left untouched — grandfathered cleanup is a separate card.

No git operations happen here: the sync script owns commit + push.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import config
from .store import PlannerStore

MANIFEST = "manifest.json"


class ExportError(RuntimeError):
    """Raised when the canonical store cannot be exported."""


def union_board(
    dest_board_dir: Path,
    dest_notes_dir: Path,
    *,
    board_dir: Path,
    notes_dir: Path,
) -> int:
    """Union-merge canonical into the repo board and re-seal the chain.

    Never clobbers: cards only the repo knows about are kept, cards both
    copies know about are taken from canonical (source of truth), and the
    result is written back with exactly one valid hash chain — the two copies
    may arrive independently sealed (canonical re-chained its cards 0..11
    while the repo chain ran to 1115), so a plain concatenation would leave
    the gate failing on duplicate ``chain_index`` values.

    Returns:
        Number of cards written to the repo board.
    """
    src = PlannerStore(board_dir=board_dir, notes_dir=notes_dir).load_board()
    dest_store = PlannerStore(board_dir=dest_board_dir, notes_dir=dest_notes_dir)
    dest = dest_store.load_board()

    by_id = {card.id: card for card in dest.cards}
    for card in src.cards:
        by_id[card.id] = card  # canonical wins for cards it owns
    # Repo order first (stable, and it drops any duplicate id the file
    # already had), then canonical-only cards in canonical order.
    repo_order = list(dict.fromkeys(card.id for card in dest.cards))
    merged = [by_id[card_id] for card_id in repo_order]
    merged.extend(card for card in src.cards if card.id not in set(repo_order))

    dest.name = src.name
    dest.columns = src.columns
    dest.cards = merged
    dest_store.save_board(dest)
    dest_store.compute_chains()  # one chain over the union (also reorders)
    return len(merged)


def export(
    repo_root: Path | None = None,
    *,
    board_dir: Path | None = None,
    notes_dir: Path | None = None,
) -> dict[str, Any]:
    """Mirror canonical board + notes into the repo and write the manifest.

    Args:
        repo_root: Destination repo root (default: cwd).
        board_dir: Canonical board dir (default ``config.BOARD_FALLBACK``).
        notes_dir: Canonical notes dir (default ``config.NOTES_FALLBACK``).

    Returns:
        Dict with destination paths and counts (card_count, note_count,
        notes_copied, notes_removed).

    Raises:
        ExportError: canonical board missing, canonical notes dir missing,
            or destination repo root is not a directory. A missing notes
            dir refuses instead of masquerading as "zero notes" — that
            would otherwise delete every mirrored note.
    """
    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    if not repo.is_dir():
        raise ExportError(f"repo root does not exist: {repo}")

    bdir = Path(board_dir) if board_dir else config.BOARD_FALLBACK
    ndir = Path(notes_dir) if notes_dir else config.NOTES_FALLBACK
    src_board = bdir / "board.jsonl"
    if not src_board.is_file():
        raise ExportError(f"no canonical board at {src_board} — refusing to export")
    if not ndir.is_dir():
        raise ExportError(f"no canonical notes dir at {ndir} — refusing to export")

    # Board mirror (live copy; .kanban/snapshots/ is a different consumer).
    kanban_dir = repo / ".kanban"
    kanban_dir.mkdir(parents=True, exist_ok=True)
    notes_dest = repo / ".dev-notes"
    notes_dest.mkdir(parents=True, exist_ok=True)
    dest_board = kanban_dir / "board.jsonl"
    board_bytes = src_board.read_bytes()
    if dest_board.is_file():
        # Existing repo board: union-merge so repo-only cards survive
        # (see union_board). board_bytes then describes the union.
        union_board(
            kanban_dir,
            notes_dest,
            board_dir=bdir,
            notes_dir=ndir,
        )
        board_bytes = dest_board.read_bytes()
    else:
        # First export into this repo — nothing exists to destroy.
        dest_board.write_bytes(board_bytes)

    # Notes mirror: byte-exact add/update, prune files canonical dropped.
    canonical_notes = {p.name: p for p in ndir.glob("*.md")}
    notes_copied = 0
    for name, src in sorted(canonical_notes.items()):
        shutil.copy2(src, notes_dest / name)
        notes_copied += 1
    notes_removed = 0
    for stale in sorted(notes_dest.glob("*.md")):
        if stale.name not in canonical_notes:
            stale.unlink()
            notes_removed += 1

    card_count = sum(1 for line in board_bytes.splitlines() if line.strip())
    manifest = {
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note_count": len(canonical_notes),
        "card_count": card_count,
        "board_sha256": hashlib.sha256(board_bytes).hexdigest(),
        "source": str(bdir),
    }
    dest_manifest = kanban_dir / MANIFEST
    dest_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "board": str(dest_board),
        "manifest": str(dest_manifest),
        "note_count": len(canonical_notes),
        "card_count": card_count,
        "notes_copied": notes_copied,
        "notes_removed": notes_removed,
    }
