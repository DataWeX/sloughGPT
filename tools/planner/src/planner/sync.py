"""
planner sync — create board cards for notes that do not have one yet.

Shared implementation used by the ``planner sync`` command, the GUI Sync
button (``planner gui``), and the ``sync-notes-to-board`` console script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from planner import config
from planner.kanban import KanbanStore


def sync_notes_to_board(note_store, kanban_store: KanbanStore) -> tuple[int, int]:
    """Create a board card for every note without a matching card.

    A note matches a card when their titles are equal (case-sensitive), the
    same convention used by the original ``sync-notes-to-board`` script.
    Card column is derived from the note status via ``config.STATUS_TO_COLUMN``.

    Args:
        note_store: planner NoteStore instance.
        kanban_store: planner KanbanStore instance.

    Returns:
        Tuple of ``(added, total)`` where *added* is the number of new cards
        and *total* the resulting board card count.

    Side effects:
        - Writes new cards to the kanban board file.
    """
    notes = note_store.list_notes(limit=9999)
    board = kanban_store.load_board()
    existing = {card.title for card in board.cards}
    added = 0
    for note in notes:
        col = config.STATUS_TO_COLUMN.get((note.status or "").lower(), "todo")
        title = note.title or "(untitled)"
        if title in existing:
            continue
        kanban_store.add_card(
            title=title,
            column=col,
            tags=list(note.tags or []),
            description=note.body or "",
        )
        existing.add(title)
        added += 1
    total = len(kanban_store.load_board().cards)
    return added, total


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``planner sync`` / ``sync-notes-to-board``."""
    parser = argparse.ArgumentParser(
        prog="planner sync",
        description="Create board cards for notes that do not have one yet.",
    )
    parser.add_argument("--notes-dir", default=None, help="Notes directory")
    parser.add_argument("--board-dir", default=None, help="Board directory")
    parser.add_argument("--backend", default=None, choices=config.BACKENDS)
    parser.add_argument("--quiet", action="store_true", help="Only print the summary line")
    args = parser.parse_args(argv)

    from planner.core import NoteStore

    notes_dir = Path(args.notes_dir) if args.notes_dir else config.default_notes_dir()
    note_store = NoteStore(
        notes_dir=notes_dir,
        backend=args.backend or config.default_backend(notes_dir=notes_dir),
    )
    kanban_store = KanbanStore(
        board_dir=Path(args.board_dir) if args.board_dir else config.default_board_dir(),
    )

    added, total = sync_notes_to_board(note_store, kanban_store)

    if not args.quiet:
        board = kanban_store.load_board()
        for card in board.cards:
            icon = "\u2713" if card.column == "done" else "\u25cb"
            print(f"  {icon} [{card.column:12s}] {card.title}")
        print(f"\n{added} new card(s) added, {total} total")
    else:
        print(f"{added} new card(s) added, {total} total")
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
