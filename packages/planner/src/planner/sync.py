"""
planner sync — reconcile notes and board cards.

Creates a card for every note without one and moves existing cards to the
column matching the note's current status. Shared implementation used by the
``planner sync`` command, the GUI Sync button (``planner gui``), and the
``sync-notes-to-board`` console script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from planner import config
from planner.kanban import KanbanStore


def sync_notes_to_board(note_store, kanban_store: KanbanStore) -> tuple[int, int, int]:
    """Create board cards for notes without one and move cards to match status.

    A note matches a card when their titles are equal (case-sensitive), the
    same convention used by the original ``sync-notes-to-board`` script.
    Card column is derived from the note status via ``config.STATUS_TO_COLUMN``;
    an existing card is moved to that column when it differs, so the board
    stays in step with note status changes.

    Args:
        note_store: planner NoteStore instance.
        kanban_store: planner KanbanStore instance.

    Returns:
        Tuple of ``(added, updated, total)`` where *added* is the number of new
        cards, *updated* the number of cards moved to a different column, and
        *total* the resulting board card count.

    Side effects:
        - Writes new cards to the kanban board file.
        - Moves existing cards whose column no longer matches the note status.
    """
    notes = note_store.list_notes(limit=9999)
    board = kanban_store.load_board()
    existing = {card.title: card for card in board.cards}
    added = 0
    updated = 0
    for note in notes:
        col = config.STATUS_TO_COLUMN.get((note.status or "").lower(), "todo")
        title = note.title or "(untitled)"
        card = existing.get(title)
        if card is None:
            kanban_store.add_card(
                title=title,
                column=col,
                tags=list(note.tags or []),
                description=note.body or "",
                assignee=note.assignee or "",
            )
            existing[title] = None
            added += 1
            continue
        if card.column != col:
            kanban_store.move_card(card.id, col)
            card.column = col
            updated += 1
        if note.assignee and card.assignee != note.assignee:
            kanban_store.update_card(card.id, assignee=note.assignee)
            updated += 1
    total = len(kanban_store.load_board().cards)
    return added, updated, total


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``planner sync`` / ``sync-notes-to-board``."""
    parser = argparse.ArgumentParser(
        prog="planner sync",
        description="Create missing board cards and move cards to match note status.",
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

    added, updated, total = sync_notes_to_board(note_store, kanban_store)

    if not args.quiet:
        board = kanban_store.load_board()
        for card in board.cards:
            icon = "\u2713" if card.column == "done" else "\u25cb"
            print(f"  {icon} [{card.column:12s}] {card.title}")
        print(f"\n{added} new card(s) added, {updated} moved, {total} total")
    else:
        print(f"{added} new card(s) added, {updated} moved, {total} total")
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
