"""
planner sync — reconcile notes and board cards.

Creates a card for every note without one and moves existing cards to the
column matching the note's current status. Shared implementation used by the
``planner sync`` command and the web API sync endpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from planner import config
from planner.store import Store
from planner.hashtree import HashTreeStore, create_hash_tree


def sync_notes_to_board(note_store, board_store: Store) -> tuple[int, int, int]:
    """Create board cards for notes without one and move cards to match status.

    A note matches a card when their titles are equal (case-sensitive).
    Card column is derived from the note status via ``config.STATUS_TO_COLUMN``;
    an existing card is moved to that column when it differs.

    Also creates/updates hash trees for cards.

    Args:
        note_store: planner NoteStore instance.
        board_store: planner Store instance.

    Returns:
        Tuple of ``(added, updated, total)`` where *added* is the number of new
        cards, *updated* the number of cards moved to a different column, and
        *total* the resulting board card count.
    """
    ht_store = HashTreeStore()
    notes = note_store.list_notes(limit=9999)
    board = board_store.load_board()
    existing = {card.title: card for card in board.cards}
    added = 0
    updated = 0
    for note in notes:
        col = config.STATUS_TO_COLUMN.get((note.status or "").lower(), "todo")
        title = note.title or "(untitled)"
        card = existing.get(title)
        if card is None:
            new_card = board_store.create_card(
                title=title,
                column=col,
                tags=list(note.tags or []),
                description=note.body or "",
            )
            # Create hash tree for new card
            tree = create_hash_tree(
                card_id=new_card.id,
                card_content=title,
                tray=col,
                position=0,
            )
            if note.body:
                tree.add_note(note.id, note.body)
            ht_store.save(tree)
            # Update card with root_hash
            board_store.update_card(new_card.id, root_hash=tree.root.root)
            existing[title] = None
            added += 1
            continue
        if card.column != col:
            board_store.move_card(card.id, col)
            card.column = col
            updated += 1
        # Update hash tree with current note content
        tree = ht_store.get(card.id)
        if tree and note.body:
            tree.add_note(note.id, note.body)
            ht_store.save(tree)
    total = len(board_store.load_board().cards)
    return added, updated, total


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``planner sync``."""
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
    board_store = Store(
        board_dir=Path(args.board_dir) if args.board_dir else config.default_board_dir(),
    )

    added, updated, total = sync_notes_to_board(note_store, board_store)

    if not args.quiet:
        board = board_store.load_board()
        for card in board.cards:
            icon = "\u2713" if card.column == "done" else "\u25cb"
            print(f"  {icon} [{card.column:12s}] {card.title}")
        print(f"\n{added} new card(s) added, {updated} moved, {total} total")
    else:
        print(f"{added} new card(s) added, {updated} moved, {total} total")
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
