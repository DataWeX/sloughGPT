"""
app_planner.sync — reconcile notes and board cards.

Supports both the new PlannerStore and the legacy NoteStore/KanbanStore pair.
"""

from __future__ import annotations

from typing import Any


def sync_notes_to_board(note_store: Any, kanban_store: Any) -> tuple[int, int, int]:
    """Sync notes to board.

    Accepts either:
    - A single PlannerStore (both args same instance) — calls store.sync()
    - Legacy (NoteStore, KanbanStore) pair — performs sync manually
    """
    from .store import PlannerStore

    if isinstance(kanban_store, PlannerStore):
        return kanban_store.sync()

    # Legacy path: NoteStore + KanbanStore
    from . import config

    notes = note_store.list_notes(limit=9999)
    board = kanban_store.load_board()
    existing = {c.title: c for c in board.cards}
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
                sprint=note.sprint or "",
            )
            added += 1
            continue

        if card.column != col:
            kanban_store.move_card(card.id, col)
            updated += 1

        if note.assignee and card.assignee != note.assignee:
            kanban_store.update_card(card.id, assignee=note.assignee)
            updated += 1

    total = len(kanban_store.load_board().cards)
    return added, updated, total
