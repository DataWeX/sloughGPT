"""
app_planner.sync — reconcile notes and board cards.

Supports both the new PlannerStore and the legacy NoteStore/KanbanStore pair.
Both paths share the hardening policy of card 3ccc38b7: unmappable note
statuses never create or move cards; ``repair=True`` is the explicit
pessimistic (disaster-recovery) reset to ``todo``; ``dry_run=True`` reports
without writing.
"""

from __future__ import annotations

from typing import Any


def sync_notes_to_board(
    note_store: Any,
    kanban_store: Any,
    *,
    repair: bool = False,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Sync notes to board.

    Accepts either:
    - A single PlannerStore (both args same instance) — calls store.sync()
    - Legacy (NoteStore, KanbanStore) pair — performs sync manually

    Returns (added, updated, total). The PlannerStore path records a full
    ``SyncReport`` on the store; the legacy path reports through the return
    value only.
    """
    from .store import PlannerStore, coerce_tags

    if isinstance(kanban_store, PlannerStore):
        return kanban_store.sync(repair=repair, dry_run=dry_run)

    # Legacy path: NoteStore + KanbanStore
    from . import config

    notes = note_store.list_notes(limit=9999)
    board = kanban_store.load_board()
    existing = {c.title: c for c in board.cards}
    added = 0
    updated = 0

    for note in notes:
        col = config.resolve_column(note.status)
        title = note.title or "(untitled)"
        card = existing.get(title)

        if col is None:
            # No column information: never create, never silently move.
            if card is None:
                continue
            if repair and card.column != "todo":
                if not dry_run:
                    kanban_store.move_card(card.id, "todo")
                updated += 1
            continue

        if card is None:
            if not dry_run:
                kanban_store.add_card(
                    title=title,
                    column=col,
                    tags=coerce_tags(note.tags),
                    description=note.body or "",
                    assignee=note.assignee or "",
                    sprint=note.sprint or "",
                )
            added += 1
            continue

        if card.column != col:
            if not dry_run:
                kanban_store.move_card(card.id, col)
            updated += 1

        if note.assignee and card.assignee != note.assignee:
            if not dry_run:
                kanban_store.update_card(card.id, assignee=note.assignee)
            updated += 1

    total = len(board.cards) + added if dry_run else len(kanban_store.load_board().cards)
    return added, updated, total
