"""
app_planner.sync — reconcile notes and board cards.

Supports both the new PlannerStore and the legacy NoteStore/KanbanStore pair.
"""

from __future__ import annotations

from typing import Any


def canonical_notes(notes: list[Any]) -> list[Any]:
    """Collapse duplicate-title notes to one deterministic winner each.

    Journal twins (same title, conflicting statuses) must produce ONE board
    transition per sync run — otherwise every run re-applies both twins and
    reports phantom moves forever while cards flip column (card 08baf13f).

    Winner per title = most recently updated note; ties break on created_at,
    then id (all descending), so the latest human intent decides the column
    and repeated syncs are net-zero. Returned in resolved-title order for
    deterministic processing.
    """

    def rank(note: Any) -> tuple[str, str, str]:
        return (
            getattr(note, "updated_at", "") or "",
            getattr(note, "created_at", "") or "",
            getattr(note, "id", "") or "",
        )

    winners: dict[str, Any] = {}
    for note in notes:
        key = note.title or "(untitled)"
        current = winners.get(key)
        if current is None or rank(note) > rank(current):
            winners[key] = note
    return sorted(winners.values(), key=lambda n: n.title or "(untitled)")


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

    notes = canonical_notes(note_store.list_notes(limit=9999))
    board = kanban_store.load_board()
    existing = {c.title: c for c in board.cards}
    added = 0
    updated = 0

    for note in notes:
        col = config.STATUS_TO_COLUMN.get((note.status or "").lower(), "todo")
        title = note.title or "(untitled)"
        card = existing.get(title)

        if card is None:
            card = kanban_store.add_card(
                title=title,
                column=col,
                tags=list(note.tags or []),
                description=note.body or "",
                assignee=note.assignee or "",
                sprint=note.sprint or "",
            )
            existing[title] = card
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
