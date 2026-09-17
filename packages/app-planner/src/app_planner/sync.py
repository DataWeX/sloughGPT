"""
app_planner.sync — reconcile notes and board cards.

Thin wrapper around PlannerStore.sync() for backward compatibility.
"""

from __future__ import annotations

from .store import PlannerStore


def sync_notes_to_board(
    note_store: PlannerStore, kanban_store: PlannerStore
) -> tuple[int, int, int]:
    """Sync notes to board using the unified store.

    Both arguments should be the same PlannerStore instance.
    Kept for backward compatibility with existing callers.
    """
    return kanban_store.sync()
