"""
Tests for planner sync — notes -> board card sync.
"""

import pytest
from app_planner.store import PlannerStore, reset_store


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


@pytest.fixture
def store(tmp_path):
    return PlannerStore(board_dir=tmp_path / "board", notes_dir=tmp_path / "notes")


def test_sync_creates_cards_for_notes(store):
    store.create_note("Fix boot order", tags=["kernel"])
    store.create_note("Ship v2.0")
    added, updated, total = store.sync()
    assert added == 2
    assert updated == 0
    assert total == 2
    titles = {c.title for c in store.list_cards()}
    assert titles == {"Fix boot order", "Ship v2.0"}


def test_sync_is_idempotent(store):
    store.create_note("Same title")
    store.sync()
    added, updated, total = store.sync()
    assert added == 0
    assert updated == 0
    assert total == 1


def test_sync_skips_notes_with_existing_card(store):
    store.create_note("Existing")
    store.add_card("Existing", column="todo")
    added, updated, total = store.sync()
    assert added == 0
    assert updated == 0
    assert total == 1


def test_sync_derives_column_from_status(store):
    store.create_note("A", status="done")
    store.create_note("B", status="wip")
    store.create_note("C", status="review")
    store.create_note("D", status="open")
    store.sync()
    columns = {c.title: c.column for c in store.list_cards()}
    assert columns == {"A": "done", "B": "in_progress", "C": "review", "D": "todo"}


def test_sync_card_carries_tags_and_body(store):
    store.create_note("Tagged", tags=["alpha", "beta"], body="some body text")
    store.sync()
    card = store.list_cards()[0]
    assert card.tags == ["alpha", "beta"]
    assert card.description.strip() == "some body text"


def test_sync_propagates_assignee(store):
    note = store.create_note("Assigned", assignee="mana")
    store.sync()
    card = store.list_cards()[0]
    assert card.assignee == "mana"
    store.update_note(note.id, assignee="alice")
    added, updated, total = store.sync()
    assert updated == 1
    assert store.list_cards()[0].assignee == "alice"


def test_sync_moves_card_when_status_changes(store):
    note = store.create_note("Rotating task", status="wip")
    store.sync()
    assert store.list_cards()[0].column == "in_progress"
    store.update_note(note.id, status="done")
    added, updated, total = store.sync()
    assert added == 0
    assert updated == 1
    assert total == 1
    assert store.list_cards()[0].column == "done"


def test_sync_does_not_move_card_with_matching_column(store):
    store.create_note("Stable", status="done")
    store.sync()
    added, updated, total = store.sync()
    assert added == 0
    assert updated == 0
    assert total == 1


def test_sync_duplicate_title_twins_get_their_own_card(store):
    """Two notes sharing a title must not collapse onto one card.

    Regression: `existing` was snapshotted before the loop and never updated
    as cards were appended, so the second twin could not see the card its
    sibling had just created.
    """
    store.create_note("Auth facade exports", status="wip")
    store.create_note("Auth facade exports", status="done")
    added, updated, total = store.sync()
    assert added == 2
    assert total == 2
    assert [c.column for c in store.list_cards()] == ["in_progress", "done"]


def test_sync_is_idempotent_with_duplicate_titles(store):
    """Card 08baf13f: twin notes made every sync run report '2 moved' forever.

    Each note must keep claiming the same card run over run, so a converged
    board stays converged.
    """
    store.create_note("Auth facade exports", status="wip")
    store.create_note("Auth facade exports", status="done")
    store.sync()
    first = [c.column for c in store.list_cards()]

    for run in (2, 3, 4):
        added, updated, total = store.sync()
        assert added == 0, f"run {run} manufactured a duplicate card"
        assert updated == 0, f"run {run} churned the board (not idempotent)"
        assert total == 2
        assert [c.column for c in store.list_cards()] == first, f"run {run} moved a card"


def test_sync_duplicate_title_twins_follow_their_own_status(store):
    """A twin's status change must move only its own card."""
    a = store.create_note("Dup", status="wip")
    store.create_note("Dup", status="done")
    store.sync()
    store.update_note(a.id, status="review")
    added, updated, total = store.sync()
    assert added == 0
    assert total == 2
    assert sorted(c.column for c in store.list_cards()) == ["done", "review"]


def test_sync_three_way_title_collision_stays_stable(store):
    for status in ("open", "wip", "done"):
        store.create_note("Triple", status=status)
    store.sync()
    assert [c.column for c in store.list_cards()] == ["todo", "in_progress", "done"]
    added, updated, total = store.sync()
    assert (added, updated, total) == (0, 0, 3)


# ── Legacy (NoteStore + KanbanStore) path, used by the GUI ────────────────


@pytest.fixture
def legacy(tmp_path):
    from app_planner.core import NoteStore
    from app_planner.kanban import KanbanStore

    notes = NoteStore(notes_dir=tmp_path / "notes", backend="file")
    board = KanbanStore(board_dir=tmp_path / "board")
    board.init_board()
    return notes, board


def test_legacy_sync_is_idempotent(legacy):
    from app_planner.sync import sync_notes_to_board

    notes, board = legacy
    notes.create("Ship it", status="wip")
    sync_notes_to_board(notes, board)
    for run in (2, 3):
        added, updated, total = sync_notes_to_board(notes, board)
        assert (added, updated) == (0, 0), f"run {run} churned the legacy board"
    assert len(board.load_board().cards) == 1


def test_legacy_sync_duplicate_title_twins_stay_stable(legacy):
    from app_planner.sync import sync_notes_to_board

    notes, board = legacy
    notes.create("Auth facade exports", status="wip")
    notes.create("Auth facade exports", status="done")
    added, updated, total = sync_notes_to_board(notes, board)
    first = [c.column for c in board.load_board().cards]

    for run in (2, 3):
        a, u, t = sync_notes_to_board(notes, board)
        assert u == 0, f"run {run} churned the legacy board (not idempotent)"
        assert t == total
        assert [c.column for c in board.load_board().cards] == first
