"""
Tests for planner sync — notes -> board card sync.
"""

import pytest
from grib.store import PlannerStore, reset_store


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
    assert columns == {"A": "done", "B": "wip", "C": "review", "D": "todo"}


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
    assert store.list_cards()[0].column == "wip"
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


# ── Twin-note idempotency (card 08baf13f) ───────────────────────────────
#
# Duplicate journal notes with the SAME title and CONFLICTING statuses made
# every sync run re-apply both twins: cards flipped column on each run and
# the count never returned to zero ("0 new, 2 moved" forever).


def test_sync_twin_notes_add_one_card_and_converge(store):
    store.create_note("Twin task", status="wip")
    store.create_note("Twin task", status="done")
    added, updated, total = store.sync()
    assert added == 1, "duplicate-title notes must produce ONE card"
    assert total == 1
    assert store.sync()[:2] == (0, 0), "second run must be net-zero"


def test_sync_twin_winner_is_latest_edit(store):
    wip = store.create_note("Rotating twin", status="wip")
    store.create_note("Rotating twin", status="done")
    # The wip twin is edited LAST — latest human intent must win.
    store.update_note(wip.id, status="wip")
    added, updated, total = store.sync()
    assert (added, updated) == (1, 0), "card is created directly in the winner's column"
    assert store.list_cards()[0].column == "wip"
    for _ in range(3):
        assert store.sync()[:2] == (0, 0), "card must never flip again"
    assert store.list_cards()[0].column == "wip"


def test_sync_twin_notes_never_duplicate_board_ids(store):
    store.create_note("Dup-id twin", status="wip")
    store.create_note("Dup-id twin", status="done")
    for _ in range(3):
        store.sync()
    ids = [c.id for c in store.list_cards()]
    assert len(ids) == len(set(ids)), "sync must not mint duplicate card ids"


def test_legacy_sync_twin_notes_converges(tmp_path):
    """The legacy NoteStore/KanbanStore branch (what the GUI --sync runs)."""
    from grib.core import Note
    from grib.kanban import KanbanStore
    from grib.sync import sync_notes_to_board

    kanban = KanbanStore(tmp_path / "board")
    kanban.init_board()
    wip = Note(id="n-1", title="Legacy twin", status="wip", updated_at="2026-10-01T00:00:00")
    done = Note(id="n-2", title="Legacy twin", status="done", updated_at="2026-10-07T00:00:00")

    class _StubNotes:
        def list_notes(self, limit: int = 50):
            return [wip, done]

    added, updated, total = sync_notes_to_board(_StubNotes(), kanban)
    assert added == 1, "legacy path must collapse twin notes to one card"
    assert total == 1
    board = kanban.load_board()
    assert board.cards[0].column == "done", "latest-edited twin decides the column"
    added, updated, total = sync_notes_to_board(_StubNotes(), kanban)
    assert (added, updated) == (0, 0), "legacy rerun must be net-zero"


# ── Orphan cards: a note deleted without its card ────────────────────────
#
# sync only ever adds and updates, so deleting a note leaves its card behind
# forever (observed live 2026-11-18: note gone, board count never dropped).
# The default must stay conservative — hand-added cards (`grib board add`)
# have no note either, so pruning without an explicit flag is a data-loss path.


def test_sync_default_keeps_card_whose_note_was_deleted(store):
    note = store.create_note("Temporary work")
    store.create_note("Keeper")
    store.sync()
    assert len(store.list_cards()) == 2

    assert store.delete_note(note.id) is True
    added, updated, total = store.sync()

    assert total == 2, "default sync must never destroy a card"
    assert {c.title for c in store.list_cards()} == {"Temporary work", "Keeper"}
    assert (added, updated) == (0, 0)


def test_sync_prune_removes_card_whose_note_was_deleted(store):
    note = store.create_note("Temporary work")
    store.create_note("Keeper")
    store.sync()

    store.delete_note(note.id)
    added, updated, total = store.sync(prune=True)

    assert total == 1, "prune must drop the card whose note is gone"
    assert {c.title for c in store.list_cards()} == {"Keeper"}
    assert (added, updated) == (0, 0), "prune must not re-add or move anything"


def test_sync_prune_leaves_cards_that_have_notes(store):
    store.create_note("Keeper A", status="wip")
    store.create_note("Keeper B", status="done")
    store.sync()

    added, updated, total = store.sync(prune=True)

    assert total == 2, "prune must not touch cards whose notes exist"
    assert (added, updated) == (0, 0)
    assert {c.title: c.column for c in store.list_cards()} == {
        "Keeper A": "wip",
        "Keeper B": "done",
    }


def test_orphan_cards_reports_drift_without_deleting(store):
    note = store.create_note("Temporary work")
    store.create_note("Keeper")
    store.sync()

    store.delete_note(note.id)
    orphans = store.orphan_cards()

    assert [c.title for c in orphans] == ["Temporary work"]
    assert len(store.list_cards()) == 2, "orphan_cards must be read-only"


def test_orphan_cards_empty_when_board_matches_notes(store):
    store.create_note("Only child")
    store.sync()

    assert store.orphan_cards() == []
