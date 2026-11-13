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


# ── card -> note propagation (card 20261112_063528) ─────────────────────────
#
# sync is a note -> card projection: it restores card.column from the bound
# note's status on EVERY run. update_card/move_card used to write only the
# card, so any direct column move was silently reverted on the next sync
# (observed twice on card 20261102_001441: wip -> todo, done -> todo rev4).


def test_update_card_column_survives_sync(store):
    """The reported defect: a direct column move must not be reverted."""
    store.create_note("Move me", status="open")  # open -> todo
    store.sync()
    card = store.list_cards()[0]
    assert card.column == "todo"

    store.update_card(card.id, column="done")
    store.sync()  # used to restore the note's status over the card

    assert store.list_cards()[0].column == "done", "sync reverted the card move"


def test_move_card_column_survives_sync(store):
    store.create_note("Mover", status="open")
    store.sync()
    card = store.list_cards()[0]

    store.move_card(card.id, "wip")
    store.sync()

    assert store.list_cards()[0].column == "wip", "sync reverted move_card"


def test_update_card_propagates_status_to_bound_note(store):
    """The contract itself: moving the card writes the bound note's status."""
    store.create_note("Propagate", status="open")
    store.sync()
    card = store.list_cards()[0]

    store.update_card(card.id, column="done")

    notes = store.list_notes(limit=10)
    assert len(notes) == 1
    assert notes[0].status == "done", "card move must reach the bound note"


def test_move_card_propagates_status_to_bound_note(store):
    store.create_note("Move propagates", status="open")
    store.sync()
    card = store.list_cards()[0]

    store.move_card(card.id, "review")

    assert store.list_notes(limit=10)[0].status == "review"


def test_update_card_propagates_via_id_bound_note(store):
    """Production shape: the grib note's id EQUALS the card id (the
    20261102_001441 workaround relied on update_note(card_id, ...))."""
    store.create_note("Id bound", status="open")
    store.sync()
    card = store.list_cards()[0]
    # Force the id-binding shape the real journal uses.
    note = store.list_notes(limit=10)[0]
    if note.id != card.id:
        store.update_note(note.id, title=card.title)  # keep title binding valid
        store._write_notes(
            [
                n if n.id != note.id else type(n)(**{**n.__dict__, "id": card.id})
                for n in store.list_notes(limit=10)
            ]
        )
    assert any(n.id == card.id for n in store.list_notes(limit=10))

    store.update_card(card.id, column="done")
    store.sync()

    assert store.list_cards()[0].column == "done"


def test_update_card_without_bound_note_creates_none(store):
    """MUST-NOT: a card with no note must not materialize one on move."""
    store.add_card("Orphan", column="todo")
    card = store.list_cards()[0]

    store.update_card(card.id, column="done")

    assert store.list_notes(limit=10) == []
    store.sync()
    assert store.list_cards()[0].column == "done", "no note -> nothing to revert"


def test_update_card_non_column_kwarg_leaves_note_status(store):
    """Only `column` maps to status; other kwargs must not rewrite it."""
    store.create_note("Metadata only", status="wip")
    store.sync()
    card = store.list_cards()[0]

    store.update_card(card.id, description="changed", assignee="mana")

    assert store.list_notes(limit=10)[0].status == "wip"


def test_update_card_column_roundtrip_maps_open_back(store):
    """todo <-> open are distinct vocabularies; moving back must not
    overwrite the note with a status sync cannot round-trip."""
    store.create_note("Round trip", status="open")
    store.sync()
    card = store.list_cards()[0]

    store.update_card(card.id, column="done")
    store.update_card(card.id, column="todo")
    store.sync()

    assert store.list_cards()[0].column == "todo"
    assert store.list_notes(limit=10)[0].status in ("open", "todo")
