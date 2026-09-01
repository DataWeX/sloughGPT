"""
Tests for planner.sync — shared notes -> board card sync used by
``planner sync`` and the web API sync endpoint.
"""

import pytest

from planner.core import NoteStore
from planner.store import Store
from planner.sync import cli_main, sync_notes_to_board


@pytest.fixture
def stores(tmp_path):
    return (
        NoteStore(notes_dir=tmp_path / "notes", backend="file"),
        Store(board_dir=tmp_path / "board"),
    )


def test_sync_creates_cards_for_notes(stores):
    note_store, board_store = stores
    note_store.create("Fix boot order", tags=["kernel"])
    note_store.create("Ship v2.0")
    added, updated, total = sync_notes_to_board(note_store, board_store)
    assert added == 2
    assert updated == 0
    assert total == 2
    titles = {c.title for c in board_store.load_board().cards}
    assert titles == {"Fix boot order", "Ship v2.0"}


def test_sync_is_idempotent(stores):
    note_store, board_store = stores
    note_store.create("Same title")
    sync_notes_to_board(note_store, board_store)
    added, updated, total = sync_notes_to_board(note_store, board_store)
    assert added == 0
    assert updated == 0
    assert total == 1


def test_sync_skips_notes_with_existing_card(stores):
    note_store, board_store = stores
    note_store.create("Existing")
    board_store.create_card(title="Existing", column="todo")
    added, updated, total = sync_notes_to_board(note_store, board_store)
    assert added == 0
    assert updated == 0
    assert total == 1


def test_sync_derives_column_from_status(stores):
    note_store, board_store = stores
    note_store.create("A", status="done")
    note_store.create("B", status="wip")
    note_store.create("C", status="review")
    note_store.create("D", status="open")
    sync_notes_to_board(note_store, board_store)
    columns = {c.title: c.column for c in board_store.load_board().cards}
    assert columns == {"A": "done", "B": "in_progress", "C": "review", "D": "todo"}


def test_sync_card_carries_tags_and_body(stores):
    note_store, board_store = stores
    note_store.create("Tagged", tags=["alpha", "beta"], body="some body text")
    sync_notes_to_board(note_store, board_store)
    card = board_store.load_board().cards[0]
    assert card.tags == ["alpha", "beta"]
    assert card.description.strip() == "some body text"


def test_sync_moves_card_when_status_changes(stores):
    note_store, board_store = stores
    note = note_store.create("Rotating task", status="wip")
    sync_notes_to_board(note_store, board_store)
    assert board_store.load_board().cards[0].column == "in_progress"
    note_store.update(note.id, status="done")
    added, updated, total = sync_notes_to_board(note_store, board_store)
    assert added == 0
    assert updated == 1
    assert total == 1
    assert board_store.load_board().cards[0].column == "done"


def test_sync_does_not_move_card_with_matching_column(stores):
    note_store, board_store = stores
    note_store.create("Stable", status="done")
    sync_notes_to_board(note_store, board_store)
    added, updated, total = sync_notes_to_board(note_store, board_store)
    assert added == 0
    assert updated == 0
    assert total == 1


def test_sync_cli_quiet(tmp_path, capsys):
    notes_dir = tmp_path / "notes"
    board_dir = tmp_path / "board"
    NoteStore(notes_dir=notes_dir, backend="file").create("CLI note")
    code = cli_main(["--notes-dir", str(notes_dir), "--board-dir", str(board_dir), "--quiet"])
    out = capsys.readouterr().out
    assert code == 0
    assert "1 new card(s) added, 0 moved, 1 total" in out
