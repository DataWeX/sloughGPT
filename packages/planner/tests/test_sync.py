"""Tests for planner.sync — reconcile notes and board cards."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from planner.sync import sync_notes_to_board
from planner.store import Store


def _make_note(id: str, title: str, status: str, body: str = "", tags: list[str] | None = None):
    """Create a mock note object."""
    note = MagicMock()
    note.id = id
    note.title = title
    note.status = status
    note.body = body
    note.tags = tags or []
    return note


def _make_note_store(notes: list):
    """Create a mock note store."""
    store = MagicMock()
    store.list_notes.return_value = notes
    return store


class TestSyncNotesToBoard:
    def test_adds_new_cards(self, tmp_path):
        notes = [
            _make_note("n1", "Task 1", "todo", "body1"),
            _make_note("n2", "Task 2", "doing", "body2"),
        ]
        note_store = _make_note_store(notes)
        board_store = Store(tmp_path / "board")

        added, updated, total = sync_notes_to_board(note_store, board_store)

        assert added == 2
        assert updated == 0
        assert total == 2

    def test_moves_existing_cards(self, tmp_path):
        board_store = Store(tmp_path / "board")
        card = board_store.create_card("Task 1", column="todo")

        notes = [_make_note("n1", "Task 1", "done")]
        note_store = _make_note_store(notes)

        added, updated, total = sync_notes_to_board(note_store, board_store)

        assert added == 0
        assert updated == 1
        assert total == 1

        got = board_store.get_card(card.id)
        assert got.column == "done"

    def test_creates_hash_trees(self, tmp_path):
        notes = [_make_note("n1", "Task 1", "todo", "body content")]
        note_store = _make_note_store(notes)
        board_store = Store(tmp_path / "board")

        sync_notes_to_board(note_store, board_store)

        card = board_store.get_cards_by_column("todo")[0]
        assert card.root_hash  # should be set

    def test_skips_untitled_notes(self, tmp_path):
        notes = [_make_note("n1", "", "todo")]
        note_store = _make_note_store(notes)
        board_store = Store(tmp_path / "board")

        added, updated, total = sync_notes_to_board(note_store, board_store)

        assert added == 1
        card = board_store.get_cards_by_column("todo")[0]
        assert card.title == "(untitled)"

    def test_no_notes(self, tmp_path):
        note_store = _make_note_store([])
        board_store = Store(tmp_path / "board")

        added, updated, total = sync_notes_to_board(note_store, board_store)

        assert added == 0
        assert updated == 0
        assert total == 0

    def test_status_mapping(self, tmp_path):
        """Test that note statuses map to correct columns."""
        from planner import config
        notes = []
        for status, expected_col in config.STATUS_TO_COLUMN.items():
            if status is None:
                continue  # skip None key
            notes.append(_make_note(f"n-{status}", f"Task {status}", status))

        note_store = _make_note_store(notes)
        board_store = Store(tmp_path / "board")

        added, updated, total = sync_notes_to_board(note_store, board_store)
        assert added == len(notes)

        for note in notes:
            expected_col = config.STATUS_TO_COLUMN.get(note.status.lower(), "todo")
            cards = board_store.get_cards_by_column(expected_col)
            assert len(cards) >= 1, f"No cards in column {expected_col} for status {note.status}"

    def test_idempotent_sync(self, tmp_path):
        """Syncing twice should not create duplicate cards."""
        notes = [_make_note("n1", "Task 1", "todo")]
        note_store = _make_note_store(notes)
        board_store = Store(tmp_path / "board")

        sync_notes_to_board(note_store, board_store)
        sync_notes_to_board(note_store, board_store)

        board = board_store.load_board()
        assert len(board.cards) == 1
