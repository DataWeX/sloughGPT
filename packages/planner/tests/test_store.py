"""
Tests for planner.store — the board CLI (``planner board``),
driven through the Store class with an isolated temp board.
"""

import json
import re

import pytest

from planner.store import Store, Card, Board, reset_store


@pytest.fixture(autouse=True)
def _isolate_store():
    reset_store()


@pytest.fixture
def board_dir(tmp_path):
    return tmp_path / "board"


@pytest.fixture
def store(board_dir):
    return Store(board_dir=board_dir)


def _add(store, title, **kwargs):
    return store.create_card(title=title, **kwargs)


def test_create_card_defaults(store):
    card = _add(store, "Fix boot order")
    assert card.title == "Fix boot order"
    assert card.column == "todo"
    assert card.priority == "medium"
    assert card.tags == []
    assert card.id


def test_create_card_with_fields(store):
    card = _add(store, "Ship v2", column="in_progress", priority="high",
                tags=["kernel", "os"], assignee="mana", dueDate="2026-08-15",
                description="release the kraken")
    assert card.column == "in_progress"
    assert card.priority == "high"
    assert card.tags == ["kernel", "os"]
    assert card.assignee == "mana"
    assert card.dueDate == "2026-08-15"
    assert card.description == "release the kraken"


def test_get_card(store):
    card = _add(store, "Findable")
    found = store.get_card(card.id)
    assert found is not None
    assert found.title == "Findable"


def test_get_card_not_found(store):
    assert store.get_card("nonexistent") is None


def test_update_card(store):
    card = _add(store, "Old name", priority="low")
    updated = store.update_card(card.id, title="New name", priority="high")
    assert updated is not None
    assert updated.title == "New name"
    assert updated.priority == "high"
    assert store.get_card(card.id).title == "New name"


def test_update_card_not_found(store):
    assert store.update_card("nonexistent", title="X") is None


def test_delete_card(store):
    card = _add(store, "Doomed")
    assert store.delete_card(card.id) is True
    assert store.get_card(card.id) is None


def test_delete_card_not_found(store):
    assert store.delete_card("nonexistent") is False


def test_move_card(store):
    card = _add(store, "Move me", column="todo")
    assert store.move_card(card.id, "done") is True
    assert store.get_card(card.id).column == "done"


def test_move_card_not_found(store):
    assert store.move_card("nonexistent", "done") is False


def test_get_cards_by_column(store):
    _add(store, "Todo 1", column="todo")
    _add(store, "Todo 2", column="todo")
    _add(store, "Done 1", column="done")
    todo_cards = store.get_cards_by_column("todo")
    assert len(todo_cards) == 2
    done_cards = store.get_cards_by_column("done")
    assert len(done_cards) == 1


def test_get_all_tags(store):
    _add(store, "Tagged", tags=["kernel", "os"])
    _add(store, "Also tagged", tags=["kernel", "ui"])
    tags = store.get_all_tags()
    assert tags == {"kernel": 2, "os": 1, "ui": 1}


def test_get_stats(store):
    _add(store, "One", column="done", priority="high")
    _add(store, "Two", column="todo", priority="low")
    _add(store, "Three", column="done", priority="medium")
    stats = store.get_stats()
    assert stats["total"] == 3
    assert stats["byColumn"] == {"done": 2, "todo": 1}
    assert stats["byPriority"] == {"high": 1, "low": 1, "medium": 1}


def test_load_board_empty(store):
    board = store.load_board()
    assert board.cards == []


def test_load_board(store):
    _add(store, "Card 1")
    _add(store, "Card 2")
    board = store.load_board()
    assert len(board.cards) == 2


def test_persistence(board_dir):
    store1 = Store(board_dir=board_dir)
    _add(store1, "Persistent card")
    store2 = Store(board_dir=board_dir)
    board = store2.load_board()
    assert len(board.cards) == 1
    assert board.cards[0].title == "Persistent card"


def test_jsonl_format(store, board_dir):
    _add(store, "Line 1")
    _add(store, "Line 2")
    board_file = board_dir / "board.jsonl"
    lines = board_file.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        data = json.loads(line)
        assert "id" in data
        assert "title" in data
