"""Tests for planner.store — JSONL board operations."""

import json
from pathlib import Path

import pytest

from planner.store import Card, Board, Store, get_store, reset_store


# ══════════════════════════════════════════════════════════════════════════════
# Card dataclass
# ══════════════════════════════════════════════════════════════════════════════


class TestCard:
    def test_defaults(self):
        c = Card()
        assert c.column == "todo"
        assert c.priority == "medium"
        assert c.tags == []

    def test_to_dict_and_back(self):
        c = Card(id="c1", title="Test", column="doing", tags=["bug"])
        d = c.to_dict()
        restored = Card.from_dict(d)
        assert restored.id == "c1"
        assert restored.column == "doing"
        assert restored.tags == ["bug"]

    def test_from_dict_extra_keys_ignored(self):
        d = {"id": "x", "title": "y", "unknown": 123}
        c = Card.from_dict(d)
        assert c.id == "x"


# ══════════════════════════════════════════════════════════════════════════════
# Store
# ══════════════════════════════════════════════════════════════════════════════


class TestStoreCreateCard:
    def test_create_card(self, tmp_path):
        store = Store(tmp_path / "board")
        card = store.create_card("My Task", column="todo", priority="high")
        assert card.title == "My Task"
        assert card.column == "todo"
        assert card.priority == "high"
        assert card.id  # auto-generated

    def test_create_card_with_tags(self, tmp_path):
        store = Store(tmp_path / "board")
        card = store.create_card("Task", tags=["bug", "urgent"])
        assert card.tags == ["bug", "urgent"]

    def test_create_card_persists(self, tmp_path):
        store = Store(tmp_path / "board")
        store.create_card("Task")
        board = store.load_board()
        assert len(board.cards) == 1
        assert board.cards[0].title == "Task"


class TestStoreGetCard:
    def test_get_existing_card(self, tmp_path):
        store = Store(tmp_path / "board")
        created = store.create_card("Task")
        got = store.get_card(created.id)
        assert got is not None
        assert got.title == "Task"

    def test_get_nonexistent_returns_none(self, tmp_path):
        store = Store(tmp_path / "board")
        assert store.get_card("missing") is None


class TestStoreUpdateCard:
    def test_update_card(self, tmp_path):
        store = Store(tmp_path / "board")
        card = store.create_card("Task")
        updated = store.update_card(card.id, title="Updated", priority="low")
        assert updated is not None
        assert updated.title == "Updated"
        assert updated.priority == "low"

    def test_update_nonexistent_returns_none(self, tmp_path):
        store = Store(tmp_path / "board")
        assert store.update_card("missing", title="x") is None


class TestStoreDeleteCard:
    def test_delete_card(self, tmp_path):
        store = Store(tmp_path / "board")
        card = store.create_card("Task")
        assert store.delete_card(card.id) is True
        assert store.get_card(card.id) is None

    def test_delete_nonexistent_returns_false(self, tmp_path):
        store = Store(tmp_path / "board")
        assert store.delete_card("missing") is False


class TestStoreMoveCard:
    def test_move_card(self, tmp_path):
        store = Store(tmp_path / "board")
        card = store.create_card("Task", column="todo")
        assert store.move_card(card.id, "doing") is True
        got = store.get_card(card.id)
        assert got.column == "doing"

    def test_move_card_with_index(self, tmp_path):
        store = Store(tmp_path / "board")
        c1 = store.create_card("Task 1", column="todo")
        c2 = store.create_card("Task 2", column="todo")
        store.move_card(c2.id, "todo", to_index=0)
        board = store.load_board()
        assert board.cards[0].id == c2.id

    def test_move_nonexistent_returns_false(self, tmp_path):
        store = Store(tmp_path / "board")
        assert store.move_card("missing", "doing") is False


class TestStoreQuery:
    def test_get_cards_by_column(self, tmp_path):
        store = Store(tmp_path / "board")
        store.create_card("T1", column="todo")
        store.create_card("T2", column="doing")
        store.create_card("T3", column="todo")
        todo = store.get_cards_by_column("todo")
        assert len(todo) == 2

    def test_get_all_tags(self, tmp_path):
        store = Store(tmp_path / "board")
        store.create_card("T1", tags=["bug", "urgent"])
        store.create_card("T2", tags=["bug"])
        tags = store.get_all_tags()
        assert tags["bug"] == 2
        assert tags["urgent"] == 1

    def test_get_stats(self, tmp_path):
        store = Store(tmp_path / "board")
        store.create_card("T1", column="todo", priority="high")
        store.create_card("T2", column="doing", priority="low")
        stats = store.get_stats()
        assert stats["total"] == 2
        assert stats["byColumn"]["todo"] == 1
        assert stats["byPriority"]["high"] == 1


class TestStoreLoadBoard:
    def test_load_empty_board(self, tmp_path):
        store = Store(tmp_path / "board")
        board = store.load_board()
        assert len(board.cards) == 0

    def test_load_board_with_cards(self, tmp_path):
        store = Store(tmp_path / "board")
        store.create_card("T1")
        store.create_card("T2")
        board = store.load_board()
        assert len(board.cards) == 2


class TestStorePersistence:
    def test_persistence_across_instances(self, tmp_path):
        store1 = Store(tmp_path / "board")
        store1.create_card("Persistent")
        store2 = Store(tmp_path / "board")
        board = store2.load_board()
        assert len(board.cards) == 1
        assert board.cards[0].title == "Persistent"

    def test_corrupted_file_skips_bad_lines(self, tmp_path):
        store = Store(tmp_path / "board")
        board_file = tmp_path / "board" / "board.jsonl"
        board_file.write_text("NOT JSON\n{\"id\":\"ok\",\"title\":\"Good\"}\n")
        board = store.load_board()
        assert len(board.cards) == 1
        assert board.cards[0].title == "Good"


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════


class TestSingleton:
    def test_get_store_returns_same_instance(self, tmp_path):
        reset_store()
        s1 = get_store(tmp_path / "board")
        s2 = get_store(tmp_path / "board")
        assert s1 is s2

    def test_reset_store(self, tmp_path):
        reset_store()
        s1 = get_store(tmp_path / "board")
        reset_store()
        s2 = get_store(tmp_path / "board")
        assert s1 is not s2
