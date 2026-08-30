"""Tests for the Kanban board module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "app-planner" / "src"))
from app_planner.kanban import (  # noqa: E402
    KanbanStore, Card, ColumnDef, Board, Note,
    _make_id, _abbrev, _render_board,
)


def test_make_id() -> None:
    i = _make_id("Fix kernel boot")
    assert i.endswith("fix-kernel-boot")


def test_make_id_special_chars() -> None:
    i = _make_id("Fix #1: kernel's boot!")
    assert "fix" in i
    assert "1" in i
    assert "kernels" in i
    assert "boot" in i


def test_abbrev_short() -> None:
    assert _abbrev("hello", 10) == "hello"


def test_abbrev_long() -> None:
    assert _abbrev("hello world", 5) == "hello..."


def test_card_short_id() -> None:
    c = Card(id="20260729_103000_fix-kernel-boot")
    assert c.short_id == "20260729"


def test_card_priority_icon() -> None:
    assert Card(priority="critical").priority_icon == "!!!"
    assert Card(priority="high").priority_icon == "!!"
    assert Card(priority="medium").priority_icon == "!"
    assert Card(priority="low").priority_icon == " "


def test_card_to_dict_roundtrip() -> None:
    c = Card(id="x", title="Test", priority="high",
             tags=["a", "b"], notes=[Note(id="n1", text="hello")])
    d = c.to_dict()
    c2 = Card.from_dict(d)
    assert c2.title == "Test"
    assert c2.priority == "high"
    assert c2.tags == ["a", "b"]
    assert len(c2.notes) == 1
    assert c2.notes[0].text == "hello"


def test_board_to_dict_roundtrip() -> None:
    b = Board(name="test", columns=[ColumnDef(name="col1", wip_limit=2, order=0)])
    b.cards.append(Card(id="c1", title="A"))
    d = b.to_dict()
    b2 = Board.from_dict(d)
    assert b2.name == "test"
    assert len(b2.columns) == 1
    assert b2.columns[0].name == "col1"
    assert len(b2.cards) == 1
    assert b2.cards[0].title == "A"


def test_board_from_dict_defaults() -> None:
    b = Board.from_dict({})
    assert b.name == "board"
    assert len(b.columns) == 4


class TestKanbanStore:
    def test_init_board(self, tmp_path):
        k = KanbanStore(tmp_path)
        b = k.init_board("mytest")
        assert b.name == "mytest"
        assert len(b.columns) == 4

    def test_init_board_idempotent(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.init_board("first")
        b2 = k.init_board("second")
        assert b2.name == "first"

    def test_init_board_force(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.init_board("first")
        b2 = k.init_board("second", force=True)
        assert b2.name == "second"

    def test_add_card(self, tmp_path):
        k = KanbanStore(tmp_path)
        c = k.add_card("Test card", column="todo")
        assert c.title == "Test card"
        assert c.column == "todo"

    def test_add_card_default_column(self, tmp_path):
        k = KanbanStore(tmp_path)
        c = k.add_card("No column")
        assert c.column == "todo"

    def test_get_card(self, tmp_path):
        k = KanbanStore(tmp_path)
        created = k.add_card("Get me")
        got = k.get_card(created.id)
        assert got is not None
        assert got.title == "Get me"

    def test_get_card_prefix(self, tmp_path):
        k = KanbanStore(tmp_path)
        created = k.add_card("Prefix test")
        got = k.get_card(created.id[:8])
        assert got is not None

    def test_get_card_nonexistent(self, tmp_path):
        k = KanbanStore(tmp_path)
        assert k.get_card("no-such-card") is None

    def test_get_card_ambiguous(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.add_card("Same card")
        k.add_card("Same card too")
        assert k.get_card("same") is None

    def test_update_card_title(self, tmp_path):
        k = KanbanStore(tmp_path)
        c = k.add_card("Old")
        updated = k.update_card(c.id, title="New")
        assert updated is not None
        assert updated.title == "New"

    def test_update_card_priority(self, tmp_path):
        k = KanbanStore(tmp_path)
        c = k.add_card("Pri", priority="low")
        updated = k.update_card(c.id, priority="high")
        assert updated is not None
        assert updated.priority == "high"

    def test_update_card_nonexistent(self, tmp_path):
        k = KanbanStore(tmp_path)
        assert k.update_card("no-such", title="Nope") is None

    def test_delete_card(self, tmp_path):
        k = KanbanStore(tmp_path)
        c = k.add_card("Delete me")
        assert k.delete_card(c.id) is True
        assert k.get_card(c.id) is None

    def test_delete_card_prefix(self, tmp_path):
        k = KanbanStore(tmp_path)
        c = k.add_card("Delete prefix")
        assert k.delete_card(c.id[:8]) is True

    def test_delete_card_nonexistent(self, tmp_path):
        k = KanbanStore(tmp_path)
        assert k.delete_card("no-such") is False

    def test_move_card(self, tmp_path):
        k = KanbanStore(tmp_path)
        c = k.add_card("Move me")
        moved = k.move_card(c.id, "in_progress")
        assert moved is not None
        assert moved.column == "in_progress"

    def test_move_card_unknown_column(self, tmp_path):
        k = KanbanStore(tmp_path)
        c = k.add_card("Bad move")
        assert k.move_card(c.id, "bogus") is None

    def test_move_card_nonexistent(self, tmp_path):
        k = KanbanStore(tmp_path)
        assert k.move_card("no-such", "done") is None

    def test_list_cards(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.add_card("A", column="todo")
        k.add_card("B", column="todo")
        assert len(k.list_cards()) == 2

    def test_list_cards_filter_column(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.add_card("A", column="todo")
        k.add_card("B", column="done")
        assert len(k.list_cards(column="todo")) == 1

    def test_list_cards_filter_priority(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.add_card("Low", priority="low")
        k.add_card("High", priority="high")
        assert len(k.list_cards(priority="high")) == 1

    def test_list_cards_filter_tag(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.add_card("A", tags=["bug"])
        k.add_card("B", tags=["feature"])
        assert len(k.list_cards(tag="bug")) == 1

    def test_list_cards_filter_assignee(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.add_card("A", assignee="alice")
        k.add_card("B", assignee="bob")
        assert len(k.list_cards(assignee="alice")) == 1

    def test_list_cards_limit(self, tmp_path):
        k = KanbanStore(tmp_path)
        for i in range(10):
            k.add_card(f"Card {i}")
        assert len(k.list_cards(limit=3)) == 3

    def test_search_cards(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.add_card("Fix kernel")
        k.add_card("Add VFS")
        assert len(k.search_cards("kernel")) == 1

    def test_search_cards_description(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.add_card("Task", description="This is about networking")
        assert len(k.search_cards("networking")) == 1

    def test_search_cards_tags(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.add_card("Task", tags=["bugfix", "kernel"])
        assert len(k.search_cards("bugfix")) == 1

    def test_search_cards_limit(self, tmp_path):
        k = KanbanStore(tmp_path)
        for i in range(10):
            k.add_card(f"Kernel bug {i}")
        assert len(k.search_cards("kernel", limit=3)) == 3

    def test_search_no_match(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.add_card("Hello")
        assert k.search_cards("xyzzy_nonexistent") == []

    def test_add_note(self, tmp_path):
        k = KanbanStore(tmp_path)
        c = k.add_card("Note test")
        n = k.add_note(c.id, "Looking into this", author="me")
        assert n is not None
        assert n.text == "Looking into this"
        assert n.author == "me"

    def test_add_note_nonexistent_card(self, tmp_path):
        k = KanbanStore(tmp_path)
        assert k.add_note("no-such", "hello") is None

    def test_list_notes(self, tmp_path):
        k = KanbanStore(tmp_path)
        c = k.add_card("Notes")
        k.add_note(c.id, "First")
        k.add_note(c.id, "Second")
        notes = k.list_notes(c.id)
        assert notes is not None
        assert len(notes) == 2

    def test_list_notes_nonexistent(self, tmp_path):
        k = KanbanStore(tmp_path)
        assert k.list_notes("no-such") is None

    def test_delete_note(self, tmp_path):
        k = KanbanStore(tmp_path)
        c = k.add_card("Delete note")
        n = k.add_note(c.id, "Remove me")
        assert k.delete_note(c.id, n.id[:8]) is True
        assert len(k.list_notes(c.id)) == 0

    def test_delete_note_nonexistent(self, tmp_path):
        k = KanbanStore(tmp_path)
        c = k.add_card("No notes")
        assert k.delete_note(c.id, "no-such") is False

    def test_archive_done(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.add_card("A", column="done")
        k.add_card("B", column="todo")
        assert k.archive_done() == 1
        assert len(k.list_cards()) == 1

    def test_archive_done_empty(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.add_card("A", column="todo")
        assert k.archive_done() == 0

    def test_column_operations(self, tmp_path):
        k = KanbanStore(tmp_path)
        cols = k.list_columns()
        assert len(cols) == 4

        col = k.add_column("blocked", wip_limit=1)
        assert col is not None
        assert col.name == "blocked"
        assert len(k.list_columns()) == 5

        assert k.add_column("blocked") is None

        assert k.rename_column("blocked", "blocked_by") is True
        assert k.rename_column("no-such", "x") is False

        assert k.remove_column("blocked_by") is True
        assert len(k.list_columns()) == 4
        assert k.remove_column("no-such") is False

    def test_remove_column_moves_cards(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.add_card("A", column="review")
        k.remove_column("review", move_to="todo")
        cards = k.list_cards()
        assert cards[0].column == "todo"

    def test_stats(self, tmp_path):
        k = KanbanStore(tmp_path)
        k.add_card("A", column="todo", priority="high")
        k.add_card("B", column="done", priority="low")
        s = k.stats()
        assert s["total"] == 2
        assert len(s["columns"]) == 2
        assert len(s["priorities"]) == 2
        assert s["column_count"] == 4

    def test_stats_empty(self, tmp_path):
        k = KanbanStore(tmp_path)
        s = k.stats()
        assert s["total"] == 0
        assert s["column_count"] == 4


def test_render_board_empty(tmp_path):
    k = KanbanStore(tmp_path)
    board = k.load_board()
    out = _render_board(board)
    assert "card(s)" in out


def test_render_board_with_cards(tmp_path):
    k = KanbanStore(tmp_path)
    k.add_card("Test card")
    out = _render_board(k.load_board())
    assert "TODO" in out
    assert "Test" in out
    assert "card(s)" in out


def test_render_board_no_columns(tmp_path):
    board = Board(columns=[])
    out = _render_board(board)
    assert "No columns" in out


def test_get_kanban_store(tmp_path):
    from app_planner.kanban import get_kanban_store, reset_kanban_store
    reset_kanban_store()
    s1 = get_kanban_store(tmp_path)
    s2 = get_kanban_store()
    assert s1 is s2


def test_reset_kanban_store(tmp_path):
    from app_planner.kanban import get_kanban_store, reset_kanban_store
    reset_kanban_store()
    s1 = get_kanban_store(tmp_path)
    reset_kanban_store()
    s2 = get_kanban_store(tmp_path)
    assert s1 is not s2
