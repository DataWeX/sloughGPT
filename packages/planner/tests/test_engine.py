"""Tests for the standalone kanban rendering engine."""

from __future__ import annotations

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from planner.engine import (
    ANSI, Box, Card, Column, Board, Store, Renderer, KanbanEngine, main,
)


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_board() -> Board:
    return Board(
        name="Test Board",
        columns=[
            Column(name="todo", wip_limit=5, order=0),
            Column(name="in_progress", wip_limit=3, order=1),
            Column(name="done", wip_limit=0, order=2),
        ],
        cards=[
            Card(id="card-1", title="Fix login", column="todo", priority="high",
                 tags=["frontend", "bug"], description="Login form broken"),
            Card(id="card-2", title="Add API", column="in_progress", priority="medium",
                 tags=["backend"]),
            Card(id="card-3", title="Write docs", column="done", priority="low"),
            Card(id="card-4", title="Old task", column="todo", priority="none",
                 trashed=True, trashed_at="2026-01-15T10:00:00Z"),
        ],
    )


@pytest.fixture
def tmp_board(tmp_path, sample_board) -> str:
    path = str(tmp_path / "board.jsonl")
    store = Store(path)
    store.save(sample_board)
    return path


# ── Card ───────────────────────────────────────────────────────────────

class TestCard:
    def test_short_id(self):
        c = Card(id="abcdef123456", title="Test")
        assert c.short_id() == "abcdef12"

    def test_short_id_short(self):
        c = Card(id="abc", title="Test")
        assert c.short_id() == "abc"

    def test_priority_icon_high(self):
        assert Card(id="x", title="x", priority="high").priority_icon() == "!!!"

    def test_priority_icon_medium(self):
        assert Card(id="x", title="x", priority="medium").priority_icon() == "!!"

    def test_priority_icon_low(self):
        assert Card(id="x", title="x", priority="low").priority_icon() == "!"

    def test_priority_icon_none(self):
        assert Card(id="x", title="x", priority="none").priority_icon() == " "

    def test_priority_color_returns_ansi(self):
        c = Card(id="x", title="x", priority="high")
        assert c.priority_color() in (ANSI.red, ANSI.yellow, ANSI.green, ANSI.gray)


# ── Board ──────────────────────────────────────────────────────────────

class TestBoard:
    def test_cards_in_column(self, sample_board):
        todo = sample_board.cards_in_column("todo")
        # card-4 is trashed, excluded
        assert len(todo) == 1
        assert todo[0].id == "card-1"

    def test_cards_in_column_include_trashed(self, sample_board):
        todo = sample_board.cards_in_column("todo", include_trashed=True)
        assert len(todo) == 2

    def test_cards_in_column_sorted_by_priority(self, sample_board):
        # Add a low-priority card to todo
        sample_board.cards.append(
            Card(id="card-5", title="Low task", column="todo", priority="low")
        )
        todo = sample_board.cards_in_column("todo")
        assert todo[0].priority == "high"
        assert todo[1].priority == "low"

    def test_trashed_cards(self, sample_board):
        trashed = sample_board.trashed_cards()
        assert len(trashed) == 1
        assert trashed[0].id == "card-4"

    def test_all_tags(self, sample_board):
        tags = sample_board.all_tags()
        assert "frontend" in tags
        assert "bug" in tags
        assert "backend" in tags

    def test_stats(self, sample_board):
        stats = sample_board.stats()
        assert stats["total"] == 3  # excluding trashed
        assert stats["trashed"] == 1
        assert stats["by_column"]["todo"] == 1
        assert stats["by_column"]["in_progress"] == 1
        assert stats["by_column"]["done"] == 1

    def test_empty_board(self):
        board = Board()
        assert board.cards_in_column("todo") == []
        assert board.trashed_cards() == []
        assert board.all_tags() == []
        assert board.stats()["total"] == 0


# ── Store ──────────────────────────────────────────────────────────────

class TestStore:
    def test_load_nonexistent(self, tmp_path):
        store = Store(str(tmp_path / "nope.jsonl"))
        board = store.load()
        assert board.name == "Kanban"
        assert board.cards == []

    def test_save_and_load(self, tmp_board):
        store = Store(tmp_board)
        board = store.load()
        assert board.name == "Test Board"
        assert len(board.cards) == 4
        assert len(board.columns) == 3

    def test_trash_persistence(self, tmp_board):
        store = Store(tmp_board)
        board = store.load()
        card = board.cards_in_column("todo")[0]
        card.trashed = True
        card.trashed_at = "2026-02-01T00:00:00Z"
        store.save(board)

        board2 = store.load()
        # card-4 was already trashed in fixture, now card-1 too = 2
        assert len(board2.trashed_cards()) == 2

    def test_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        open(path, "w").close()
        store = Store(path)
        board = store.load()
        assert board.cards == []


# ── Renderer ───────────────────────────────────────────────────────────

class TestRenderer:
    def test_render_board(self, sample_board):
        r = Renderer(width=80, color=False)
        output = r.render_board(sample_board)
        assert "Test Board" in output
        assert "TODO" in output
        assert "IN PROGRESS" in output
        assert "DONE" in output
        assert "Fix login" in output

    def test_render_with_column_filter(self, sample_board):
        r = Renderer(width=80, color=False)
        output = r.render_board(sample_board, column_filter="todo")
        assert "TODO" in output
        assert "IN PROGRESS" not in output
        assert "Fix login" in output

    def test_render_with_search(self, sample_board):
        r = Renderer(width=80, color=False)
        output = r.render_board(sample_board, search="login")
        # Search highlights (with ANSI in color mode, plain text in no-color)
        assert "Fix login" in output

    def test_render_trash(self, sample_board):
        r = Renderer(width=80, color=False)
        output = r.render_board(sample_board, show_trash=True)
        assert "TRASH" in output
        assert "Old task" in output

    def test_render_empty_board(self):
        r = Renderer(width=80, color=False)
        board = Board()
        output = r.render_board(board)
        assert "No columns" in output

    def test_render_no_cards(self):
        r = Renderer(width=80, color=False)
        board = Board(
            name="Empty",
            columns=[Column(name="todo", order=0)],
        )
        output = r.render_board(board)
        assert "Empty" in output
        assert "0 card" in output

    def test_render_wip_over_limit(self):
        r = Renderer(width=80, color=False)
        board = Board(
            name="WIP Test",
            columns=[Column(name="todo", wip_limit=2, order=0)],
            cards=[
                Card(id="a", title="A", column="todo"),
                Card(id="b", title="B", column="todo"),
                Card(id="c", title="C", column="todo"),
            ],
        )
        output = r.render_board(board)
        assert "OVER" in output

    def test_strip_ansi(self):
        r = Renderer()
        text = f"{ANSI.red}hello{ANSI.reset}"
        assert r._strip_ansi(text) == "hello"

    def test_truncate(self):
        r = Renderer()
        assert r._truncate("hello world", 5) == "hell…"
        assert r._truncate("hi", 10) == "hi"
        assert r._truncate("hello", 5) == "hello"
        assert r._truncate("hello", 3) == "hel"

    def test_pad_raw_accounts_ansi(self):
        r = Renderer(color=True)
        text = f"{ANSI.red}ab{ANSI.reset}"
        padded = r._pad_raw(text, 5)
        assert r._strip_ansi(padded) == "ab   "


# ── KanbanEngine ───────────────────────────────────────────────────────

class TestKanbanEngine:
    def test_from_file(self, tmp_board):
        engine = KanbanEngine.from_file(tmp_board)
        assert engine.board.name == "Test Board"
        assert engine.store is not None

    def test_render(self, tmp_board):
        engine = KanbanEngine.from_file(tmp_board)
        engine.renderer.color = False
        output = engine.render()
        assert "Test Board" in output
        assert "Fix login" in output

    def test_render_card_detail(self, tmp_board):
        engine = KanbanEngine.from_file(tmp_board)
        engine.renderer.color = False
        card = engine.board.cards[0]
        output = engine.render_card_detail(card)
        assert "Fix login" in output
        assert "card-1" in output
        assert "high" in output

    def test_search_filter(self, tmp_board):
        engine = KanbanEngine.from_file(tmp_board)
        engine.renderer.color = False
        engine._search_query = "login"
        output = engine.render()
        assert "Fix login" in output

    def test_column_filter(self, tmp_board):
        engine = KanbanEngine.from_file(tmp_board)
        engine.renderer.color = False
        engine._column_filter = "done"
        output = engine.render()
        assert "DONE" in output
        assert "Write docs" in output

    def test_trash_view(self, tmp_board):
        engine = KanbanEngine.from_file(tmp_board)
        engine.renderer.color = False
        engine._show_trash = True
        output = engine.render()
        assert "TRASH" in output
        assert "Old task" in output

    def test_initial_state(self, tmp_board):
        engine = KanbanEngine.from_file(tmp_board)
        assert engine._cursor_col == 0
        assert engine._cursor_row == 0
        assert engine._selected_card is None
        assert engine._search_query == ""
        assert engine._column_filter == ""
        assert engine._show_trash is False


# ── CLI ────────────────────────────────────────────────────────────────

class TestCLI:
    def test_main_renders(self, tmp_board, capsys):
        ret = main([tmp_board, "--no-color"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Test Board" in captured.out

    def test_main_trash_flag(self, tmp_board, capsys):
        ret = main([tmp_board, "--no-color", "--trash"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "TRASH" in captured.out

    def test_main_column_filter(self, tmp_board, capsys):
        ret = main([tmp_board, "--no-color", "--column", "todo"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "TODO" in captured.out

    def test_main_search(self, tmp_board, capsys):
        ret = main([tmp_board, "--no-color", "--search", "login"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Fix login" in captured.out


# ── ANSI ───────────────────────────────────────────────────────────────

class TestANSI:
    def test_cursor_to(self):
        assert ANSI.cursor_to(5, 10) == "\033[5;10H"

    def test_clear_screen(self):
        assert ANSI.clear_screen() == "\033[2J\033[H"

    def test_hide_show_cursor(self):
        assert ANSI.hide_cursor() == "\033[?25l"
        assert ANSI.show_cursor() == "\033[?25h"

    def test_rgb_fg(self):
        assert ANSI.rgb_fg(255, 0, 128) == "\033[38;2;255;0;128m"

    def test_rgb_bg(self):
        assert ANSI.rgb_bg(0, 255, 0) == "\033[48;2;0;255;0m"


# ── Box ────────────────────────────────────────────────────────────────

class TestBox:
    def test_corners(self):
        assert Box.TL == "┌"
        assert Box.TR == "┐"
        assert Box.BL == "└"
        assert Box.BR == "┘"

    def test_lines(self):
        assert Box.H == "─"
        assert Box.V == "│"

    def test_double(self):
        assert Box.DOUBLE_TL == "╔"
        assert Box.DOUBLE_BR == "╝"
