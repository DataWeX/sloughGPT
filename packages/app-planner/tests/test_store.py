"""Tests for the planner store and CLI — board operations."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from app_planner.cli import main as cli_main
from app_planner.store import PlannerStore, reset_store


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def store(tmp_dir):
    return PlannerStore(board_dir=tmp_dir / "board", notes_dir=tmp_dir / "notes")


def _run(board_dir: Path, *parts: str) -> tuple[int, str]:
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            code = cli_main(["--board-dir", str(board_dir), "--notes-dir", str(board_dir), *parts])
        except SystemExit as e:
            code = e.code or 1
    return code, buf.getvalue()


def _add(board_dir: Path, title: str, *extra: str) -> tuple[int, str]:
    return _run(board_dir, "board", "add", title, *extra)


# ── Store tests ──────────────────────────────────────────────────────────


class TestPlannerStore:
    def test_add_card(self, store):
        card = store.add_card("Test card", column="todo", priority="high")
        assert card.title == "Test card"
        assert card.column == "todo"
        assert card.priority == "high"
        assert card.id

    def test_get_card(self, store):
        card = store.add_card("Findable")
        found = store.get_card(card.id)
        assert found is not None
        assert found.title == "Findable"

    def test_get_card_missing(self, store):
        assert store.get_card("nonexistent") is None

    def test_update_card(self, store):
        card = store.add_card("Update me")
        updated = store.update_card(card.id, title="Updated", priority="critical")
        assert updated is not None
        assert updated.title == "Updated"
        assert updated.priority == "critical"

    def test_delete_card(self, store):
        card = store.add_card("Delete me")
        assert store.delete_card(card.id) is True
        assert store.get_card(card.id) is None

    def test_delete_card_missing(self, store):
        assert store.delete_card("nonexistent") is False

    def test_move_card(self, store):
        card = store.add_card("Move me", column="todo")
        assert store.move_card(card.id, "done") is True
        assert store.get_card(card.id).column == "done"

    def test_move_card_missing(self, store):
        assert store.move_card("nonexistent", "done") is False

    def test_list_cards(self, store):
        store.add_card("A", column="todo")
        store.add_card("B", column="done")
        assert len(store.list_cards()) == 2
        assert len(store.list_cards(column="todo")) == 1

    def test_search_cards(self, store):
        store.add_card("Engine tuning", tags=["kernel"])
        store.add_card("UI fix")
        results = store.search_cards("engine")
        assert len(results) == 1
        assert results[0].title == "Engine tuning"

    def test_get_tags(self, store):
        store.add_card("A", tags=["kernel", "bug"])
        store.add_card("B", tags=["kernel"])
        tags = store.get_tags()
        assert tags["kernel"] == 2
        assert tags["bug"] == 1

    def test_get_stats(self, store):
        store.add_card("A", column="todo")
        store.add_card("B", column="done")
        stats = store.get_stats()
        assert stats["total"] == 2
        assert stats["byColumn"]["todo"] == 1
        assert stats["byColumn"]["done"] == 1


# ── CLI tests ────────────────────────────────────────────────────────────


class TestCLI:
    def test_board_add_and_show(self, tmp_dir):
        code, out = _add(tmp_dir, "My Task", "--column", "todo", "--priority", "high")
        assert code == 0
        assert "Created" in out

        code, out = _run(tmp_dir, "board", "show")
        assert code == 0
        assert "My Task" in out

    def test_board_move(self, tmp_dir):
        store = PlannerStore(board_dir=tmp_dir, notes_dir=tmp_dir / "notes")
        card = store.add_card("Movable", column="todo")
        code, out = _run(tmp_dir, "board", "move", card.id, "done")
        assert code == 0
        assert "Moved" in out

    def test_board_delete(self, tmp_dir):
        store = PlannerStore(board_dir=tmp_dir, notes_dir=tmp_dir / "notes")
        card = store.add_card("Doomed")
        code, out = _run(tmp_dir, "board", "delete", card.id)
        assert code == 0
        assert "Deleted" in out

    def test_board_tags(self, tmp_dir):
        _add(tmp_dir, "A", "--tags", "kernel,bug")
        code, out = _run(tmp_dir, "board", "tags")
        assert code == 0
        assert "kernel" in out

    def test_board_stats(self, tmp_dir):
        _add(tmp_dir, "One", "--column", "done")
        code, out = _run(tmp_dir, "board", "stats")
        assert code == 0
        assert "Total cards: 1" in out

    def test_note_new_and_list(self, tmp_dir):
        code, out = _run(tmp_dir, "note", "new", "My Note", "--tags", "dev")
        assert code == 0
        assert "Created note" in out

        code, out = _run(tmp_dir, "note", "list")
        assert code == 0
        assert "My Note" in out

    def test_sync(self, tmp_dir):
        # Create note directly without auto-sync
        store = PlannerStore(board_dir=tmp_dir, notes_dir=tmp_dir)
        store.create_note("Synced Task", status="wip")
        code, out = _run(tmp_dir, "sync")
        assert code == 0
        assert "1 new card" in out

    def test_unknown_command(self, tmp_dir):
        code, _ = _run(tmp_dir, "nonexistent")
        assert code != 0
