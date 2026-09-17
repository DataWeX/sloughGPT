"""
Tests for the planner store and CLI — board operations.
"""

from __future__ import annotations

import pytest
from app_planner.cli import main as cli_main
from app_planner.store import PlannerStore


@pytest.fixture(autouse=True)
def _reset():
    from app_planner.store import reset_store

    reset_store()
    yield
    reset_store()


@pytest.fixture
def store(tmp_path):
    return PlannerStore(board_dir=tmp_path / "board", notes_dir=tmp_path / "notes")


def _run(store_dir, *parts) -> tuple[int, str]:
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            code = cli_main(["--board-dir", str(store_dir), "--notes-dir", str(store_dir), *parts])
        except SystemExit as e:
            code = e.code or 1
    return code, buf.getvalue()


def _add(store_dir, title, *extra) -> tuple[int, str]:
    return _run(store_dir, "board", "add", title, *extra)


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

    def test_list_cards_by_assignee(self, store):
        store.add_card("Mine", assignee="mana")
        store.add_card("Theirs")
        results = store.list_cards(assignee="mana")
        assert [c.title for c in results] == ["Mine"]

    def test_search_cards(self, store):
        store.add_card("Engine tuning", tags=["kernel"], description="sweep the camshaft")
        store.add_card("Paint the wall", tags=["ui"])
        results = store.search_cards("engine")
        assert len(results) == 1
        assert results[0].title == "Engine tuning"

    def test_search_cards_by_tag(self, store):
        store.add_card("A", tags=["kernel"])
        store.add_card("B", tags=["ui"])
        results = store.search_cards("kernel")
        assert len(results) == 1
        assert results[0].title == "A"

    def test_search_cards_by_description(self, store):
        store.add_card("A", description="spinny bits")
        store.add_card("B", description="nothing")
        results = store.search_cards("spinny")
        assert len(results) == 1

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

    def test_archive_done(self, store):
        store.add_card("Keep", column="in_progress")
        store.add_card("Done one", column="done")
        archived = store.archive_done()
        assert archived == 1
        cards = store.list_cards()
        assert len(cards) == 1
        assert cards[0].title == "Keep"

    def test_archive_done_empty(self, store):
        store.add_card("Keep", column="todo")
        assert store.archive_done() == 0

    def test_block_and_unblock_card(self, store):
        blocker = store.add_card("Blocker")
        blocked = store.add_card("Blocked")
        assert store.block_card(blocked.id, blocker.id) is not None
        assert store.is_blocked(blocked.id) is True
        assert store.unblock_card(blocked.id, blocker.id) is not None
        assert store.is_blocked(blocked.id) is False

    def test_block_duplicate_ignored(self, store):
        blocker = store.add_card("Blocker")
        blocked = store.add_card("Blocked")
        store.block_card(blocked.id, blocker.id)
        store.block_card(blocked.id, blocker.id)
        assert len(store.get_card(blocked.id).blocked_by) == 1

    def test_add_card_with_type(self, store):
        card = store.add_card("Bug fix", card_type="bug")
        assert card.card_type == "bug"

    def test_edit_card_type(self, store):
        card = store.add_card("Task")
        updated = store.update_card(card.id, card_type="feature")
        assert updated.card_type == "feature"

    def test_update_card_ignores_unknown_keys(self, store):
        card = store.add_card("Stable")
        updated = store.update_card(card.id, bogus="x")
        assert updated is not None
        assert store.get_card(card.id).title == "Stable"


# ── CLI tests ────────────────────────────────────────────────────────────


class TestCLI:
    def test_board_add_and_show(self, tmp_path):
        code, out = _add(tmp_path, "My Task", "--column", "todo", "--priority", "high")
        assert code == 0
        assert "Created" in out

        code, out = _run(tmp_path, "board", "show")
        assert code == 0
        assert "My Task" in out

    def test_board_add_with_all_fields(self, tmp_path):
        code, out = _add(
            tmp_path,
            "Ship v2",
            "--column",
            "in_progress",
            "--priority",
            "high",
            "--tags",
            "kernel,os",
            "--description",
            "release the kraken",
        )
        assert code == 0

        store = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path)
        cards = store.list_cards()
        assert len(cards) == 1
        card = cards[0]
        assert card.title == "Ship v2"
        assert card.column == "in_progress"
        assert card.priority == "high"
        assert card.tags == ["kernel", "os"]
        assert "release the kraken" in card.description

    def test_board_move(self, tmp_path):
        store = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path)
        card = store.add_card("Movable", column="todo")
        code, out = _run(tmp_path, "board", "move", card.id, "done")
        assert code == 0
        assert "Moved" in out

    def test_board_delete(self, tmp_path):
        store = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path)
        card = store.add_card("Doomed")
        code, out = _run(tmp_path, "board", "delete", card.id)
        assert code == 0
        assert "Deleted" in out

    def test_board_tags(self, tmp_path):
        _add(tmp_path, "A", "--tags", "kernel,bug")
        code, out = _run(tmp_path, "board", "tags")
        assert code == 0
        assert "kernel" in out

    def test_board_stats(self, tmp_path):
        _add(tmp_path, "One", "--column", "done")
        code, out = _run(tmp_path, "board", "stats")
        assert code == 0
        assert "Total cards: 1" in out

    def test_board_show_empty(self, tmp_path):
        code, out = _run(tmp_path, "board", "show")
        assert code == 0
        assert "(empty)" in out

    def test_board_show_unknown_card(self, tmp_path):
        code, out = _run(tmp_path, "board", "show", "00000000_nope")
        assert code == 1

    def test_board_move_unknown_column(self, tmp_path):
        store = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path)
        card = store.add_card("Stay")
        code, out = _run(tmp_path, "board", "move", card.id, "frozen")
        assert code == 1

    def test_board_delete_unknown(self, tmp_path):
        code, out = _run(tmp_path, "board", "delete", "00000000_nope")
        assert code == 1

    def test_note_new_and_list(self, tmp_path):
        code, out = _run(tmp_path, "note", "new", "My Note", "--tags", "dev")
        assert code == 0
        assert "Created note" in out

        code, out = _run(tmp_path, "note", "list")
        assert code == 0
        assert "My Note" in out

    def test_sync(self, tmp_path):
        store = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path)
        store.create_note("Synced Task", status="wip")
        code, out = _run(tmp_path, "sync")
        assert code == 0
        assert "1 new card" in out

    def test_unknown_command(self, tmp_path):
        code, _ = _run(tmp_path, "nonexistent")
        assert code != 0

    def test_no_command_prints_help(self, tmp_path):
        code, out = _run(tmp_path)
        assert code == 0
        assert "usage:" in out
