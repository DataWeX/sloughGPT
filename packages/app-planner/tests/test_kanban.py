"""
Tests for the planner store and CLI — board operations.
"""

from __future__ import annotations

import json
from pathlib import Path

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
        store.add_card("Keep", column="wip")
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
            "wip",
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
        assert card.column == "wip"
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


# ── Column vocabulary ──────────────────────────────────────────────────────


class TestColumnValidation:
    """The column name is a primary key: every write path must reject retired
    or misspelled spellings (in_progress, in-progress, TODO) so the board
    can never drift away from its schema header again."""

    def test_add_rejects_retired_spelling(self, store):
        with pytest.raises(ValueError, match="Invalid column: in_progress"):
            store.add_card("Legacy", column="in_progress")

    def test_add_rejects_case_drift(self, store):
        with pytest.raises(ValueError, match="Invalid column: TODO"):
            store.add_card("Shouty", column="TODO")

    def test_add_accepts_canonical_wip(self, store):
        card = store.add_card("Real work", column="wip")
        assert card.column == "wip"

    def test_rejected_add_writes_nothing(self, store):
        with pytest.raises(ValueError):
            store.add_card("Ghost", column="in_progress")
        assert store.load_board().cards == []

    def test_move_rejects_unknown_column_and_leaves_card(self, store):
        card = store.add_card("Mover", column="todo")
        with pytest.raises(ValueError, match="Invalid column: in-progress"):
            store.move_card(card.id, "in-progress")
        assert store.get_card(card.id).column == "todo"

    def test_update_column_rejects_unknown_and_leaves_card(self, store):
        card = store.add_card("Updater", column="todo")
        with pytest.raises(ValueError, match="Invalid column: in_progress"):
            store.update_card(card.id, column="in_progress")
        assert store.get_card(card.id).column == "todo"

    def test_valid_move_still_works(self, store):
        card = store.add_card("Mover", column="todo")
        assert store.move_card(card.id, "review") is True
        assert store.get_card(card.id).column == "review"

    def test_cli_add_rejects_unknown_column(self, tmp_path):
        code, _ = _add(tmp_path, "Nope", "--column", "in_progress")
        assert code == 1
        store = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path)
        assert store.load_board().cards == []

    def test_cli_move_rejects_unknown_column(self, tmp_path):
        code, _ = _add(tmp_path, "Mover")
        assert code == 0
        store = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path)
        card = store.load_board().cards[0]
        code, _ = _run(tmp_path, "board", "move", card.id, "in_progress")
        assert code == 1
        assert store.get_card(card.id).column == "todo"


class TestReadPathColumnValidation:
    """Defense on READ (card b538ecbd): board.jsonl is hand-editable by
    design, so write-path rejection alone is not enough — load_board() must
    coerce an undeclared column to a real one and warn, instead of producing
    a card that board show never renders while stats counts it (the two
    views then disagree on the total)."""

    HEADER = json.dumps(
        {
            "schema": "planner/1",
            "name": "Main",
            "columns": [
                {"name": "todo"},
                {"name": "wip"},
                {"name": "review"},
                {"name": "done"},
            ],
        }
    )

    def _write_board(self, store, *cards):
        lines = [self.HEADER, *(json.dumps(c) for c in cards)]
        store._board_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_load_board_coerces_undeclared_column(self, store):
        self._write_board(
            store,
            {"id": "good-1", "title": "Good", "column": "wip"},
            {"id": "bad-1", "title": "Hand-edited", "column": "in-progress"},
        )
        cards = {c.id: c for c in store.load_board().cards}
        # The phantom card stays visible (coerced, not dropped) and the
        # healthy card is untouched.
        assert set(cards) == {"good-1", "bad-1"}
        assert cards["bad-1"].column == "todo"
        assert cards["good-1"].column == "wip"

    def test_load_board_warns_on_undeclared_column(self, store, caplog):
        self._write_board(
            store, {"id": "bad-1", "title": "Typo", "column": "in-progres"}
        )
        with caplog.at_level("WARNING", logger="app_planner.store"):
            store.load_board()
        assert any("in-progres" in r.message for r in caplog.records)

    def test_stats_and_load_agree_when_phantom_present(self, store):
        # DONE-WHEN 3: show (header columns) and stats (card counts) must
        # report the same universe once read-path coercion applies.
        self._write_board(
            store,
            {"id": "good-1", "title": "Good", "column": "done"},
            {"id": "bad-1", "title": "Phantom", "column": "in-progress"},
        )
        stats = store.get_stats()
        board = store.load_board()
        assert stats["total"] == len(board.cards) == 2
        assert "in-progress" not in stats["byColumn"]
        assert stats["byColumn"]["todo"] == 1
        declared = {c["name"] for c in board.columns}
        assert set(stats["byColumn"]) <= declared

    def test_repo_board_every_card_in_declared_set(self):
        # Recurrence guard for the REAL artifact (DONE-WHEN 4): three cards
        # sat in a phantom 'in-progress' column for days — invisible to
        # board show, counted by stats, unrepairable by sync (their notes
        # were gone).
        repo_board = Path(__file__).resolve().parents[3] / ".kanban" / "board.jsonl"
        if not repo_board.exists():
            pytest.skip("repo board not present in this checkout")
        store = PlannerStore(board_dir=repo_board.parent)
        board = store.load_board()
        declared = {c["name"] for c in board.columns}
        offenders = [c.id for c in board.cards if c.column not in declared]
        assert offenders == [], f"cards outside declared columns: {offenders}"
