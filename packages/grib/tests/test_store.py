"""Tests for the planner store and CLI — board operations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from grib.cli import main as cli_main
from grib.store import PlannerStore, reset_store


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

    def test_get_stats_reports_dup_diagnostics(self, store):
        """The hash chain does not catch id/title duplication — stats must."""
        from grib.store import _token_of

        solo = store.add_card("Solo")
        store.add_card("Shared")
        store.add_card("Shared", column="done")
        stats = store.get_stats()
        assert stats["uniqueIds"] == 3
        assert stats["dupIdLines"] == 0
        assert stats["titleCollisions"] == 1

        # Simulate historical union-merge damage: a duplicate-id line.
        text = store._board_file.read_text(encoding="utf-8")
        store._atomic_write(text + store._card_line(solo) + "\n", expect=_token_of(text))
        stats = store.get_stats()
        assert stats["total"] == 4
        assert stats["uniqueIds"] == 3
        assert stats["dupIdLines"] == 1
        assert stats["titleCollisions"] == 2


# ── Surgical JSONL write tests ────────────────────────────────────────────


class TestSurgicalBoardUpdates:
    """Board writes must be surgical: untouched content survives every write.

    The real board.jsonl contains mixed note-derived lines
    ({"body","id","priority","status","title"}) and card-schema lines. A full
    re-serialize re-keys every line and drops legacy keys, so the write
    funnels merge the fresh line over the original (legacy ``body`` survives)
    and re-seal the chain (card 20261109_112756). Contract: untouched card
    lines change only by added chain fields — never lost content — and
    non-card lines stay byte-for-byte.
    """

    def _seed_note_schema(self, store, board_file):
        """Write two note-derived lines (as the migrated board has), no header."""
        board_file.write_text(
            json.dumps({"id": "n1", "title": "One", "body": "b1", "priority": "med"})
            + "\n"
            + json.dumps(
                {"id": "n2", "title": "Two", "body": "b2", "priority": "low", "column": "done"}
            )
            + "\n"
        )

    def test_update_only_rewrites_target_line(self, store):
        board_file = store._board_file
        self._seed_note_schema(store, board_file)

        card = store.get_card("n1")
        assert card is not None
        store.update_card("n1", column="done")

        after = board_file.read_text().splitlines()
        assert len(after) == 2
        by_id = {json.loads(line)["id"]: json.loads(line) for line in after}
        # The target's legacy ``body`` survives the surgical rewrite...
        assert by_id["n1"]["body"] == "b1"
        assert by_id["n1"]["column"] == "done"
        # ...the sibling's note-schema content survives the re-seal,
        # gaining only chain fields.
        assert {k: by_id["n2"][k] for k in ("id", "title", "body", "priority", "column")} == {
            "id": "n2",
            "title": "Two",
            "body": "b2",
            "priority": "low",
            "column": "done",
        }
        assert "chain_hash" in by_id["n2"]
        assert store.get_card("n1").column == "done"
        assert store.verify_chain() == []

    def test_move_only_rewrites_target_line(self, store):
        board_file = store._board_file
        self._seed_note_schema(store, board_file)

        store.move_card("n1", "review")

        after = board_file.read_text().splitlines()
        assert len(after) == 2
        by_id = {json.loads(line)["id"]: json.loads(line) for line in after}
        # sibling content survives the re-seal — legacy ``body`` included
        assert by_id["n2"]["body"] == "b2"
        assert by_id["n2"]["priority"] == "low"
        assert by_id["n2"]["column"] == "done"
        assert store.get_card("n1").column == "review"
        assert store.verify_chain() == []

    def test_delete_removes_only_target_line(self, store):
        board_file = store._board_file
        self._seed_note_schema(store, board_file)

        assert store.delete_card("n1") is True

        after = board_file.read_text().splitlines()
        assert len(after) == 1
        # only the target line is gone; the survivor keeps its content
        # through the re-seal (legacy ``body`` included) and is re-chained
        survivor = json.loads(after[0])
        assert survivor["id"] == "n2"
        assert survivor["body"] == "b2"
        assert survivor["priority"] == "low"
        assert survivor["column"] == "done"
        assert "chain_hash" in survivor
        assert store.get_card("n2") is not None
        assert store.verify_chain() == []

    def test_add_appends_without_touching_existing_lines(self, store):
        board_file = store._board_file
        store.add_card("Alpha")
        store.add_card("Beta")
        before = board_file.read_text().splitlines()

        store.add_card("Gamma")

        after = board_file.read_text().splitlines()
        assert len(after) == len(before) + 1
        assert after[: len(before)] == before  # every existing line unchanged
        assert "Gamma" in after[-1]

    def test_archive_done_preserves_other_lines(self, store):
        board_file = store._board_file
        self._seed_note_schema(store, board_file)

        assert store.archive_done() == 1
        after = board_file.read_text().splitlines()
        assert len(after) == 1
        survivor = json.loads(after[0])
        # content intact through archive + re-seal (legacy ``body`` included)
        assert {k: survivor[k] for k in ("id", "title", "body", "priority")} == {
            "id": "n1",
            "title": "One",
            "body": "b1",
            "priority": "med",
        }
        assert store.verify_chain() == []


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
