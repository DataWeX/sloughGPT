"""Tests for the planner store and CLI — board operations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from app_planner.cli import main as cli_main
from app_planner.store import (
    BoardIntegrityError,
    PlannerStore,
    dedupe_board,
    duplicate_ids,
    reset_store,
)


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


# ── Surgical JSONL write tests ────────────────────────────────────────────


class TestSurgicalBoardUpdates:
    """Board writes must be surgical: only the touched card's line changes.

    The real board.jsonl contains mixed note-derived lines
    ({"body","id","priority","status","title"}) and card-schema lines. A full
    re-serialize re-keys every line, producing whole-file diffs. These tests
    pin byte-for-byte preservation of untouched lines.
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
        before = board_file.read_text().splitlines()

        card = store.get_card("n1")
        assert card is not None
        store.update_card("n1", column="done")

        after = board_file.read_text().splitlines()
        assert len(after) == 2
        # sibling note-schema line preserved byte-for-byte
        assert after[1] == before[1]
        assert json.loads(after[1]) == {
            "id": "n2",
            "title": "Two",
            "body": "b2",
            "priority": "low",
            "column": "done",
        }
        assert store.get_card("n1").column == "done"

    def test_move_only_rewrites_target_line(self, store):
        board_file = store._board_file
        self._seed_note_schema(store, board_file)
        before = board_file.read_text().splitlines()

        store.move_card("n1", "review")

        after = board_file.read_text().splitlines()
        assert after[1] == before[1]
        assert store.get_card("n1").column == "review"

    def test_delete_removes_only_target_line(self, store):
        board_file = store._board_file
        self._seed_note_schema(store, board_file)
        before = board_file.read_text().splitlines()

        assert store.delete_card("n1") is True

        after = board_file.read_text().splitlines()
        assert len(after) == 1
        assert after[0] == before[1]  # surviving sibling byte-identical
        assert store.get_card("n2") is not None

    def test_add_appends_without_touching_existing_lines(self, store):
        board_file = store._board_file
        store.add_card("Alpha")
        store.add_card("Beta")
        before = board_file.read_text().splitlines()

        store.add_card("Gamma")

        after = board_file.read_text().splitlines()
        assert len(after) == len(before) + 1
        # Existing card lines stay byte-for-byte identical; only the header's
        # chain aggregates may refresh (Gamma opens a new slot -> new tray root).
        before_cards = [raw for raw in before if json.loads(raw).get("title")]
        after_cards = [raw for raw in after if json.loads(raw).get("title")]
        assert after_cards[: len(before_cards)] == before_cards
        assert json.loads(after[0]).get("schema") == "planner/1"
        assert "Gamma" in after[-1]

    def test_archive_done_preserves_other_lines(self, store):
        board_file = store._board_file
        self._seed_note_schema(store, board_file)

        assert store.archive_done() == 1
        after = board_file.read_text().splitlines()
        assert len(after) == 1
        assert json.loads(after[0]) == {"id": "n1", "title": "One", "body": "b1", "priority": "med"}


class TestDuplicateIds:
    """A targeted write must touch ONE line per id, never the whole group.

    The live board carries 13 ids with 2-4 lines each (identical titles; 4
    groups have diverged notes/assignee, 3 of those also conflict on column).
    move/update/delete pick their target through load_board(), which returns
    the FIRST line for an id — so the rewrite must land on that same line.
    Rewriting every matching line collapsed the group into N copies of one
    card, destroying the siblings' column and notes payload.

    Seeded without a schema header on purpose: apply_board() is a documented
    quiet no-op on headerless boards, so sibling lines stay byte-stable and
    these assertions can be exact.
    """

    @staticmethod
    def _seed_dupes(board_file) -> list[str]:
        """Three lines sharing one id, distinguishable only by column."""
        board_file.write_text(
            json.dumps({"id": "dup", "title": "Alpha", "column": "todo"})
            + "\n"
            + json.dumps({"id": "dup", "title": "Alpha", "column": "in_progress"})
            + "\n"
            + json.dumps({"id": "dup", "title": "Alpha", "column": "done"})
            + "\n"
        )
        return board_file.read_text().splitlines()

    @staticmethod
    def _columns(board_file) -> list[str]:
        return [
            json.loads(raw)["column"]
            for raw in board_file.read_text().splitlines()
            if raw.strip() and json.loads(raw).get("title")
        ]

    def test_move_rewrites_only_first_line_of_group(self, store):
        board_file = store._board_file
        before = self._seed_dupes(board_file)

        store.move_card("dup", "review")

        after = board_file.read_text().splitlines()
        assert len(after) == 3, "duplicate-id group must not collapse"
        assert self._columns(board_file) == ["review", "in_progress", "done"]
        # siblings byte-identical — the write landed only on the line
        # load_board() selected
        assert after[1] == before[1]
        assert after[2] == before[2]

    def test_update_rewrites_only_first_line_of_group(self, store):
        board_file = store._board_file
        before = self._seed_dupes(board_file)

        store.update_card("dup", assignee="session")

        after = board_file.read_text().splitlines()
        assert len(after) == 3
        assert json.loads(after[0])["assignee"] == "session"
        assert after[1] == before[1]
        assert after[2] == before[2]

    def test_delete_removes_only_first_line_of_group(self, store):
        board_file = store._board_file
        before = self._seed_dupes(board_file)

        assert store.delete_card("dup") is True

        after = board_file.read_text().splitlines()
        assert len(after) == 2, "delete must remove one line, not the group"
        # the two surviving siblings are byte-identical to what preceded them
        assert after == before[1:]
        assert self._columns(board_file) == ["in_progress", "done"]

    def test_repeat_move_is_stable_across_the_group(self, store):
        """Moving twice must not walk through (or flatten) the siblings."""
        board_file = store._board_file
        self._seed_dupes(board_file)

        store.move_card("dup", "review")
        store.move_card("dup", "done")

        assert self._columns(board_file) == ["done", "in_progress", "done"]
        assert len(board_file.read_text().splitlines()) == 3


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


def _line(card_id: str, column: str = "todo", **extra) -> str:
    return json.dumps({"id": card_id, "title": "T", "column": column, **extra})


class TestBoardIdIntegrity:
    """The board must never gain a second line carrying one card id."""

    def test_write_refuses_a_new_duplicate_id(self, store):
        board_file = store._board_file
        board_file.write_text(_line("alpha") + "\n")
        before = board_file.read_text()

        with pytest.raises(BoardIntegrityError, match="alpha"):
            store._atomic_write(before + _line("alpha", column="done") + "\n")

        # nothing was persisted — the guard fires before the tmp->fsync->replace
        assert board_file.read_text() == before

    def test_pre_existing_duplicates_are_grandfathered(self, store):
        """Repair must not wedge a board that is already corrupt.

        The live board carries 11 such groups; banning duplicates outright
        would refuse every write sessions make against it.
        """
        board_file = store._board_file
        board_file.write_text(_line("dup") + "\n" + _line("dup", column="done") + "\n")

        assert store.move_card("dup", "review") is True

        # the write actually landed (first line of the group) ...
        first = json.loads(board_file.read_text().splitlines()[0])
        assert first["column"] == "review"
        # ... and the pre-existing duplicate was allowed to persist
        assert store.duplicate_board_ids() == {"dup"}

    def test_append_cannot_reuse_an_id_already_on_the_board(self, store):
        board_file = store._board_file
        board_file.write_text(_line("alpha") + "\n")
        stale = PlannerStore(board_dir=store._board_dir, notes_dir=store._notes_dir)
        stale_card = stale.add_card("Existing")
        # rewrite its id to one already present, simulating a re-append of a
        # card object that another writer has already landed
        stale_card.id = "alpha"

        with pytest.raises(BoardIntegrityError, match="alpha"):
            store._append_card(stale_card)

        assert board_file.read_text().count('"alpha"') == 1

    def test_duplicate_ids_skips_header_and_unparsable_lines(self, store):
        board_file = store._board_file
        board_file.write_text(
            json.dumps({"schema": "planner/1", "columns": []})
            + "\n"
            + _line("alpha")
            + "\n"
            + "{not json"
            + "\n"
            + _line("alpha")
            + "\n"
        )
        assert duplicate_ids(board_file) == {"alpha"}

    def test_duplicate_ids_on_a_missing_board_is_empty(self, store):
        assert duplicate_ids(store._board_file) == set()


class TestDedupeBoard:
    """Remediation plans by default and never touches bytes unasked."""

    def _seed(self, board_file) -> list[str]:
        """One duplicate pair + one unique card. The pair is byte-distinguishable."""
        board_file.write_text(
            _line("dup", slot_position=1, updated_at="2026-10-02T07:53:16", notes=["old"])
            + "\n"
            + _line(
                "dup",
                column="done",
                slot_position=2,
                updated_at="2026-10-11T02:14:39",
                notes=["new"],
            )
            + "\n"
            + _line("solo", updated_at="2026-10-05T00:00:00")
            + "\n"
        )
        return board_file.read_text().splitlines()

    def test_plan_only_reports_without_writing(self, store):
        before = self._seed(store._board_file)

        plan = dedupe_board(store._board_file)

        assert plan["groups"] == 1
        assert plan["removed"] == 1
        assert plan["applied"] is False
        assert plan["details"][0]["id"] == "dup"
        assert plan["details"][0]["lines"] == [1, 2]
        assert store._board_file.read_text().splitlines() == before

    def test_apply_keeps_first_slot_but_newest_content(self, store):
        """Position preserved, bytes of the newest copy survive.

        Position matters because sync() maps the Nth note to the Nth card by
        title; bytes matter because legacy lines carry notes/slot_* fields
        Card does not model and would silently strip.
        """
        self._seed(store._board_file)

        plan = dedupe_board(store._board_file, apply=True)

        assert plan["applied"] is True
        assert plan["removed"] == 1
        lines = store._board_file.read_text().splitlines()
        assert len(lines) == 2
        # first slot still holds the dup card ...
        survivor = json.loads(lines[0])
        assert survivor["id"] == "dup"
        # ... but with the NEWEST copy's content, legacy fields intact
        assert survivor["updated_at"] == "2026-10-11T02:14:39"
        assert survivor["notes"] == ["new"]
        assert survivor["slot_position"] == 2
        assert duplicate_ids(store._board_file) == set()

    def test_apply_writes_through_the_guarded_funnel(self, store):
        """The repair itself is subject to the same ratchet it restores."""
        self._seed(store._board_file)

        dedupe_board(store._board_file, apply=True)

        assert duplicate_ids(store._board_file) == set()
        # and a board with no duplicates now passes a fresh write
        store.move_card("solo", "review")
        assert store.duplicate_board_ids() == set()

    def test_apply_is_a_noop_on_a_clean_board(self, store):
        store._board_file.write_text(_line("a") + "\n" + _line("b") + "\n")
        before = store._board_file.read_text()

        plan = dedupe_board(store._board_file, apply=True)

        assert plan["groups"] == 0
        assert plan["removed"] == 0
        assert plan["applied"] is False
        assert store._board_file.read_text() == before

    def test_missing_board_is_a_noop(self, store):
        plan = dedupe_board(store._board_file)
        assert plan == {"groups": 0, "removed": 0, "applied": False, "details": []}


class TestBoardDedupeCLI:
    """`app-planner board dedupe` — plan by default, --apply to write."""

    def _seed(self, board_dir: Path) -> list[str]:
        board_dir.mkdir(parents=True, exist_ok=True)
        (board_dir / "board.jsonl").write_text(
            _line("dup", updated_at="2026-10-02T07:53:16")
            + "\n"
            + _line("dup", column="done", updated_at="2026-10-11T02:14:39")
            + "\n"
            + _line("solo")
            + "\n"
        )
        return (board_dir / "board.jsonl").read_text().splitlines()

    def test_plan_is_read_only_and_exits_one_on_drift(self, tmp_dir):
        before = self._seed(tmp_dir)

        code, out = _run(tmp_dir, "board", "dedupe")

        assert code == 1
        assert "duplicate-id group" in out
        assert "Plan only" in out
        assert (tmp_dir / "board.jsonl").read_text().splitlines() == before

    def test_apply_removes_the_lines_and_exits_zero(self, tmp_dir):
        self._seed(tmp_dir)

        code, out = _run(tmp_dir, "board", "dedupe", "--apply")

        assert code == 0
        assert "sync" in out
        assert duplicate_ids(tmp_dir / "board.jsonl") == set()
        assert len((tmp_dir / "board.jsonl").read_text().splitlines()) == 2

    def test_clean_board_exits_zero(self, tmp_dir):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "board.jsonl").write_text(_line("a") + "\n" + _line("b") + "\n")

        code, out = _run(tmp_dir, "board", "dedupe")

        assert code == 0
        assert "no duplicate" in out

    def test_kanban_alias_works(self, tmp_dir):
        self._seed(tmp_dir)
        code, _ = _run(tmp_dir, "kanban", "dedupe")
        assert code == 1
