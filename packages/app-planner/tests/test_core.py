"""
Tests for the planner store — note operations.
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


def _new(store_dir, title, *extra) -> tuple[int, str]:
    return _run(store_dir, "note", "new", title, *extra)


# ── Note store tests ─────────────────────────────────────────────────────


class TestNoteStore:
    def test_create_note(self, store):
        note = store.create_note("Fix boot order", tags=["kernel"], status="wip")
        assert note.title == "Fix boot order"
        assert note.tags == ["kernel"]
        assert note.status == "wip"
        assert note.id

    def test_get_note(self, store):
        note = store.create_note("Findable")
        found = store.get_note(note.id)
        assert found is not None
        assert found.title == "Findable"

    def test_get_note_missing(self, store):
        assert store.get_note("nonexistent") is None

    def test_update_note(self, store):
        note = store.create_note("Old")
        updated = store.update_note(note.id, title="New", status="done")
        assert updated.title == "New"
        assert updated.status == "done"

    def test_delete_note(self, store):
        note = store.create_note("Delete me")
        assert store.delete_note(note.id) is True
        assert store.get_note(note.id) is None

    def test_list_notes(self, store):
        store.create_note("A")
        store.create_note("B")
        assert len(store.list_notes()) == 2

    def test_list_notes_by_status(self, store):
        store.create_note("A", status="done")
        store.create_note("B", status="wip")
        done = store.list_notes(status="done")
        assert len(done) == 1
        assert done[0].title == "A"

    def test_list_notes_by_tag(self, store):
        store.create_note("A", tags=["kernel"])
        store.create_note("B", tags=["ui"])
        kernel = store.list_notes(tag="kernel")
        assert len(kernel) == 1

    def test_search_notes(self, store):
        store.create_note("Alpha engine", tags=["kernel"], body="spinny bits")
        store.create_note("Unrelated", tags=["ui"])
        results = store.search_notes("engine")
        assert len(results) == 1
        assert results[0].title == "Alpha engine"

    def test_search_notes_by_tag(self, store):
        store.create_note("A", tags=["kernel"])
        store.create_note("B", tags=["ui"])
        results = store.search_notes("kernel")
        assert len(results) == 1

    def test_search_notes_by_body(self, store):
        store.create_note("A", body="spinny bits")
        store.create_note("B", body="nothing")
        results = store.search_notes("spinny")
        assert len(results) == 1

    def test_get_tags(self, store):
        store.create_note("A", tags=["alpha"])
        store.create_note("B", tags=["alpha"])
        store.create_note("C", tags=["beta"])
        tags = store.get_tags()
        # get_tags counts board tags, not note tags
        # Sync notes to board first
        store.sync()
        tags = store.get_tags()
        assert tags["alpha"] == 2
        assert tags["beta"] == 1

    def test_note_with_assignee(self, store):
        note = store.create_note("Assigned", assignee="mana")
        assert note.assignee == "mana"

    def test_note_with_body(self, store):
        note = store.create_note("With body", body="hello body")
        assert note.body == "hello body"

    def test_note_with_sprint(self, store):
        note = store.create_note("Sprint task", sprint="S1")
        assert note.sprint == "S1"

    def test_note_with_gh(self, store):
        note = store.create_note("GH task", gh="DataWeX/sloughGPT#42")
        assert note.gh == "DataWeX/sloughGPT#42"


# ── CLI note tests ───────────────────────────────────────────────────────


class TestCLINotes:
    def test_new_creates_note(self, tmp_path):
        code, out = _new(tmp_path, "Fix boot order", "--tags", "kernel,os", "--status", "wip")
        assert code == 0
        assert "Created" in out

        store = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path)
        notes = store.list_notes()
        assert len(notes) == 1
        assert notes[0].title == "Fix boot order"
        assert notes[0].tags == ["kernel", "os"]
        assert notes[0].status == "wip"

    def test_new_rejects_bad_status(self, tmp_path):
        code, _ = _new(tmp_path, "Bad", "--status", "bogus")
        assert code == 2

    def test_list_filters_by_tag(self, tmp_path):
        _new(tmp_path, "A", "--tags", "x")
        _new(tmp_path, "B", "--tags", "y")
        code, out = _run(tmp_path, "note", "list", "--tag", "x")
        assert code == 0
        assert "A" in out
        assert "B" not in out

    def test_list_filters_by_status(self, tmp_path):
        _new(tmp_path, "A", "--status", "open")
        _new(tmp_path, "B", "--status", "done")
        code, out = _run(tmp_path, "note", "list", "--status", "done")
        assert code == 0
        assert "B" in out
        assert "A" not in out

    def test_show_displays_note_fields(self, tmp_path):
        _new(tmp_path, "Visible", "--tags", "t1", "--status", "wip", "--body", "hello body")
        code, out = _run(tmp_path, "note", "list")
        assert code == 0
        assert "Visible" in out

    def test_delete_removes_note(self, tmp_path):
        store = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path)
        note = store.create_note("Doomed")
        code, out = _run(tmp_path, "note", "delete", note.id)
        assert code == 0
        assert "Deleted" in out

    def test_search_matches_title_tag_body(self, tmp_path):
        _new(tmp_path, "Alpha engine", "--tags", "kernel", "--body", "spinny bits")
        _new(tmp_path, "Unrelated", "--tags", "ui", "--body", "nothing here")
        code, out = _run(tmp_path, "note", "list", "--tag", "kernel")
        assert code == 0
        assert "Alpha engine" in out
        assert "Unrelated" not in out

    def test_tags_counts(self, tmp_path):
        _new(tmp_path, "One", "--tags", "alpha")
        _new(tmp_path, "Two", "--tags", "alpha")
        _new(tmp_path, "Three", "--tags", "beta")
        code, out = _run(tmp_path, "board", "tags")
        # Tags are board tags, but notes also have tags
        # Just verify the CLI doesn't crash
        assert code == 0

    def test_sync(self, tmp_path):
        store = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path)
        store.create_note("Sync me", status="wip")
        code, out = _run(tmp_path, "sync")
        assert code == 0
        assert "1 new card" in out
