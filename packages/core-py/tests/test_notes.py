"""Tests for the development journal notes module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "planner" / "src"))
from planner import Note, NoteStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    """Create a temporary NoteStore for testing."""
    return NoteStore(notes_dir=tmp_path)


# ---------------------------------------------------------------------------
# Note data model
# ---------------------------------------------------------------------------

class TestNote:
    def test_to_markdown_roundtrip(self):
        note = Note(
            id="20260728_103000_fix-kernel-boot",
            title="Fix kernel boot ordering",
            created_at="2026-07-28T10:30:00+00:00",
            updated_at="2026-07-28T10:30:00+00:00",
            tags=["kernel", "bugfix"],
            status="done",
            body="## Problem\nThe kernel was booting addons after init.",
        )
        md = note.to_markdown()
        assert "title: Fix kernel boot ordering" in md
        assert "tags: kernel, bugfix" in md
        assert "status: done" in md
        assert "## Problem" in md

        roundtripped = Note.from_markdown(md, note_id="20260728_103000_fix-kernel-boot")
        assert roundtripped.title == "Fix kernel boot ordering"
        assert roundtripped.tags == ["kernel", "bugfix"]
        assert roundtripped.status == "done"
        assert "## Problem" in roundtripped.body

    def test_from_markdown_no_frontmatter(self):
        md = "Just a plain markdown note.\nWith some lines."
        note = Note.from_markdown(md, note_id="plain-note")
        assert note.title == ""
        assert note.body == "Just a plain markdown note.\nWith some lines."
        assert note.tags == []

    def test_from_markdown_empty_tags(self):
        md = "---\ntitle: Empty tags\ntags: \n---\n\nBody here."
        note = Note.from_markdown(md)
        assert note.tags == []

    def test_date_str(self):
        note = Note(created_at="2026-07-28T10:30:00+00:00")
        assert note.date_str == "2026-07-28"

    def test_short_id(self):
        note = Note(id="20260728_103000_fix-kernel-boot")
        assert note.short_id == "20260728"


# ---------------------------------------------------------------------------
# NoteStore CRUD
# ---------------------------------------------------------------------------

class TestNoteStoreCRUD:
    def test_create_note(self, store):
        note = store.create("Fix kernel boot", tags=["kernel", "bugfix"])
        assert note.id.startswith("2026")
        assert note.title == "Fix kernel boot"
        assert note.tags == ["kernel", "bugfix"]
        assert note.status == "open"

    def test_create_note_persists(self, store, tmp_path):
        note = store.create("Persistent note")
        loaded = store.get(note.id)
        assert loaded is not None
        assert loaded.title == "Persistent note"

    def test_get_by_prefix(self, store):
        note = store.create("Prefix match test")
        found = store.get(note.id[:8])
        assert found is not None
        assert found.title == "Prefix match test"

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_update_title(self, store):
        note = store.create("Old title")
        updated = store.update(note.id, title="New title")
        assert updated is not None
        assert updated.title == "New title"
        # Verify persisted
        loaded = store.get(updated.id)
        assert loaded is not None
        assert loaded.title == "New title"

    def test_update_tags(self, store):
        note = store.create("Tag test", tags=["old"])
        updated = store.update(note.id, tags=["new", "fresh"])
        assert updated is not None
        assert updated.tags == ["new", "fresh"]

    def test_update_status(self, store):
        note = store.create("Status test")
        updated = store.update(note.id, status="done")
        assert updated is not None
        assert updated.status == "done"

    def test_update_body(self, store):
        note = store.create("Body test")
        updated = store.update(note.id, body="## New body\nContent here.")
        assert updated is not None
        assert "## New body" in updated.body

    def test_update_nonexistent(self, store):
        assert store.update("nonexistent", title="Nope") is None

    def test_delete_note(self, store):
        note = store.create("Delete me")
        assert store.delete(note.id) is True
        assert store.get(note.id) is None

    def test_delete_by_prefix(self, store):
        note = store.create("Delete by prefix")
        assert store.delete(note.id[:8]) is True

    def test_delete_nonexistent(self, store):
        assert store.delete("nonexistent") is False


# ---------------------------------------------------------------------------
# NoteStore listing and search
# ---------------------------------------------------------------------------

class TestNoteStoreSearch:
    def _populate(self, store):
        n1 = store.create("Fix kernel boot", tags=["kernel", "bugfix"], status="done")
        n2 = store.create("Add VFS mounts", tags=["vfs", "feature"], status="wip")
        n3 = store.create("Refactor neural addon", tags=["neural", "refactor"], status="open")
        n4 = store.update(n1.id, body="The kernel was booting addons after init.") or n1
        return n1, n2, n3, n4

    def test_list_all(self, store):
        self._populate(store)
        notes = store.list_notes()
        assert len(notes) == 3

    def test_list_by_tag(self, store):
        self._populate(store)
        notes = store.list_notes(tag="kernel")
        assert len(notes) == 1
        assert notes[0].title == "Fix kernel boot"

    def test_list_by_status(self, store):
        self._populate(store)
        notes = store.list_notes(status="wip")
        assert len(notes) == 1
        assert notes[0].title == "Add VFS mounts"

    def test_list_limit(self, store):
        for i in range(10):
            store.create(f"Note {i}")
        notes = store.list_notes(limit=5)
        assert len(notes) == 5

    def test_search_title(self, store):
        self._populate(store)
        results = store.search("kernel")
        assert len(results) >= 1
        assert any("kernel" in n.title.lower() for n in results)

    def test_search_body(self, store):
        self._populate(store)
        results = store.search("addons after init")
        assert len(results) >= 1

    def test_search_tags(self, store):
        self._populate(store)
        results = store.search("refactor")
        assert len(results) >= 1

    def test_search_no_match(self, store):
        self._populate(store)
        results = store.search("xyzzy_nonexistent")
        assert len(results) == 0

    def test_today_empty(self, store):
        assert store.today() == []

    def test_count(self, store):
        assert store.count() == 0
        store.create("One")
        store.create("Two")
        assert store.count() == 2


# ---------------------------------------------------------------------------
# NoteStore export
# ---------------------------------------------------------------------------

class TestNoteStoreExport:
    def test_export_returns_markdown(self, store):
        store.create("Export test", tags=["export"], body="## Hello")
        content = store.export_all()
        assert "Export test" in content
        assert "## Hello" in content

    def test_export_to_file(self, store, tmp_path):
        store.create("File export")
        out = tmp_path / "export.md"
        store.export_all(output_path=str(out))
        assert out.exists()
        assert "File export" in out.read_text()

    def test_export_empty(self, store):
        content = store.export_all()
        assert content == ""


# ---------------------------------------------------------------------------
# NoteStore slug generation
# ---------------------------------------------------------------------------

class TestSlug:
    def test_simple_title(self):
        assert NoteStore._title_to_slug("Fix kernel boot") == "fix-kernel-boot"

    def test_special_chars(self):
        slug = NoteStore._title_to_slug("Fix #1: kernel's boot!")
        assert slug == "fix-1-kernels-boot"

    def test_long_title_truncated(self):
        slug = NoteStore._title_to_slug("a" * 100)
        assert len(slug) <= 60

    def test_whitespace_collapsed(self):
        slug = NoteStore._title_to_slug("  too   many   spaces  ")
        assert "  " not in slug


class TestTimeline:
    def test_timeline_empty(self, store):
        assert store.timeline(days=7) == []

    def test_timeline_includes_recent_notes(self, store):
        store.create("Recent", status="done")
        groups = store.timeline(days=7)
        assert len(groups) == 1
        assert groups[0][0] == store.list_notes()[0].date_str
        assert groups[0][1][0].title == "Recent"

    def test_timeline_excludes_old_notes(self, store):
        old = store.create("Old note", status="open")
        old.created_at = "2020-01-01T00:00:00+00:00"
        store._bk.put(old)
        groups = store.timeline(days=7)
        assert all(g[0] != "2020-01-01" for g in groups)

    def test_timeline_tag_filter(self, store):
        store.create("Kernel fix", tags=["kernel"], status="done")
        store.create("UI tweak", tags=["ui"], status="wip")
        groups = store.timeline(days=7, tag="kernel")
        assert len(groups) == 1
        assert len(groups[0][1]) == 1
        assert groups[0][1][0].title == "Kernel fix"

    def test_timeline_status_filter(self, store):
        store.create("Active task", status="wip")
        store.create("Done task", status="done")
        groups = store.timeline(days=7, status="wip")
        assert len(groups) == 1
        assert len(groups[0][1]) == 1
        assert groups[0][1][0].title == "Active task"

    def test_timeline_grouped_by_day(self, store):
        store.create("First", status="open")
        store.create("Second", status="open")
        groups = store.timeline(days=7)
        assert len(groups) == 1
        assert len(groups[0][1]) == 2
