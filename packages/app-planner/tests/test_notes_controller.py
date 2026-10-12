"""
Tests for app_planner.notes_controller — the note contract shared by both stores.

Two dialects are exercised for every behaviour the contract claims to unify
(validation, read model, CRUD, linkage), because "it works" on one store and
silently drops a field on the other is the exact failure this module exists
to prevent.
"""

from __future__ import annotations

import threading

import pytest
from app_planner import config
from app_planner.notes_controller import (
    NotesController,
    NoteStoreError,
    NoteValidationError,
)
from app_planner.store import PlannerStore, reset_store


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


@pytest.fixture
def store(tmp_path):
    return PlannerStore(board_dir=tmp_path / "board", notes_dir=tmp_path / "notes")


@pytest.fixture
def ctrl(store):
    return NotesController(store)


@pytest.fixture
def legacy(tmp_path):
    from app_planner.core import NoteStore

    return NoteStore(notes_dir=tmp_path / "legacy_notes", backend="file")


@pytest.fixture
def legacy_ctrl(legacy):
    return NotesController(legacy)


# ── Construction / dialect detection ──────────────────────────────────────


def test_detects_planner_dialect(store):
    assert NotesController(store).dialect == "planner"


def test_detects_legacy_dialect(legacy):
    assert NotesController(legacy).dialect == "legacy"


def test_rejects_unknown_store():
    with pytest.raises(TypeError, match="unsupported note store"):
        NotesController(object())


def test_fields_differ_per_dialect(store, legacy):
    assert "author" not in NotesController(store).fields
    assert "author" in NotesController(legacy).fields


# ── Validation (write side) ───────────────────────────────────────────────


def test_create_requires_title(ctrl, store):
    with pytest.raises(NoteValidationError, match="title is required"):
        ctrl.create("   ")
    assert store.list_notes(limit=9999) == []


def test_create_rejects_unknown_status(ctrl, store):
    with pytest.raises(NoteValidationError, match="invalid status"):
        ctrl.create("Task", status="sideways")
    assert store.list_notes(limit=9999) == []


def test_create_accepts_every_config_status(store, ctrl):
    for status in config.STATUSES:
        note = ctrl.create(f"Task {status}", status=status)
        assert note.status == status


def test_create_normalises_tags(ctrl):
    note = ctrl.create("Tagged", tags=["  alpha ", "", "alpha", "beta"])
    assert note.tags == ["alpha", "beta"]


def test_create_rejects_non_string_tag(ctrl):
    with pytest.raises(NoteValidationError, match="tags must be strings"):
        ctrl.create("Tagged", tags=["ok", 7])


def test_create_rejects_non_string_body(ctrl):
    with pytest.raises(NoteValidationError, match="body must be a string"):
        ctrl.create("Task", body=42)


def test_create_rejects_author_on_planner_store(ctrl):
    """PlannerStore has no ``author`` — a silent drop would be data loss."""
    with pytest.raises(NoteValidationError, match="author.*not storable by the planner store"):
        ctrl.create("Task", author="mana")


def test_update_rejects_author_on_planner_store(ctrl, store):
    note = ctrl.create("Task")
    with pytest.raises(NoteValidationError, match="author.*not storable"):
        ctrl.update(note.id, author="mana")
    assert store.get_note(note.id).title == "Task"


def test_update_rejects_unknown_field(ctrl):
    note = ctrl.create("Task")
    with pytest.raises(NoteValidationError, match="unknown note field"):
        ctrl.update(note.id, colour="red")


def test_update_rejects_empty_field_set(ctrl):
    note = ctrl.create("Task")
    with pytest.raises(NoteValidationError, match="no fields to update"):
        ctrl.update(note.id)


def test_update_rejects_bad_status(ctrl, store):
    note = ctrl.create("Task")
    with pytest.raises(NoteValidationError, match="invalid status"):
        ctrl.update(note.id, status="nope")
    assert store.get_note(note.id).status == "open"


def test_update_none_fields_are_ignored(ctrl):
    """Partial update semantics: None means "leave alone", not "clear"."""
    note = ctrl.create("Task", body="keep me", status="wip")
    updated = ctrl.update(note.id, body=None, status="done")
    assert updated.body == "keep me"
    assert updated.status == "done"


def test_note_validation_error_is_a_value_error(ctrl):
    """gui.py maps ``ValueError`` to HTTP 400 — the contract depends on it."""
    with pytest.raises(ValueError):
        ctrl.create("")


# ── CRUD ──────────────────────────────────────────────────────────────────


def test_create_get_update_roundtrip(ctrl, store):
    note = ctrl.create("Boot the VM", body="press any key", tags=["vm"])
    assert ctrl.get(note.id).title == "Boot the VM"

    updated = ctrl.update(note.id, title="Boot the x86 VM", status="done")
    assert updated.title == "Boot the x86 VM"
    assert updated.status == "done"
    assert store.get_note(note.id).title == "Boot the x86 VM"
    assert ctrl.list_notes(query="x86")[0].title == "Boot the x86 VM"


def test_delete_claims_success_only_when_the_note_is_gone(ctrl):
    """The honesty contract for delete — no projection may be lied to.

    ``PlannerStore._write_notes`` currently re-appends a deleted id during
    the union-on-write merge, so the delete path may raise instead of
    succeeding. Both outcomes are correct: success implies the note is gone,
    and a raise implies it really is still there (the reason it raised). The
    test asserts the invariant, not one branch of it, so it stays valid both
    before and after that data-layer defect is fixed.
    """
    note = ctrl.create("Delete me")
    try:
        ok = ctrl.delete(note.id)
    except NoteStoreError as exc:
        assert ctrl.get(note.id) is not None, "controller refused a delete that had worked"
        assert "still readable" in str(exc)
        return
    assert ok is True
    assert ctrl.get(note.id) is None, "delete() returned True but the note survived"


def test_get_missing_returns_none(ctrl):
    assert ctrl.get("does-not-exist") is None


def test_get_requires_id(ctrl):
    with pytest.raises(NoteValidationError, match="note id is required"):
        ctrl.get("   ")


def test_delete_missing_returns_false(ctrl):
    assert ctrl.delete("does-not-exist") is False


def test_delete_requires_id(ctrl):
    with pytest.raises(NoteValidationError, match="note id is required"):
        ctrl.delete("")


def test_update_missing_returns_none(ctrl):
    assert ctrl.update("does-not-exist", title="x") is None


# ── Read model ────────────────────────────────────────────────────────────


@pytest.fixture
def corpus(ctrl):
    ctrl.create("Fix boot order", tags=["kernel", "vm"], status="wip", body="uefi path")
    ctrl.create("Ship v2.0 release", tags=["release"], status="done")
    ctrl.create("Triage flaky tests", tags=["tests"], status="open", body="boot flake")
    ctrl.create("Write kernel docs", tags=["kernel"], status="review")
    return ctrl


def test_list_filters_by_tag(corpus):
    titles = {n.title for n in corpus.list_notes(tag="kernel")}
    assert titles == {"Fix boot order", "Write kernel docs"}


def test_list_filters_by_status(corpus):
    titles = {n.title for n in corpus.list_notes(status="done")}
    assert titles == {"Ship v2.0 release"}


def test_list_combines_tag_and_status(corpus):
    assert [n.title for n in corpus.list_notes(tag="kernel", status="wip")] == ["Fix boot order"]


def test_list_query_matches_title_body_and_tags(corpus):
    assert {n.title for n in corpus.list_notes(query="boot")} == {
        "Fix boot order",
        "Triage flaky tests",
    }
    assert {n.title for n in corpus.list_notes(query="uefi")} == {"Fix boot order"}
    assert {n.title for n in corpus.list_notes(query="RELEASE")} == {"Ship v2.0 release"}


def test_list_applies_limit_after_query(corpus):
    """Truncating first would hide the match behind an unrelated page."""
    assert [n.title for n in corpus.list_notes(limit=1)] == ["Fix boot order"]
    assert [n.title for n in corpus.list_notes(query="kernel", limit=1)] == ["Fix boot order"]
    assert len(corpus.list_notes(query="kernel", limit=2)) == 2


def test_list_rejects_bad_limit(corpus):
    with pytest.raises(NoteValidationError, match="limit"):
        corpus.list_notes(limit=-1)
    with pytest.raises(NoteValidationError, match="limit"):
        corpus.list_notes(limit="10")


def test_list_rejects_bad_status(corpus):
    with pytest.raises(NoteValidationError, match="invalid status"):
        corpus.list_notes(status="nope")


def test_list_rejects_author_filter_on_planner_store(corpus):
    with pytest.raises(NoteValidationError, match="does not support author"):
        corpus.list_notes(author="mana")


def test_search_requires_query(corpus):
    with pytest.raises(NoteValidationError, match="query is required"):
        corpus.search("   ")


def test_search_delegates_to_list(corpus):
    assert [n.title for n in corpus.search("boot")] == [
        n.title for n in corpus.list_notes(query="boot")
    ]


def test_empty_corpus_lists_empty(ctrl):
    assert ctrl.list_notes() == []


def test_tags_counts_sorted_by_frequency_then_name(ctrl):
    ctrl.create("A", tags=["kernel", "vm"])
    ctrl.create("B", tags=["kernel"])
    ctrl.create("C", tags=["misc", "kernel"])
    assert list(ctrl.tags().items()) == [("kernel", 3), ("misc", 1), ("vm", 1)]


def test_stats_counts_total_and_by_status(corpus):
    stats = corpus.stats()
    assert stats["total"] == 4
    assert stats["by_status"] == {"wip": 1, "done": 1, "open": 1, "review": 1}


def test_stats_counts_today(corpus, store):
    """store.Note has no ``date_str``; stats must derive it from created_at."""
    assert corpus.stats()["today"] == 4


# ── Read model: to_dict ───────────────────────────────────────────────────


def test_to_dict_omits_author_on_planner_note(ctrl):
    note = ctrl.create("Task")
    payload = NotesController.to_dict(note)
    assert "author" not in payload
    assert payload["title"] == "Task"
    assert payload["tags"] == []


def test_to_dict_includes_author_on_legacy_note(legacy_ctrl):
    note = legacy_ctrl.create("Task", author="mana")
    payload = NotesController.to_dict(note)
    assert payload["author"] == "mana"


# ── Linkage with the board ────────────────────────────────────────────────


def test_column_for_status_uses_config_map():
    assert NotesController.column_for("wip") == "in_progress"
    assert NotesController.column_for("done") == "done"
    assert NotesController.column_for("open") == "todo"
    assert NotesController.column_for("") == "todo"
    assert NotesController.column_for("sideways") == "todo"


def test_column_for_reads_note_status(ctrl):
    note = ctrl.create("Task", status="review")
    assert NotesController.column_for(note) == "review"


def test_status_for_column_round_trips():
    assert NotesController.status_for("done") == "done"
    assert NotesController.status_for("in_progress") == "doing"
    assert NotesController.status_for("unknown-column") == "open"


# ── Sync ──────────────────────────────────────────────────────────────────


def test_sync_delegates_to_planner_store(ctrl, store):
    ctrl.create("Card me")
    assert ctrl.sync() == (1, 0, 1)
    assert [c.title for c in store.list_cards()] == ["Card me"]


def test_sync_without_kanban_store_on_legacy_raises(legacy_ctrl):
    legacy_ctrl.create("Card me")
    with pytest.raises(NoteValidationError, match="requires the kanban store"):
        legacy_ctrl.sync()


def test_sync_legacy_uses_handed_in_kanban_store(legacy_ctrl, tmp_path):
    from app_planner.kanban import KanbanStore

    board = KanbanStore(board_dir=tmp_path / "legacy_board")
    board.init_board()
    legacy_ctrl.create("Card me", status="wip")
    added, updated, total = legacy_ctrl.sync(board)
    assert (added, total) == (1, 1)
    assert board.load_board().cards[0].column == "in_progress"


# ── Locking ───────────────────────────────────────────────────────────────


def test_mutations_take_the_supplied_lock(store):
    calls: list[str] = []

    class RecordingLock:
        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *exc):
            calls.append("exit")
            return False

    ctrl = NotesController(store, lock=RecordingLock())  # type: ignore[arg-type]
    note = ctrl.create("Locked task")
    ctrl.update(note.id, title="Renamed")
    try:
        ctrl.delete(note.id)  # may raise while the store resurrects deletes
    except NoteStoreError:
        pass
    # create + update + delete each took the lock exactly once
    assert calls == ["enter", "exit"] * 3


def test_reads_do_not_take_the_lock(store):
    calls: list[str] = []

    class RecordingLock:
        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *exc):
            return False

    ctrl = NotesController(store, lock=RecordingLock())  # type: ignore[arg-type]
    ctrl.create("Task")
    calls.clear()
    ctrl.list_notes()
    ctrl.tags()
    ctrl.stats()
    assert calls == []


def test_lock_defaults_to_none(store):
    assert NotesController(store).lock is None


def test_works_without_a_lock(store):
    """CLI path — no lock supplied, mutations still run."""
    ctrl = NotesController(store)
    note = ctrl.create("Task")
    assert ctrl.get(note.id) is not None
    try:
        assert ctrl.delete(note.id) is True
    except NoteStoreError:
        # store resurrected it (known data-layer defect) — controller said so
        assert ctrl.get(note.id) is not None


def test_accepts_a_real_threading_lock(store):
    ctrl = NotesController(store, lock=threading.Lock())
    assert ctrl.create("Task") is not None


# ── Legacy dialect specifics ──────────────────────────────────────────────


def test_legacy_create_stores_author(legacy_ctrl):
    note = legacy_ctrl.create("Task", author="mana")
    assert note.author == "mana"
    assert legacy_ctrl.get(note.id).author == "mana"


def test_legacy_list_filters_by_author_and_sprint(legacy_ctrl):
    legacy_ctrl.create("A", author="mana", sprint="s1")
    legacy_ctrl.create("B", author="alice", sprint="s2")
    assert [n.title for n in legacy_ctrl.list_notes(author="mana")] == ["A"]
    assert [n.title for n in legacy_ctrl.list_notes(sprint="s2")] == ["B"]


def test_legacy_search_matches_author(legacy_ctrl):
    legacy_ctrl.create("Task", author="mana")
    assert [n.title for n in legacy_ctrl.search("mana")] == ["Task"]


def test_legacy_crud_roundtrip(legacy_ctrl):
    note = legacy_ctrl.create("Task", tags=["a"])
    # core.NoteStore rewrites the id when the title changes (slug follows
    # title), so the updated note — not the original — carries the live id.
    updated = legacy_ctrl.update(note.id, title="Renamed", status="done")
    assert (updated.title, updated.status) == ("Renamed", "done")
    assert updated.id != note.id
    assert legacy_ctrl.delete(updated.id) is True
    assert legacy_ctrl.get(updated.id) is None
    assert legacy_ctrl.get(note.id) is None


def test_legacy_stats_and_tags(legacy_ctrl):
    legacy_ctrl.create("A", tags=["x"], status="done")
    legacy_ctrl.create("B", tags=["x", "y"])
    assert legacy_ctrl.tags() == {"x": 2, "y": 1}
    assert legacy_ctrl.stats()["by_status"] == {"done": 1, "open": 1}
