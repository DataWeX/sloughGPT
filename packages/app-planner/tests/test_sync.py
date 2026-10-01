"""
Tests for planner sync — notes -> board card sync.
"""

import pytest
from app_planner.store import PlannerStore, reset_store


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


@pytest.fixture
def store(tmp_path):
    return PlannerStore(board_dir=tmp_path / "board", notes_dir=tmp_path / "notes")


def test_sync_creates_cards_for_notes(store):
    store.create_note("Fix boot order", tags=["kernel"])
    store.create_note("Ship v2.0")
    added, updated, total = store.sync()
    assert added == 2
    assert updated == 0
    assert total == 2
    titles = {c.title for c in store.list_cards()}
    assert titles == {"Fix boot order", "Ship v2.0"}


def test_sync_is_idempotent(store):
    store.create_note("Same title")
    store.sync()
    added, updated, total = store.sync()
    assert added == 0
    assert updated == 0
    assert total == 1


def test_sync_skips_notes_with_existing_card(store):
    store.create_note("Existing")
    store.add_card("Existing", column="todo")
    added, updated, total = store.sync()
    assert added == 0
    assert updated == 0
    assert total == 1


def test_sync_derives_column_from_status(store):
    store.create_note("A", status="done")
    store.create_note("B", status="wip")
    store.create_note("C", status="review")
    store.create_note("D", status="open")
    store.sync()
    columns = {c.title: c.column for c in store.list_cards()}
    assert columns == {"A": "done", "B": "in_progress", "C": "review", "D": "todo"}


def test_sync_card_carries_tags_and_body(store):
    store.create_note("Tagged", tags=["alpha", "beta"], body="some body text")
    store.sync()
    card = store.list_cards()[0]
    assert card.tags == ["alpha", "beta"]
    assert card.description.strip() == "some body text"


def test_sync_propagates_assignee(store):
    note = store.create_note("Assigned", assignee="mana")
    store.sync()
    card = store.list_cards()[0]
    assert card.assignee == "mana"
    store.update_note(note.id, assignee="alice")
    added, updated, total = store.sync()
    assert updated == 1
    assert store.list_cards()[0].assignee == "alice"


def test_sync_moves_card_when_status_changes(store):
    note = store.create_note("Rotating task", status="wip")
    store.sync()
    assert store.list_cards()[0].column == "in_progress"
    store.update_note(note.id, status="done")
    added, updated, total = store.sync()
    assert added == 0
    assert updated == 1
    assert total == 1
    assert store.list_cards()[0].column == "done"


def test_sync_does_not_move_card_with_matching_column(store):
    store.create_note("Stable", status="done")
    store.sync()
    added, updated, total = store.sync()
    assert added == 0
    assert updated == 0
    assert total == 1


# ── Hardening: no silent column reversion (card 3ccc38b7) ────────────────


class TestStatusPolicy:
    @pytest.mark.parametrize("bad_status", ["", None, "partial", "garbage"])
    def test_sync_leaves_column_when_status_unmappable(self, store, bad_status):
        store.add_card("Keeper", column="done")
        store.create_note("Keeper", status=bad_status)
        store.sync()
        assert store.list_cards()[0].column == "done"
        assert store.last_sync_report.reverted == ()

    @pytest.mark.parametrize("active", ["doing", "wip", "in_progress"])
    def test_sync_maps_active_statuses_to_in_progress(self, store, active):
        store.create_note("Active work", status=active)
        store.sync()
        assert store.list_cards()[0].column == "in_progress"

    @pytest.mark.parametrize("empty", ["", None])
    def test_sync_skips_card_creation_for_statusless_note(self, store, empty):
        note = store.create_note("Not a task yet")
        if empty is None:
            store.update_note(note.id, status=None)
        else:
            store.update_note(note.id, status=empty)
        added, updated, total = store.sync()
        assert added == 0
        assert store.list_cards() == []

    def test_repair_reverts_unmappable_and_reports(self, store):
        card = store.add_card("Stale done", column="done")
        note = store.create_note("Stale done")
        store.update_note(note.id, status=None)
        store.sync(repair=True)
        assert store.list_cards()[0].column == "todo"
        report = store.last_sync_report
        assert report.repaired is True
        assert report.reverted == (card.id,)

    def test_repair_dry_run_writes_nothing(self, store):
        card = store.add_card("Stale done", column="done")
        note = store.create_note("Stale done")
        store.update_note(note.id, status=None)
        store.sync(repair=True, dry_run=True)
        assert store.get_card(card.id).column == "done"
        report = store.last_sync_report
        assert report.dry_run is True
        assert report.reverted == (card.id,)

    def test_normal_sync_reports_no_reverts(self, store):
        store.create_note("Plain", status="open")
        store.sync()
        report = store.last_sync_report
        assert report.added == 1
        assert report.reverted == ()
        assert report.repaired is False
        assert report.dry_run is False

    def test_sync_coerces_comma_string_tags_from_journal(self, store):
        import json

        store._notes_file.write_text(
            json.dumps({"id": "raw-1", "title": "StrTags", "status": "open", "tags": "alpha, beta"})
            + "\n"
        )
        store.sync()
        assert store.list_cards()[0].tags == ["alpha", "beta"]


def test_cli_sync_accepts_repair_and_dry_run(tmp_path):
    import contextlib
    import io

    from app_planner.cli import main as cli_main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            code = cli_main(
                [
                    "--board-dir",
                    str(tmp_path),
                    "--notes-dir",
                    str(tmp_path),
                    "sync",
                    "--repair",
                    "--dry-run",
                ]
            )
        except SystemExit as e:
            code = e.code or 1
    assert code == 0, buf.getvalue()


def test_cli_note_new_accepts_canonical_doing(tmp_path):
    import contextlib
    import io

    from app_planner.cli import main as cli_main

    with contextlib.redirect_stdout(io.StringIO()):
        try:
            code = cli_main(
                ["--board-dir", str(tmp_path), "--notes-dir", str(tmp_path),
                 "note", "new", "T", "--status", "doing"]
            )
        except SystemExit as e:
            code = e.code or 1
    assert code == 0


def test_cli_note_new_rejects_unknown_status(tmp_path):
    import contextlib
    import io

    from app_planner.cli import main as cli_main

    with contextlib.redirect_stdout(io.StringIO()):
        try:
            code = cli_main(
                ["--board-dir", str(tmp_path), "--notes-dir", str(tmp_path),
                 "note", "new", "T", "--status", "bogus"]
            )
        except SystemExit as e:
            code = e.code or 1
    assert code == 2
