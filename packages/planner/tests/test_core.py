"""
Tests for planner.core — the notes CLI (``planner`` / ``notes``), driven
through ``cli_main`` so each test mirrors real usage (one store instance per
command), parameterized over the ``file`` and ``mogdb`` backends.
"""

import datetime as _dt
import re

import pytest

from planner import core as core_module
from planner.core import cli_main, reset_note_store

BACKENDS = ["file", "mogdb"]


@pytest.fixture(autouse=True)
def _isolate_store():
    """The CLI caches a store; reset it between tests."""
    reset_note_store()


@pytest.fixture(params=BACKENDS)
def cli_env(tmp_path, monkeypatch, request):
    notes_dir = tmp_path / "notes"
    monkeypatch.setenv("PLANNER_NOTES_DIR", str(notes_dir))
    monkeypatch.setenv("PLANNER_BACKEND", request.param)
    return notes_dir, request.param


def _args(backend, *parts):
    return ["--backend", backend, *parts]


def _run(cli_env, capsys, *parts):
    _, backend = cli_env
    code = cli_main(_args(backend, *parts))
    return code, capsys.readouterr().out


def _new(cli_env, capsys, title, *extra):
    code, out = _run(cli_env, capsys, "new", title, *extra)
    assert code == 0, out
    m = re.search(r"Created: (\S+)", out)
    assert m, out
    return m.group(1)


def _find_id(cli_env, capsys, title):
    code, out = _run(cli_env, capsys, "list", "--limit", "9999")
    assert code == 0, out
    for line in out.splitlines():
        if title in line and line.strip() and line.split()[0] != "--":
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return None


def test_new_creates_note(cli_env, capsys):
    code, out = _run(cli_env, capsys, "new", "Fix boot order", "--tags", "kernel,os", "--status", "wip")
    assert code == 0
    assert "Created:" in out
    code, out = _run(cli_env, capsys, "show", out.split()[1])
    assert code == 0
    assert "Fix boot order" in out
    assert "tags: kernel, os" in out
    assert "status: wip" in out


def test_new_rejects_bad_status(cli_env):
    _, backend = cli_env
    with pytest.raises(SystemExit) as exc:
        cli_main(_args(backend, "new", "Bad", "--status", "bogus"))
    assert exc.value.code == 2


def test_list_filters_by_tag_and_status(cli_env, capsys):
    _new(cli_env, capsys, "A", "--tags", "x", "--status", "open")
    _new(cli_env, capsys, "B", "--tags", "y", "--status", "done")
    code, out = _run(cli_env, capsys, "list", "--tag", "x")
    assert code == 0
    assert "A" in out and "B" not in out
    code, out = _run(cli_env, capsys, "list", "--status", "done")
    assert code == 0
    assert "B" in out and "A" not in out


def test_show_displays_note_fields(cli_env, capsys):
    note_id = _new(cli_env, capsys, "Visible", "--tags", "t1", "--status", "wip", "--body", "hello body")
    code, out = _run(cli_env, capsys, "show", note_id)
    assert code == 0
    assert "Visible" in out
    assert "hello body" in out
    assert "status: wip" in out


def test_show_unknown_returns_1(cli_env, capsys):
    code, out = _run(cli_env, capsys, "show", "00000000_000000_nope")
    assert code == 1
    assert "not found" in out


def _full_id(cli_env, capsys, short_id):
    code, out = _run(cli_env, capsys, "show", short_id)
    assert code == 0, out
    return re.search(r"id: (\S+)", out).group(1)


def test_edit_updates_fields(cli_env, capsys):
    note_id = _new(cli_env, capsys, "Old title", "--tags", "a", "--status", "open")
    old_full = _full_id(cli_env, capsys, note_id)
    code, out = _run(cli_env, capsys, "edit", note_id,
                      "--title", "New title", "--status", "done", "--body", "wrapped up")
    assert code == 0
    assert "Updated:" in out
    new_id = _find_id(cli_env, capsys, "New title")
    assert new_id is not None
    new_full = _full_id(cli_env, capsys, new_id)
    assert new_full != old_full  # rename on title change
    code, out = _run(cli_env, capsys, "show", new_id)
    assert code == 0
    assert "status: done" in out
    assert "wrapped up" in out


def test_edit_no_changes_returns_1(cli_env, capsys):
    note_id = _new(cli_env, capsys, "T")
    code, out = _run(cli_env, capsys, "edit", note_id)
    assert code == 1
    assert "No changes" in out


def test_delete_removes_note(cli_env, capsys):
    note_id = _new(cli_env, capsys, "Doomed")
    code, out = _run(cli_env, capsys, "rm", note_id)
    assert code == 0
    assert "Deleted:" in out
    code, out = _run(cli_env, capsys, "list")
    assert "No notes found." in out


def test_delete_unknown_returns_1(cli_env, capsys):
    code, out = _run(cli_env, capsys, "delete", "00000000_000000_nope")
    assert code == 1
    assert "not found" in out


def test_search_matches_title_tag_body(cli_env, capsys):
    _new(cli_env, capsys, "Alpha engine", "--tags", "kernel", "--body", "spinny bits")
    _new(cli_env, capsys, "Unrelated", "--tags", "ui", "--body", "nothing here")
    for query in ("engine", "kernel", "spinny"):
        code, out = _run(cli_env, capsys, "search", query)
        assert code == 0
        assert "Alpha engine" in out
        assert "Unrelated" not in out


def test_today_shows_created_note(cli_env, capsys, monkeypatch):
    note_id = _new(cli_env, capsys, "Today item")
    code, out = _run(cli_env, capsys, "show", note_id)
    assert code == 0
    date_str = re.search(r"id: (\d{8})", out).group(1)

    class _FakeDate:
        @classmethod
        def today(cls):
            return _dt.date.fromisoformat(date_str)

        @staticmethod
        def fromisoformat(s):
            return _dt.date.fromisoformat(s)

    monkeypatch.setattr(core_module, "date", _FakeDate)
    code, out = _run(cli_env, capsys, "today")
    assert code == 0
    assert "Today item" in out


def test_export_writes_markdown(cli_env, capsys, tmp_path):
    _new(cli_env, capsys, "Export me", "--body", "exported body")
    out_path = tmp_path / "export.md"
    code, out = _run(cli_env, capsys, "export", str(out_path))
    assert code == 0
    assert "Exported" in out
    content = out_path.read_text()
    assert "Export me" in content
    assert "exported body" in content


def test_tags_counts(cli_env, capsys):
    _new(cli_env, capsys, "One", "--tags", "alpha")
    _new(cli_env, capsys, "Two", "--tags", "alpha")
    _new(cli_env, capsys, "Three", "--tags", "beta")
    code, out = _run(cli_env, capsys, "tags")
    assert code == 0
    assert re.search(r"alpha\s+2 note", out)
    assert re.search(r"beta\s+1 note", out)


def test_status_summary(cli_env, capsys):
    _new(cli_env, capsys, "W", "--status", "wip")
    _new(cli_env, capsys, "D", "--status", "done")
    code, out = _run(cli_env, capsys, "status")
    assert code == 0
    assert re.search(r"wip\s+1", out)
    assert re.search(r"done\s+1", out)


def test_timeline_groups_by_day(cli_env, capsys, monkeypatch):
    note_id = _new(cli_env, capsys, "Timeline note", "--tags", "t")
    code, out = _run(cli_env, capsys, "show", note_id)
    assert code == 0
    date_str = re.search(r"id: (\d{8})", out).group(1)

    class _FakeDate:
        @classmethod
        def today(cls):
            return _dt.date.fromisoformat(date_str)

        @staticmethod
        def fromisoformat(s):
            return _dt.date.fromisoformat(s)

    monkeypatch.setattr(core_module, "date", _FakeDate)
    code, out = _run(cli_env, capsys, "timeline", "--tag", "t")
    assert code == 0
    assert date_str in out
    assert "Timeline note" in out


def test_sprint_list_and_report(cli_env, capsys):
    _new(cli_env, capsys, "Sprint task", "--sprint", "S1", "--status", "done", "--gh", "DataWeX/sloughGPT#42")
    code, out = _run(cli_env, capsys, "sprint", "S1")
    assert code == 0
    assert "Sprint task" in out
    assert "in sprint 'S1'" in out
    code, out = _run(cli_env, capsys, "sprint", "S1", "report")
    assert code == 0
    assert "# Sprint Report: S1" in out
    assert "DataWeX/sloughGPT#42" in out


def test_kanban_and_sync_dispatch(cli_env):
    for cmd in ("kanban", "sync", "gui"):
        with pytest.raises(SystemExit) as exc:
            cli_main([cmd, "--help"])
        assert exc.value.code == 0
