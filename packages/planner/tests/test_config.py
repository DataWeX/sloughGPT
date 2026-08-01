"""
Tests for planner.config — repo-aware path resolution, backend inference,
and the shared status <-> column maps.
"""

import json
import os
from pathlib import Path

import pytest

from planner import config


def _write_board(root: Path) -> Path:
    board = root / ".kanban" / "board.json"
    board.parent.mkdir(parents=True, exist_ok=True)
    board.write_text(json.dumps({"cards": []}))
    return root


# ---------------------------------------------------------------------------
# find_project_root
# ---------------------------------------------------------------------------


def test_find_project_root_finds_nearest_ancestor(tmp_path):
    root = _write_board(tmp_path)
    nested = root / "a" / "b" / "c"
    nested.mkdir(parents=True)
    assert config.find_project_root(nested) == root


def test_find_project_root_returns_none_without_board(tmp_path):
    assert config.find_project_root(tmp_path) is None


def test_find_project_root_ignores_board_without_file(tmp_path):
    (tmp_path / ".kanban").mkdir()
    assert config.find_project_root(tmp_path) is None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_default_notes_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANNER_NOTES_DIR", str(tmp_path / "env-notes"))
    assert config.default_notes_dir() == tmp_path / "env-notes"


def test_default_board_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANNER_BOARD_DIR", str(tmp_path / "env-board"))
    assert config.default_board_dir() == tmp_path / "env-board"


def test_default_notes_dir_project_root(tmp_path, monkeypatch):
    root = _write_board(tmp_path)
    monkeypatch.setattr(config, "find_project_root", lambda start=None: root)
    assert config.default_notes_dir() == root / ".dev-notes"


def test_default_board_dir_project_root(tmp_path, monkeypatch):
    root = _write_board(tmp_path)
    monkeypatch.setattr(config, "find_project_root", lambda start=None: root)
    assert config.default_board_dir() == root / ".kanban"


def test_default_dirs_fall_back_to_user_config(monkeypatch):
    monkeypatch.setattr(config, "find_project_root", lambda start=None: None)
    assert config.default_notes_dir() == config.NOTES_FALLBACK
    assert config.default_board_dir() == config.BOARD_FALLBACK


# ---------------------------------------------------------------------------
# Backend inference
# ---------------------------------------------------------------------------


def test_default_backend_env_override(monkeypatch):
    monkeypatch.setenv("PLANNER_BACKEND", "mogdb")
    assert config.default_backend() == "mogdb"


def test_default_backend_ignores_invalid_env(monkeypatch):
    monkeypatch.setenv("PLANNER_BACKEND", "bogus")
    assert config.default_backend() in config.BACKENDS


def test_default_backend_infers_mogdb_from_journal(tmp_path):
    notes = tmp_path / "notes"
    (notes / "store").mkdir(parents=True)
    (notes / "store" / "notes.journal.jsonl").write_text("")
    assert config.default_backend(notes_dir=notes) == "mogdb"


def test_default_backend_infers_file_without_journal(tmp_path):
    assert config.default_backend(notes_dir=tmp_path / "notes") == "file"


def test_default_backend_infers_from_default_notes_dir(tmp_path, monkeypatch):
    notes = tmp_path / "notes"
    (notes / "store").mkdir(parents=True)
    (notes / "store" / "notes.journal.jsonl").write_text("")
    monkeypatch.setattr(config, "default_notes_dir", lambda: notes)
    monkeypatch.delenv("PLANNER_BACKEND", raising=False)
    assert config.default_backend() == "mogdb"


# ---------------------------------------------------------------------------
# Status <-> column maps
# ---------------------------------------------------------------------------


def test_every_status_has_a_column():
    for status in config.STATUSES:
        assert config.STATUS_TO_COLUMN[status] in config.COLUMN_TO_STATUS


def test_status_to_column_mapping():
    expected = {
        "done": "done",
        "wip": "in_progress",
        "review": "review",
        "todo": "todo",
        "open": "todo",
        "blocked": "todo",
        "": "todo",
    }
    for status, col in expected.items():
        assert config.STATUS_TO_COLUMN[status] == col


def test_column_to_status_mapping():
    expected = {"todo": "open", "in_progress": "wip", "review": "review", "done": "done"}
    for col, status in expected.items():
        assert config.COLUMN_TO_STATUS[col] == status
