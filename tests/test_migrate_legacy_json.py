"""Tests for scripts/migrate_legacy_json.py — one-time legacy JSON import tool."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "migrate_legacy_json.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("migrate_legacy_json", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_script()


@pytest.fixture
def data_file(tmp_path):
    p = tmp_path / "entries.json"
    p.write_text(json.dumps([
        {"id": "a", "name": "Alice"},
        {"id": "b", "name": "Bob"},
    ]))
    return p


def test_main_inserts_and_syncs(tmp_path, data_file, tool):
    db = tmp_path / "db"
    sync = tmp_path / "sync"
    rc = tool.main([
        str(data_file), "--db", str(db), "--collection", "entries",
        "--key", "id", "--sync-dir", str(sync),
    ])
    assert rc == 0
    assert (sync / "entries.json").exists()
    docs = json.loads((sync / "entries.json").read_text())
    assert len(docs) == 2


def test_main_is_idempotent(tmp_path, data_file, tool):
    db = tmp_path / "db"
    tool.main([str(data_file), "--db", str(db), "--collection", "entries", "--key", "id"])
    rc = tool.main([str(data_file), "--db", str(db), "--collection", "entries", "--key", "id"])
    assert rc == 0


def test_main_dry_run_writes_nothing(tmp_path, data_file, tool):
    db = tmp_path / "db"
    rc = tool.main([
        str(data_file), "--db", str(db), "--collection", "entries",
        "--key", "id", "--dry-run",
    ])
    assert rc == 0
    assert not list(db.glob("*.jsonl"))
    assert not list(db.glob("*.mogdb"))


def test_main_missing_file(tmp_path, tool):
    rc = tool.main([str(tmp_path / "nope.json"), "--key", "id"])
    assert rc == 2


def test_default_collection(tool):
    assert tool.default_collection("data/knowledge/entries.json") == "entries"
    assert tool.default_collection("response_logs/chat-2026.jsonl") == "chat_2026"


def test_update_detected(tmp_path, tool):
    data_file = tmp_path / "entries.json"
    data_file.write_text(json.dumps([{"id": "a", "name": "Alice"}]))
    db = tmp_path / "db"
    tool.main([str(data_file), "--db", str(db), "--collection", "entries", "--key", "id"])
    data_file.write_text(json.dumps([{"id": "a", "name": "Alice Updated"}]))
    rc = tool.main([str(data_file), "--db", str(db), "--collection", "entries", "--key", "id"])
    assert rc == 0
