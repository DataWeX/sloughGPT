"""Tests for mogdb.sync — diff-based file sync for MogDB collections."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mogdb.sync import (
    SyncResult,
    _content_hash,
    _cast_csv_value,
    _load_json,
    _load_jsonl,
    _load_csv,
    sync_from_files,
    sync_from_json,
    sync_from_jsonl,
    sync_from_csv,
    preview_from_files,
)


# ── SyncResult ──────────────────────────────────────────────────────

class TestSyncResult:
    def test_defaults(self):
        r = SyncResult()
        assert r.inserted == 0
        assert r.updated == 0
        assert r.deleted == 0
        assert r.unchanged == 0
        assert r.errors == []

    def test_to_dict(self):
        r = SyncResult()
        r.inserted = 3
        r.updated = 1
        d = r.to_dict()
        assert d["inserted"] == 3
        assert d["updated"] == 1
        assert d["deleted"] == 0
        assert d["unchanged"] == 0
        assert d["errors"] == []

    def test_repr(self):
        r = SyncResult()
        r.inserted = 2
        r.errors.append("err")
        assert "inserted=2" in repr(r)
        assert "errors=1" in repr(r)


# ── _content_hash ───────────────────────────────────────────────────

class TestContentHash:
    def test_same_content_same_hash(self):
        assert _content_hash({"a": 1, "b": 2}) == _content_hash({"a": 1, "b": 2})

    def test_different_content_different_hash(self):
        assert _content_hash({"a": 1}) != _content_hash({"a": 2})

    def test_ignores_metadata_fields(self):
        assert _content_hash({"a": 1, "_id": "x"}) == _content_hash({"a": 1, "_id": "y"})

    def test_order_independent(self):
        assert _content_hash({"a": 1, "b": 2}) == _content_hash({"b": 2, "a": 1})

    def test_handles_non_string_values(self):
        h = _content_hash({"count": 42, "rate": 3.14, "flag": True})
        assert len(h) == 16


# ── _cast_csv_value ─────────────────────────────────────────────────

class TestCastCsvValue:
    def test_empty_string(self):
        assert _cast_csv_value("") is None

    def test_true(self):
        assert _cast_csv_value("true") is True
        assert _cast_csv_value("TRUE") is True
        assert _cast_csv_value("yes") is True

    def test_false(self):
        assert _cast_csv_value("false") is False
        assert _cast_csv_value("FALSE") is False
        assert _cast_csv_value("no") is False

    def test_null(self):
        assert _cast_csv_value("null") is None
        assert _cast_csv_value("none") is None
        assert _cast_csv_value("NONE") is None

    def test_integer(self):
        assert _cast_csv_value("42") == 42
        assert isinstance(_cast_csv_value("42"), int)

    def test_float(self):
        assert _cast_csv_value("3.14") == 3.14
        assert isinstance(_cast_csv_value("3.14"), float)

    def test_string(self):
        assert _cast_csv_value("hello") == "hello"

    def test_negative_int(self):
        assert _cast_csv_value("-5") == -5

    def test_negative_float(self):
        assert _cast_csv_value("-3.14") == -3.14


# ── _load_json ──────────────────────────────────────────────────────

class TestLoadJson:
    def test_loads_array(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"a": 1}, {"a": 2}]))
        assert _load_json(str(path)) == [{"a": 1}, {"a": 2}]

    def test_loads_dict_with_data_key(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"data": [{"a": 1}]}))
        assert _load_json(str(path)) == [{"a": 1}]

    def test_loads_dict_with_items_key(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"items": [{"a": 1}]}))
        assert _load_json(str(path)) == [{"a": 1}]

    def test_loads_dict_with_root_key(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"root": [{"a": 1}]}))
        assert _load_json(str(path)) == [{"a": 1}]

    def test_loads_single_dict(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"a": 1}))
        assert _load_json(str(path)) == [{"a": 1}]

    def test_empty_array(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text("[]")
        assert _load_json(str(path)) == []


# ── _load_jsonl ─────────────────────────────────────────────────────

class TestLoadJsonl:
    def test_loads_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\n{"a": 2}\n')
        assert _load_jsonl(str(path)) == [{"a": 1}, {"a": 2}]

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n')
        assert _load_jsonl(str(path)) == [{"a": 1}, {"a": 2}]

    def test_empty_file(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text("")
        assert _load_jsonl(str(path)) == []


# ── _load_csv ───────────────────────────────────────────────────────

class TestLoadCsv:
    def test_loads_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("name,value\nhello,42\nworld,7")
        docs = _load_csv(str(path))
        assert len(docs) == 2
        assert docs[0]["name"] == "hello"
        assert docs[0]["value"] == 42

    def test_csv_with_types(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("flag,count\ntrue,10\nfalse,0")
        docs = _load_csv(str(path))
        assert docs[0]["flag"] is True
        assert docs[0]["count"] == 10
        assert docs[1]["flag"] is False


# ── sync_from_files ─────────────────────────────────────────────────

class TestSyncFromFiles:
    def _mock_collection(self, existing=None):
        coll = MagicMock()
        coll.find.return_value = existing or []
        return coll

    def test_inserts_new_documents(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([
            {"email": "a@test.com", "name": "Alice"},
            {"email": "b@test.com", "name": "Bob"},
        ]))
        coll = self._mock_collection()
        result = sync_from_files(coll, str(path), "email")
        assert result.inserted == 2
        assert result.updated == 0
        assert result.deleted == 0
        assert coll.insert_one.call_count == 2

    def test_updates_changed_documents(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"email": "a@test.com", "name": "Alice Updated"}]))
        existing = [{"_id": "1", "email": "a@test.com", "name": "Alice"}]
        coll = self._mock_collection(existing)
        result = sync_from_files(coll, str(path), "email")
        assert result.inserted == 0
        assert result.updated == 1
        assert result.unchanged == 0
        coll.update_one.assert_called_once()

    def test_unchanged_documents(self, tmp_path):
        doc = {"email": "a@test.com", "name": "Alice"}
        path = tmp_path / "data.json"
        path.write_text(json.dumps([doc.copy()]))
        existing = [{"_id": "1", **doc}]
        coll = self._mock_collection(existing)
        result = sync_from_files(coll, str(path), "email")
        assert result.unchanged == 1
        assert result.inserted == 0
        assert result.updated == 0

    def test_deletes_missing_documents(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"email": "a@test.com"}]))
        existing = [
            {"_id": "1", "email": "a@test.com"},
            {"_id": "2", "email": "b@test.com"},
        ]
        coll = self._mock_collection(existing)
        result = sync_from_files(coll, str(path), "email", delete_missing=True)
        assert result.deleted == 1
        coll.delete_one.assert_called_once_with({"_id": "2"})

    def test_no_delete_without_flag(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"email": "a@test.com"}]))
        existing = [{"_id": "1", "email": "b@test.com"}]
        coll = self._mock_collection(existing)
        result = sync_from_files(coll, str(path), "email", delete_missing=False)
        assert result.deleted == 0

    def test_file_not_found(self):
        coll = self._mock_collection()
        with pytest.raises(FileNotFoundError):
            sync_from_files(coll, "/nonexistent.json", "id")

    def test_unknown_extension(self, tmp_path):
        path = tmp_path / "data.xyz"
        path.write_text("data")
        coll = self._mock_collection()
        with pytest.raises(ValueError, match="Cannot detect format"):
            sync_from_files(coll, str(path), "id")

    def test_explicit_format(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text(json.dumps([{"id": 1}]))
        coll = self._mock_collection()
        result = sync_from_files(coll, str(path), "id", file_format="json")
        assert result.inserted == 1

    def test_jsonl_sync(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"id": 1, "val": "a"}\n{"id": 2, "val": "b"}\n')
        coll = self._mock_collection()
        result = sync_from_files(coll, str(path), "id", file_format="jsonl")
        assert result.inserted == 2

    def test_csv_sync(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("id,name\n1,Alice\n2,Bob\n")
        coll = self._mock_collection()
        result = sync_from_files(coll, str(path), "id", file_format="csv")
        assert result.inserted == 2

    def test_mixed_operations(self, tmp_path):
        """Insert new, update changed, keep unchanged."""
        path = tmp_path / "data.json"
        path.write_text(json.dumps([
            {"id": "a", "val": 1},    # unchanged
            {"id": "b", "val": 99},   # updated
            {"id": "c", "val": 3},    # new
        ]))
        existing = [
            {"_id": "1", "id": "a", "val": 1},
            {"_id": "2", "id": "b", "val": 2},
        ]
        coll = self._mock_collection(existing)
        result = sync_from_files(coll, str(path), "id")
        assert result.inserted == 1
        assert result.updated == 1
        assert result.unchanged == 1


# ── preview_from_files ───────────────────────────────────────────────

class TestPreviewFromFiles:
    def _mock_collection(self, existing=None):
        coll = MagicMock()
        coll.find.return_value = existing or []
        return coll

    def test_counts_without_mutating(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"id": "a", "val": 1}, {"id": "b", "val": 2}]))
        coll = self._mock_collection()
        result = preview_from_files(coll, str(path), "id")
        assert result.inserted == 2
        assert result.updated == 0
        assert result.deleted == 0
        coll.insert_one.assert_not_called()
        coll.update_one.assert_not_called()
        coll.delete_one.assert_not_called()

    def test_matches_sync_plan(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([
            {"id": "a", "val": 1},    # unchanged
            {"id": "b", "val": 99},   # updated
            {"id": "c", "val": 3},    # new
        ]))
        existing = [
            {"_id": "1", "id": "a", "val": 1},
            {"_id": "2", "id": "b", "val": 2},
        ]
        coll = self._mock_collection(existing)
        result = preview_from_files(coll, str(path), "id")
        assert result.inserted == 1
        assert result.updated == 1
        assert result.unchanged == 1

    def test_delete_missing_counts(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"id": "a", "val": 1}]))
        existing = [
            {"_id": "1", "id": "a", "val": 1},
            {"_id": "2", "id": "b", "val": 2},
        ]
        coll = self._mock_collection(existing)
        result = preview_from_files(coll, str(path), "id", delete_missing=True)
        assert result.deleted == 1
        coll.delete_one.assert_not_called()

    def test_missing_file(self):
        coll = self._mock_collection()
        with pytest.raises(FileNotFoundError):
            preview_from_files(coll, "/nonexistent.json", "id")


# ── Convenience aliases ─────────────────────────────────────────────

class TestConvenienceAliases:
    def test_sync_from_json(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"id": 1}]))
        coll = MagicMock()
        coll.find.return_value = []
        result = sync_from_json(coll, str(path), "id")
        assert result.inserted == 1

    def test_sync_from_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"id": 1}\n')
        coll = MagicMock()
        coll.find.return_value = []
        result = sync_from_jsonl(coll, str(path), "id")
        assert result.inserted == 1

    def test_sync_from_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("id,name\n1,Alice\n")
        coll = MagicMock()
        coll.find.return_value = []
        result = sync_from_csv(coll, str(path), "id")
        assert result.inserted == 1
