"""Tests for domains.memory.task_memory — archive helpers and constants."""

import json
import time
import pytest
from pathlib import Path

from domains.memory.memory_config import MemoryConfig
from domains.memory.task_memory import (
    TASK_REMEMBER,
    TASK_STORE,
    TASK_CONSOLIDATE,
    _ARCHIVE_FILENAME,
    _archive_path,
    _append_archive,
    _read_archive,
    list_archive,
    archive_stats,
    prune_archive,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_task_remember_value(self):
        assert TASK_REMEMBER == "memory.remember"

    def test_task_store_value(self):
        assert TASK_STORE == "memory.store"

    def test_task_consolidate_value(self):
        assert TASK_CONSOLIDATE == "memory.consolidate"

    def test_archive_filename(self):
        assert _ARCHIVE_FILENAME == "facts.jsonl"


# ---------------------------------------------------------------------------
# _archive_path
# ---------------------------------------------------------------------------

class TestArchivePath:
    def test_returns_path_with_filename(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path))
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        p = _archive_path()
        assert p.name == "facts.jsonl"
        assert p.parent == tmp_path

    def test_default_store_path_ends_with_filename(self):
        p = _archive_path()
        assert p.name == "facts.jsonl"


# ---------------------------------------------------------------------------
# _append_archive / _read_archive (round-trip)
# ---------------------------------------------------------------------------

class TestAppendReadRoundTrip:
    def test_single_record(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        record = {"ts": 1.0, "task_type": "memory.remember", "stored": True}
        _append_archive(record)
        records = _read_archive()
        assert len(records) == 1
        assert records[0]["task_type"] == "memory.remember"
        assert records[0]["ts"] == 1.0

    def test_multiple_records_appended_in_order(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        for i in range(5):
            _append_archive({"ts": float(i), "idx": i})
        records = _read_archive()
        assert len(records) == 5
        assert [r["idx"] for r in records] == [0, 1, 2, 3, 4]

    def test_creates_parent_directory(self, tmp_path, monkeypatch):
        nested = tmp_path / "a" / "b" / "c"
        cfg = MemoryConfig(store_path=str(nested), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        _append_archive({"ts": 1.0})
        assert nested.is_dir()
        records = _read_archive()
        assert len(records) == 1

    def test_unicode_content_preserved(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        record = {"content": "日本語テスト 🎉 émojis"}
        _append_archive(record)
        records = _read_archive()
        assert records[0]["content"] == "日本語テスト 🎉 émojis"


# ---------------------------------------------------------------------------
# _read_archive — edge cases
# ---------------------------------------------------------------------------

class TestReadArchive:
    def test_empty_when_file_missing(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        assert _read_archive() == []

    def test_skips_blank_lines(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        path = _archive_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"a":1}\n\n\n{"b":2}\n', encoding="utf-8")
        records = _read_archive()
        assert len(records) == 2

    def test_skips_corrupt_json_lines(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        path = _archive_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"valid": true}\nnot json\n{"also_valid": 2}\n',
            encoding="utf-8",
        )
        records = _read_archive()
        assert len(records) == 2
        assert records[0]["valid"] is True
        assert records[1]["also_valid"] == 2

    def test_empty_file_returns_empty_list(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        path = _archive_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        assert _read_archive() == []


# ---------------------------------------------------------------------------
# list_archive
# ---------------------------------------------------------------------------

class TestListArchive:
    def test_returns_newest_first(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        for i in range(5):
            _append_archive({"ts": float(i), "idx": i})
        result = list_archive(limit=10)
        assert [r["idx"] for r in result] == [4, 3, 2, 1, 0]

    def test_respects_limit(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        for i in range(10):
            _append_archive({"ts": float(i), "idx": i})
        result = list_archive(limit=3)
        assert len(result) == 3
        assert result[0]["idx"] == 9

    def test_limit_one_returns_single_newest(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        for i in range(5):
            _append_archive({"ts": float(i), "idx": i})
        result = list_archive(limit=1)
        assert len(result) == 1
        assert result[0]["idx"] == 4

    def test_empty_archive_returns_empty_list(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        assert list_archive(limit=10) == []

    def test_limit_clamped_to_at_least_one(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        _append_archive({"ts": 1.0})
        result = list_archive(limit=0)
        assert len(result) == 1
        result = list_archive(limit=-5)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# archive_stats
# ---------------------------------------------------------------------------

class TestArchiveStats:
    def test_empty_archive(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        stats = archive_stats()
        assert stats["records"] == 0
        assert stats["bytes"] == 0
        assert stats["task_types"] == {}
        assert stats["oldest_ts"] is None
        assert stats["newest_ts"] is None

    def test_counts_by_task_type(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        _append_archive({"task_type": "memory.remember", "ts": 1.0})
        _append_archive({"task_type": "memory.remember", "ts": 2.0})
        _append_archive({"task_type": "memory.store", "ts": 3.0})
        stats = archive_stats()
        assert stats["records"] == 3
        assert stats["task_types"]["memory.remember"] == 2
        assert stats["task_types"]["memory.store"] == 1

    def test_timestamps(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        _append_archive({"ts": 100.0})
        _append_archive({"ts": 50.0})
        _append_archive({"ts": 200.0})
        stats = archive_stats()
        assert stats["oldest_ts"] == 50.0
        assert stats["newest_ts"] == 200.0

    def test_bytes_nonzero_when_records_exist(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        _append_archive({"ts": 1.0, "data": "x" * 100})
        stats = archive_stats()
        assert stats["bytes"] > 100

    def test_unknown_task_type(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        _append_archive({"ts": 1.0})
        stats = archive_stats()
        assert stats["task_types"]["unknown"] == 1

    def test_path_in_stats(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        stats = archive_stats()
        assert stats["path"].endswith("facts.jsonl")


# ---------------------------------------------------------------------------
# prune_archive
# ---------------------------------------------------------------------------

class TestPruneArchive:
    def test_no_file_returns_zero(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        assert prune_archive(retain_days=30) == 0

    def test_removes_old_records(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        now = time.time()
        old = now - 100 * 86400
        recent = now - 1 * 86400
        _append_archive({"ts": old, "label": "old"})
        _append_archive({"ts": recent, "label": "recent"})
        removed = prune_archive(retain_days=7)
        assert removed == 1
        records = _read_archive()
        assert len(records) == 1
        assert records[0]["label"] == "recent"

    def test_keeps_all_when_within_retention(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        now = time.time()
        _append_archive({"ts": now - 1 * 86400, "label": "a"})
        _append_archive({"ts": now - 2 * 86400, "label": "b"})
        removed = prune_archive(retain_days=7)
        assert removed == 0
        assert len(_read_archive()) == 2

    def test_retain_days_zero_removes_everything(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        now = time.time()
        _append_archive({"ts": now, "label": "a"})
        _append_archive({"ts": now, "label": "b"})
        removed = prune_archive(retain_days=0)
        assert removed == 2
        assert _read_archive() == []

    def test_records_without_ts_are_pruned(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        _append_archive({"label": "no_ts"})
        _append_archive({"ts": time.time(), "label": "with_ts"})
        removed = prune_archive(retain_days=30)
        assert removed == 1
        records = _read_archive()
        assert len(records) == 1
        assert records[0]["label"] == "with_ts"

    def test_defaults_to_config_retention(self, tmp_path, monkeypatch):
        cfg = MemoryConfig(store_path=str(tmp_path), enabled=True, archive_retention_days=0.001)
        monkeypatch.setattr(
            "domains.memory.task_memory.MemoryConfig",
            type("MC", (), {"get": staticmethod(lambda: cfg)}),
        )
        old = time.time() - 10 * 86400
        _append_archive({"ts": old})
        removed = prune_archive()
        assert removed == 1
