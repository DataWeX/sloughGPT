"""Tests for domains.memory.task_memory — archive CRUD and pruning."""

import json
import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch

from domains.memory.task_memory import (
    _archive_path, _append_archive, _read_archive,
    list_archive, archive_stats, prune_archive,
)
from domains.memory.memory_config import MemoryConfig


@pytest.fixture(autouse=True)
def _use_tmp_archive(tmp_path):
    """Redirect archive to a temp directory for every test."""
    archive_dir = tmp_path / "memory"
    archive_dir.mkdir()
    archive_file = archive_dir / "facts.jsonl"
    with patch.object(MemoryConfig, "get", return_value=MemoryConfig(store_path=str(archive_dir))):
        yield archive_file


class TestArchivePath:
    def test_returns_path(self):
        p = _archive_path()
        assert p.name == "facts.jsonl"


class TestAppendReadArchive:
    def test_append_and_read(self, _use_tmp_archive):
        _append_archive({"content": "hello", "ts": 1000})
        records = _read_archive()
        assert len(records) == 1
        assert records[0]["content"] == "hello"

    def test_multiple_appends(self, _use_tmp_archive):
        for i in range(5):
            _append_archive({"i": i, "ts": i})
        records = _read_archive()
        assert len(records) == 5

    def test_corrupt_line_skipped(self, _use_tmp_archive):
        _append_archive({"content": "good"})
        _use_tmp_archive.write_text('{"content":"good"}\nbad json\n{"content":"also good"}\n')
        records = _read_archive()
        assert len(records) == 2

    def test_empty_file(self, _use_tmp_archive):
        records = _read_archive()
        assert records == []


class TestListArchive:
    def test_newest_first(self, _use_tmp_archive):
        for i in range(5):
            _append_archive({"i": i, "ts": i})
        result = list_archive(limit=3)
        assert len(result) == 3
        assert result[0]["i"] == 4

    def test_limit_clamped(self, _use_tmp_archive):
        _append_archive({"i": 0, "ts": 0})
        result = list_archive(limit=0)
        assert len(result) == 1


class TestArchiveStats:
    def test_empty_archive(self, _use_tmp_archive):
        stats = archive_stats()
        assert stats["records"] == 0
        assert stats["bytes"] == 0

    def test_with_records(self, _use_tmp_archive):
        _append_archive({"task_type": "remember", "ts": 1000})
        _append_archive({"task_type": "store", "ts": 2000})
        stats = archive_stats()
        assert stats["records"] == 2
        assert stats["task_types"]["remember"] == 1
        assert stats["task_types"]["store"] == 1
        assert stats["oldest_ts"] == 1000
        assert stats["newest_ts"] == 2000

    def test_no_timestamps(self, _use_tmp_archive):
        _append_archive({"task_type": "remember"})
        stats = archive_stats()
        assert stats["oldest_ts"] is None


class TestPruneArchive:
    def test_no_file(self, tmp_path):
        with patch.object(MemoryConfig, "get", return_value=MemoryConfig(store_path=str(tmp_path / "empty"))):
            assert prune_archive() == 0

    def test_prune_old(self, _use_tmp_archive):
        now = time.time()
        _append_archive({"ts": now - 86400 * 100})  # very old
        _append_archive({"ts": now - 10})  # recent
        removed = prune_archive(retain_days=1)
        assert removed == 1
        records = _read_archive()
        assert len(records) == 1

    def test_prune_zero_days_removes_all(self, _use_tmp_archive):
        _append_archive({"ts": 1000})
        removed = prune_archive(retain_days=0)
        assert removed == 1

    def test_nothing_to_prune(self, _use_tmp_archive):
        now = time.time()
        _append_archive({"ts": now})
        removed = prune_archive(retain_days=365)
        assert removed == 0

    def test_records_without_ts_pruned(self, _use_tmp_archive):
        _append_archive({"no_ts": True})
        removed = prune_archive(retain_days=0)
        assert removed == 1
