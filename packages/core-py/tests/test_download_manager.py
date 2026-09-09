"""Tests for download_manager — generic orchestration via DownloadBackend."""

import asyncio
import sys
import time
import types
from pathlib import Path
from typing import List

import pytest

import domains.infrastructure.download_manager as dm
from domains.infrastructure.download_backend import DownloadBackend, FileEstimate


# ---------------------------------------------------------------------------
# Fake backend for testing
# ---------------------------------------------------------------------------


class FakeBackend(DownloadBackend):
    """In-memory backend for unit tests."""

    def __init__(self):
        self.cached: set = set()
        self.downloaded: dict = {}
        self.incomplete: List[str] = []
        self.cleaned: List[str] = []
        self.cancelled: List[str] = []
        self._download_result = {"status": "complete", "cache_dir": "/tmp/cache"}
        self.download_calls: list = []

    def is_cached(self, resource_id, deep_check=False):
        return resource_id in self.cached

    def get_cache_dir(self, resource_id):
        return self.downloaded.get(resource_id, {}).get("cache_dir", "/tmp/cache")

    def estimate_total(self, resource_id):
        return self.downloaded.get(resource_id, {}).get("total", 1000)

    def list_files(self, resource_id):
        return []

    def download(self, resource_id, on_progress, on_file_complete):
        self.download_calls.append(resource_id)
        if on_progress:
            on_progress(resource_id, 500, 1000, 250.0)
        if on_file_complete:
            on_file_complete(resource_id, "model.safetensors")
        return self._download_result

    def cleanup(self, resource_id):
        self.cleaned.append(resource_id)
        return True

    def list_incomplete(self):
        return self.incomplete

    def on_cancel(self, resource_id):
        self.cancelled.append(resource_id)


# ---------------------------------------------------------------------------
# TestDownloadStatus
# ---------------------------------------------------------------------------


class TestDownloadStatus:
    def test_enum_values(self):
        assert dm.DownloadStatus.QUEUED.value == "queued"
        assert dm.DownloadStatus.DOWNLOADING.value == "downloading"
        assert dm.DownloadStatus.COMPLETE.value == "complete"
        assert dm.DownloadStatus.FAILED.value == "failed"
        assert dm.DownloadStatus.CANCELLED.value == "cancelled"


# ---------------------------------------------------------------------------
# TestDownloadProgress
# ---------------------------------------------------------------------------


class TestDownloadProgress:
    def test_to_dict_basic(self):
        p = dm.DownloadProgress(model_id="gpt2", status=dm.DownloadStatus.DOWNLOADING)
        d = p.to_dict()
        assert d["model_id"] == "gpt2"
        assert d["status"] == "downloading"
        assert d["speed_mb_per_sec"] == 0.0
        assert d["files_total"] == 0

    def test_to_dict_rounds_speed_and_pct(self):
        p = dm.DownloadProgress(
            model_id="m",
            status=dm.DownloadStatus.DOWNLOADING,
            bytes_downloaded=2_621_440,
            total_bytes=10_000_000,
            speed_bytes_per_sec=5_242_880,
            percentage=26.2333,
        )
        d = p.to_dict()
        assert d["speed_mb_per_sec"] == 5.0
        assert d["percentage"] == 26.2


# ---------------------------------------------------------------------------
# TestModuleLevelFunctions
# ---------------------------------------------------------------------------


class TestModuleLevelFunctions:
    def test_is_download_complete_delegates(self, monkeypatch):
        fake = FakeBackend()
        fake.cached.add("gpt2")
        monkeypatch.setattr(dm, "_backend", fake)
        assert dm.is_download_complete("gpt2") is True

    def test_cleanup_incomplete_delegates(self, monkeypatch):
        fake = FakeBackend()
        monkeypatch.setattr(dm, "_backend", fake)
        assert dm.cleanup_incomplete("gpt2") is True
        assert "gpt2" in fake.cleaned

    def test_list_incomplete_models_delegates(self, monkeypatch):
        fake = FakeBackend()
        fake.incomplete = ["org/model_a", "org/model_b"]
        monkeypatch.setattr(dm, "_backend", fake)
        assert dm.list_incomplete_models() == ["org/model_a", "org/model_b"]


# ---------------------------------------------------------------------------
# TestBackendProtocol
# ---------------------------------------------------------------------------


class TestBackendProtocol:
    def test_fake_backend_is_cached(self):
        b = FakeBackend()
        assert b.is_cached("x") is False
        b.cached.add("x")
        assert b.is_cached("x") is True

    def test_fake_backend_download(self):
        b = FakeBackend()
        progress_calls = []
        file_calls = []

        def on_progress(rid, done, total, speed):
            progress_calls.append((rid, done, total, speed))

        def on_file(rid, path):
            file_calls.append((rid, path))

        result = b.download("model", on_progress, on_file)
        assert result["status"] == "complete"
        assert len(progress_calls) == 1
        assert len(file_calls) == 1

    def test_fake_backend_cleanup(self):
        b = FakeBackend()
        b.cleanup("model")
        assert "model" in b.cleaned

    def test_fake_backend_list_incomplete(self):
        b = FakeBackend()
        b.incomplete = ["a", "b"]
        assert b.list_incomplete() == ["a", "b"]

    def test_fake_backend_on_cancel(self):
        b = FakeBackend()
        b.on_cancel("model")
        assert "model" in b.cancelled


# ---------------------------------------------------------------------------
# TestDownloadManager
# ---------------------------------------------------------------------------


class TestDownloadManager:
    def test_get_progress_unknown(self):
        mgr = dm.DownloadManager()
        assert mgr.get_progress("gpt2") is None

    def test_set_progress_creates_entry(self):
        mgr = dm.DownloadManager()
        mgr._set_progress("gpt2", status=dm.DownloadStatus.QUEUED, total_bytes=100)
        assert mgr.get_progress("gpt2")["status"] == "queued"
        assert mgr.get_progress("gpt2")["total_bytes"] == 100

    def test_list_downloads(self):
        mgr = dm.DownloadManager()
        mgr._set_progress("a")
        mgr._set_progress("b")
        assert sorted(mgr.list_downloads()) == ["a", "b"]

    def test_is_downloading(self):
        mgr = dm.DownloadManager()
        mgr._set_progress("q", status=dm.DownloadStatus.QUEUED)
        mgr._set_progress("d", status=dm.DownloadStatus.DOWNLOADING)
        mgr._set_progress("c", status=dm.DownloadStatus.COMPLETE)
        assert mgr.is_downloading("q") is True
        assert mgr.is_downloading("d") is True
        assert mgr.is_downloading("c") is False
        assert mgr.is_downloading("nope") is False

    def test_cancel_queued(self, monkeypatch):
        fake = FakeBackend()
        monkeypatch.setattr(dm, "_backend", fake)
        mgr = dm.DownloadManager()
        mgr._set_progress("gpt2", status=dm.DownloadStatus.QUEUED)
        assert mgr.cancel("gpt2") is True
        assert mgr.get_progress("gpt2")["status"] == "cancelled"
        assert "gpt2" in fake.cancelled

    def test_cancel_complete_returns_false(self, monkeypatch):
        fake = FakeBackend()
        monkeypatch.setattr(dm, "_backend", fake)
        mgr = dm.DownloadManager()
        mgr._set_progress("gpt2", status=dm.DownloadStatus.COMPLETE)
        assert mgr.cancel("gpt2") is False

    def test_on_progress_notifies(self):
        mgr = dm.DownloadManager()
        seen = []
        mgr.on_progress("gpt2", lambda d: seen.append(d))
        mgr._set_progress("gpt2", status=dm.DownloadStatus.DOWNLOADING, percentage=50.0)
        mgr._notify_callbacks("gpt2")
        assert len(seen) == 1
        assert seen[0]["percentage"] == 50.0

    def test_callback_errors_swallowed(self):
        mgr = dm.DownloadManager()

        def bad(d):
            raise RuntimeError("cb boom")

        mgr.on_progress("gpt2", bad)
        mgr._set_progress("gpt2")  # should not raise
        mgr._notify_callbacks("gpt2")  # should not raise

    def test_download_already_cached(self, monkeypatch):
        fake = FakeBackend()
        fake.cached.add("gpt2")
        monkeypatch.setattr(dm, "_backend", fake)
        mgr = dm.DownloadManager()
        result = asyncio.run(mgr.download("gpt2"))
        assert result == {"status": "already_cached", "model_id": "gpt2"}

    def test_download_already_downloading(self, monkeypatch):
        fake = FakeBackend()
        monkeypatch.setattr(dm, "_backend", fake)
        mgr = dm.DownloadManager()
        mgr._set_progress("gpt2", status=dm.DownloadStatus.DOWNLOADING)
        result = asyncio.run(mgr.download("gpt2"))
        assert result["status"] == "already_downloading"

    def test_download_completes(self, monkeypatch):
        fake = FakeBackend()
        monkeypatch.setattr(dm, "_backend", fake)
        mgr = dm.DownloadManager()
        result = asyncio.run(mgr.download("gpt2"))
        assert result["status"] == "complete"
        assert mgr.is_downloading("gpt2") is False

    def test_download_failure_reported(self, monkeypatch):
        fake = FakeBackend()

        def fail_download(rid, on_progress, on_file_complete):
            raise ValueError("network down")

        fake.download = fail_download
        monkeypatch.setattr(dm, "_backend", fake)
        mgr = dm.DownloadManager()
        result = asyncio.run(mgr.download("gpt2"))
        assert result["status"] == "failed"
        assert "network down" in result["error"]

    def test_download_cancelled(self, monkeypatch):
        fake = FakeBackend()

        def cancel_download(rid, on_progress, on_file_complete):
            raise asyncio.CancelledError()

        fake.download = cancel_download
        monkeypatch.setattr(dm, "_backend", fake)
        mgr = dm.DownloadManager()
        result = asyncio.run(mgr.download("gpt2"))
        assert result["status"] == "cancelled"

    def test_is_cached(self, monkeypatch):
        fake = FakeBackend()
        fake.cached.add("gpt2")
        monkeypatch.setattr(dm, "_backend", fake)
        mgr = dm.DownloadManager()
        assert mgr.is_cached("gpt2") is True

    def test_download_calls_prepare(self, monkeypatch):
        fake = FakeBackend()
        monkeypatch.setattr(dm, "_backend", fake)
        mgr = dm.DownloadManager()
        asyncio.run(mgr.download("gpt2"))
        assert "gpt2" in fake.download_calls

    def test_cleanup_stale(self):
        mgr = dm.DownloadManager()
        old = time.time() - 10_000
        mgr._set_progress("old_done", status=dm.DownloadStatus.COMPLETE, completed_at=old)
        mgr._set_progress("recent_done", status=dm.DownloadStatus.COMPLETE,
                          completed_at=time.time())
        mgr._set_progress("running", status=dm.DownloadStatus.DOWNLOADING,
                          started_at=old)
        mgr.cleanup_stale(max_age=3600)
        assert mgr.get_progress("old_done") is None
        assert mgr.get_progress("recent_done") is not None
        assert mgr.get_progress("running") is not None

    def test_cleanup_stale_drops_task(self):
        mgr = dm.DownloadManager()
        old = time.time() - 10_000
        mgr._set_progress("x", status=dm.DownloadStatus.FAILED, started_at=old)
        mgr._tasks["x"] = object()
        mgr.cleanup_stale(max_age=1)
        assert "x" not in mgr._tasks


# ---------------------------------------------------------------------------
# TestSingleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_download_manager_singleton(self, monkeypatch):
        monkeypatch.setattr(dm, "_download_manager", None)
        a = dm.get_download_manager()
        b = dm.get_download_manager()
        assert a is b


# ---------------------------------------------------------------------------
# TestBackendManagement
# ---------------------------------------------------------------------------


class TestBackendManagement:
    def test_set_and_get_backend(self, monkeypatch):
        monkeypatch.setattr(dm, "_backend", None)
        fake = FakeBackend()
        dm.set_backend(fake)
        assert dm.get_backend() is fake
        dm.reset_backend()

    def test_null_backend_fallback(self, monkeypatch):
        import threading
        monkeypatch.setattr(dm, "_backend", None)
        monkeypatch.setattr(dm, "_backend_lock", threading.Lock())
        old_val = sys.modules.get("domains.infrastructure.hf_hub")
        sys.modules["domains.infrastructure.hf_hub"] = None
        try:
            backend = dm.get_backend()
            assert backend.is_cached("x") is False
            assert backend.get_cache_dir("x") == ""
            assert backend.list_incomplete() == []
        finally:
            if old_val is not None:
                sys.modules["domains.infrastructure.hf_hub"] = old_val
            else:
                sys.modules.pop("domains.infrastructure.hf_hub", None)
            dm._backend = None
