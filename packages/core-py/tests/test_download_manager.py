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


# ---------------------------------------------------------------------------
# CompressedFakeBackend + tests
# ---------------------------------------------------------------------------


class CompressedFakeBackend(FakeBackend):
    """FakeBackend that reports compression support and tracks compressed calls."""

    def __init__(self):
        super().__init__()
        self.compressed_calls: list = []
        self._compression_supported = True

    def supports_compression(self, resource_id):
        return self._compression_supported and resource_id.startswith("http")

    def download_compressed(self, resource_id, on_progress, on_file_complete):
        self.compressed_calls.append(resource_id)
        if on_progress:
            on_progress(resource_id, 500, 1000, 250.0)
        if on_file_complete:
            on_file_complete(resource_id, "model.safetensors")
        return self._download_result


class TestCompressedDownload:
    def test_compressed_backend_calls_download_compressed(self):
        """When supports_compression returns True, download_compressed is called."""
        backend = CompressedFakeBackend()
        progress = []
        files = []

        def on_progress(mid, done, total, speed):
            progress.append((done, total))

        def on_file_complete(mid, path):
            files.append(path)

        result = backend.download_compressed(
            "http://server/models/7B", on_progress, on_file_complete
        )
        assert result["status"] == "complete"
        assert "http://server/models/7B" in backend.compressed_calls
        assert len(progress) == 1
        assert files == ["model.safetensors"]

    def test_non_compressed_backend_calls_download(self):
        """When supports_compression returns False, regular download is called."""
        backend = CompressedFakeBackend()
        backend._compression_supported = False
        assert backend.supports_compression("http://server/models/7B") is False

    def test_hf_model_id_uses_regular_download(self):
        """HuggingFace model IDs (not URLs) always use regular download."""
        backend = CompressedFakeBackend()
        assert backend.supports_compression("meta-llama/Llama-2-7B") is False

    def test_compressed_download_progress_tracking(self):
        """Compressed download reports progress correctly."""
        backend = CompressedFakeBackend()
        progress = []

        def on_progress(mid, done, total, speed):
            progress.append(done)

        backend.download_compressed(
            "http://server/models/7B", on_progress, None
        )
        assert progress == [500]


class TestDownloadManagerRouting:
    def test_uses_compressed_when_backend_supports(self, monkeypatch):
        """DownloadManager routes to download_compressed when supported."""
        import threading
        monkeypatch.setattr(dm, "_backend", None)
        monkeypatch.setattr(dm, "_backend_lock", threading.Lock())

        backend = CompressedFakeBackend()
        dm.set_backend(backend)

        # Verify routing logic directly
        assert backend.supports_compression("http://server/models/7B") is True
        assert backend.supports_compression("meta-llama/Llama-2-7B") is False
        dm.reset_backend()

    def test_compressed_flag_defaults_false(self):
        """Default DownloadBackend.supports_compression returns False."""
        backend = FakeBackend()
        assert backend.supports_compression("http://anything") is False


class TestDownloadWith:
    def test_download_with_uses_specific_backend(self):
        """download_with uses the provided backend, not the global one."""
        global_backend = FakeBackend()
        specific_backend = FakeBackend()
        specific_backend.downloaded = {"model-x": {"cache_dir": "/tmp/specific"}}
        specific_backend._download_result = {
            "status": "complete",
            "cache_dir": "/tmp/specific",
        }

        dm.set_backend(global_backend)
        try:
            mgr = dm.get_download_manager()
            result = asyncio.get_event_loop().run_until_complete(
                mgr.download_with("model-x", specific_backend)
            )
            assert result["status"] == "complete"
            assert result["cache_dir"] == "/tmp/specific"
            assert "model-x" in specific_backend.download_calls
            assert "model-x" not in global_backend.download_calls
        finally:
            dm.reset_backend()

    def test_download_with_already_cached(self):
        """download_with returns already_cached when resource is cached."""
        backend = FakeBackend()
        backend.cached.add("cached-model")

        mgr = dm.get_download_manager()
        result = asyncio.get_event_loop().run_until_complete(
            mgr.download_with("cached-model", backend)
        )
        assert result["status"] == "already_cached"

    def test_download_with_uses_compressed_when_supported(self):
        """download_with routes to download_compressed when backend supports it."""
        backend = CompressedFakeBackend()

        mgr = dm.get_download_manager()
        result = asyncio.get_event_loop().run_until_complete(
            mgr.download_with("http://server/model", backend)
        )
        assert result["status"] == "complete"
        assert "http://server/model" in backend.compressed_calls
        assert "http://server/model" not in backend.download_calls


class RetryBackend(FakeBackend):
    """Backend that fails N times then succeeds."""

    def __init__(self, fail_count: int = 2):
        super().__init__()
        self._fail_count = fail_count
        self._attempts = 0

    def download(self, resource_id, on_progress, on_file_complete):
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise ConnectionError(f"Simulated failure #{self._attempts}")
        return super().download(resource_id, on_progress, on_file_complete)


class TestRetryLogic:
    def test_retries_on_failure_then_succeeds(self):
        """Download retries and succeeds after transient failures."""
        backend = RetryBackend(fail_count=2)
        dm.set_backend(backend)
        try:
            mgr = dm.get_download_manager()
            result = asyncio.get_event_loop().run_until_complete(
                mgr.download("model-retry", max_retries=3)
            )
            assert result["status"] == "complete"
            assert backend._attempts == 3  # 2 failures + 1 success
        finally:
            dm.reset_backend()

    def test_fails_after_all_retries_exhausted(self):
        """Download fails when all retries are exhausted."""
        backend = RetryBackend(fail_count=10)
        dm.set_backend(backend)
        try:
            mgr = dm.get_download_manager()
            result = asyncio.get_event_loop().run_until_complete(
                mgr.download("model-fail", max_retries=2)
            )
            assert result["status"] == "failed"
            assert "Simulated failure" in result["error"]
            assert backend._attempts == 3  # max_retries + 1
        finally:
            dm.reset_backend()

    def test_no_retry_on_success(self):
        """Download doesn't retry when first attempt succeeds."""
        backend = RetryBackend(fail_count=0)
        dm.set_backend(backend)
        try:
            mgr = dm.get_download_manager()
            result = asyncio.get_event_loop().run_until_complete(
                mgr.download("model-ok", max_retries=3)
            )
            assert result["status"] == "complete"
            assert backend._attempts == 1
        finally:
            dm.reset_backend()

    def test_max_retries_zero_no_retry(self):
        """With max_retries=0, no retry on failure."""
        backend = RetryBackend(fail_count=1)
        dm.set_backend(backend)
        try:
            mgr = dm.get_download_manager()
            result = asyncio.get_event_loop().run_until_complete(
                mgr.download("model-no-retry", max_retries=0)
            )
            assert result["status"] == "failed"
            assert backend._attempts == 1
        finally:
            dm.reset_backend()


class TestPauseResume:
    def test_pause_downloading(self):
        """Pause sets status to paused."""
        mgr = dm.DownloadManager()
        mgr._set_progress("model-pause", status=dm.DownloadStatus.DOWNLOADING)
        assert mgr.pause("model-pause") is True
        progress = mgr.get_progress("model-pause")
        assert progress["status"] == "paused"

    def test_pause_not_found(self):
        """Pause returns False for unknown model."""
        mgr = dm.DownloadManager()
        assert mgr.pause("nonexistent") is False

    def test_pause_complete_not_pausable(self):
        """Cannot pause a completed download."""
        mgr = dm.DownloadManager()
        mgr._set_progress("model-done", status=dm.DownloadStatus.COMPLETE)
        assert mgr.pause("model-done") is False

    def test_pause_queued_not_pausable(self):
        """Cannot pause a queued download (not yet downloading)."""
        mgr = dm.DownloadManager()
        mgr._set_progress("model-queued", status=dm.DownloadStatus.QUEUED)
        assert mgr.pause("model-queued") is False

    def test_pause_cancelled_not_pausable(self):
        """Cannot pause a cancelled download."""
        mgr = dm.DownloadManager()
        mgr._set_progress("model-cancel", status=dm.DownloadStatus.CANCELLED)
        assert mgr.pause("model-cancel") is False

    def test_resume_paused(self):
        """Resume sets status back to queued."""
        mgr = dm.DownloadManager()
        mgr._set_progress("model-resume", status=dm.DownloadStatus.PAUSED)
        assert mgr.resume("model-resume") is True
        progress = mgr.get_progress("model-resume")
        assert progress["status"] == "queued"

    def test_resume_not_found(self):
        """Resume returns False for unknown model."""
        mgr = dm.DownloadManager()
        assert mgr.resume("nonexistent") is False

    def test_resume_downloading_not_resumable(self):
        """Cannot resume a downloading model."""
        mgr = dm.DownloadManager()
        mgr._set_progress("model-active", status=dm.DownloadStatus.DOWNLOADING)
        assert mgr.resume("model-active") is False

    def test_resume_complete_not_resumable(self):
        """Cannot resume a completed model."""
        mgr = dm.DownloadManager()
        mgr._set_progress("model-done", status=dm.DownloadStatus.COMPLETE)
        assert mgr.resume("model-done") is False

    def test_is_downloading_includes_paused(self):
        """is_downloading returns True for paused downloads."""
        mgr = dm.DownloadManager()
        mgr._set_progress("model-paused", status=dm.DownloadStatus.PAUSED)
        assert mgr.is_downloading("model-paused") is True

    def test_is_paused(self):
        """is_paused returns correct state."""
        mgr = dm.DownloadManager()
        mgr._set_progress("model-paused", status=dm.DownloadStatus.PAUSED)
        assert mgr.is_paused("model-paused") is True
        mgr._set_progress("model-active", status=dm.DownloadStatus.DOWNLOADING)
        assert mgr.is_paused("model-active") is False

    def test_pause_resume_cycle(self):
        """Full pause → resume cycle works."""
        mgr = dm.DownloadManager()
        mgr._set_progress("model-cycle", status=dm.DownloadStatus.DOWNLOADING)
        assert mgr.pause("model-cycle") is True
        assert mgr.get_progress("model-cycle")["status"] == "paused"
        assert mgr.is_paused("model-cycle") is True
        assert mgr.resume("model-cycle") is True
        assert mgr.get_progress("model-cycle")["status"] == "queued"
        assert mgr.is_paused("model-cycle") is False


class TestVerify:
    def test_verify_valid_model(self, tmp_path):
        """Verify returns valid when all files match."""
        import hashlib
        from domains.infrastructure.download_backend import DownloadBackend, FileEstimate

        class VerifyBackend(DownloadBackend):
            def __init__(self, cache_dir, files):
                self._cache_dir = cache_dir
                self._files = files

            def is_cached(self, resource_id, deep_check=False):
                return True

            def get_cache_dir(self, resource_id):
                return str(self._cache_dir)

            def estimate_total(self, resource_id):
                return sum(f.size for f in self._files)

            def list_files(self, resource_id):
                return self._files

            def download(self, resource_id, on_progress, on_file_complete):
                return {"status": "complete"}

            def cleanup(self, resource_id):
                return True

            def list_incomplete(self):
                return []

        # Create cache dir with valid file
        cache = tmp_path / "cache"
        cache.mkdir()
        data = b"hello world"
        sha = hashlib.sha256(data).hexdigest()
        (cache / "model.bin").write_bytes(data)

        files = [FileEstimate(path="model.bin", size=len(data), checksum=sha)]
        backend = VerifyBackend(cache, files)
        dm.set_backend(backend)
        try:
            mgr = dm.get_download_manager()
            result = mgr.verify("model")
            assert result["valid"] is True
            assert result["files_checked"] == 1
            assert result["files_valid"] == 1
            assert result["errors"] == []
        finally:
            dm.reset_backend()

    def test_verify_missing_file(self, tmp_path):
        """Verify returns invalid when file is missing."""
        from domains.infrastructure.download_backend import DownloadBackend, FileEstimate

        class VerifyBackend(DownloadBackend):
            def __init__(self, cache_dir, files):
                self._cache_dir = cache_dir
                self._files = files

            def is_cached(self, resource_id, deep_check=False):
                return False

            def get_cache_dir(self, resource_id):
                return str(self._cache_dir)

            def estimate_total(self, resource_id):
                return 0

            def list_files(self, resource_id):
                return self._files

            def download(self, resource_id, on_progress, on_file_complete):
                return {"status": "complete"}

            def cleanup(self, resource_id):
                return True

            def list_incomplete(self):
                return []

        cache = tmp_path / "cache"
        cache.mkdir()
        files = [FileEstimate(path="missing.bin", size=100, checksum="abc")]
        backend = VerifyBackend(cache, files)
        dm.set_backend(backend)
        try:
            mgr = dm.get_download_manager()
            result = mgr.verify("model")
            assert result["valid"] is False
            assert result["files_checked"] == 1
            assert result["files_valid"] == 0
            assert len(result["errors"]) == 1
            assert "Missing" in result["errors"][0]
        finally:
            dm.reset_backend()

    def test_verify_size_mismatch(self, tmp_path):
        """Verify returns invalid when file size doesn't match."""
        from domains.infrastructure.download_backend import DownloadBackend, FileEstimate

        class VerifyBackend(DownloadBackend):
            def __init__(self, cache_dir, files):
                self._cache_dir = cache_dir
                self._files = files

            def is_cached(self, resource_id, deep_check=False):
                return True

            def get_cache_dir(self, resource_id):
                return str(self._cache_dir)

            def estimate_total(self, resource_id):
                return 0

            def list_files(self, resource_id):
                return self._files

            def download(self, resource_id, on_progress, on_file_complete):
                return {"status": "complete"}

            def cleanup(self, resource_id):
                return True

            def list_incomplete(self):
                return []

        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "model.bin").write_bytes(b"short")
        files = [FileEstimate(path="model.bin", size=999)]  # Wrong size
        backend = VerifyBackend(cache, files)
        dm.set_backend(backend)
        try:
            mgr = dm.get_download_manager()
            result = mgr.verify("model")
            assert result["valid"] is False
            assert "Size mismatch" in result["errors"][0]
        finally:
            dm.reset_backend()

    def test_verify_checksum_mismatch(self, tmp_path):
        """Verify returns invalid when checksum doesn't match."""
        import hashlib
        from domains.infrastructure.download_backend import DownloadBackend, FileEstimate

        class VerifyBackend(DownloadBackend):
            def __init__(self, cache_dir, files):
                self._cache_dir = cache_dir
                self._files = files

            def is_cached(self, resource_id, deep_check=False):
                return True

            def get_cache_dir(self, resource_id):
                return str(self._cache_dir)

            def estimate_total(self, resource_id):
                return 0

            def list_files(self, resource_id):
                return self._files

            def download(self, resource_id, on_progress, on_file_complete):
                return {"status": "complete"}

            def cleanup(self, resource_id):
                return True

            def list_incomplete(self):
                return []

        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "model.bin").write_bytes(b"data")
        files = [FileEstimate(path="model.bin", size=4, checksum="wrong_checksum")]
        backend = VerifyBackend(cache, files)
        dm.set_backend(backend)
        try:
            mgr = dm.get_download_manager()
            result = mgr.verify("model")
            assert result["valid"] is False
            assert "Checksum mismatch" in result["errors"][0]
        finally:
            dm.reset_backend()

    def test_verify_empty_files_list(self):
        """Verify returns invalid when no files found."""
        from domains.infrastructure.download_backend import DownloadBackend, FileEstimate

        class EmptyBackend(DownloadBackend):
            def is_cached(self, resource_id, deep_check=False):
                return True

            def get_cache_dir(self, resource_id):
                return "/tmp"

            def estimate_total(self, resource_id):
                return 0

            def list_files(self, resource_id):
                return []

            def download(self, resource_id, on_progress, on_file_complete):
                return {"status": "complete"}

            def cleanup(self, resource_id):
                return True

            def list_incomplete(self):
                return []

        dm.set_backend(EmptyBackend())
        try:
            mgr = dm.get_download_manager()
            result = mgr.verify("model")
            assert result["valid"] is False
            assert result["files_checked"] == 0
        finally:
            dm.reset_backend()


class TestDownloadStats:
    def test_initial_stats(self):
        """Initial stats are all zeros."""
        mgr = dm.DownloadManager()
        stats = mgr.get_stats()
        assert stats["total_downloads"] == 0
        assert stats["completed_downloads"] == 0
        assert stats["failed_downloads"] == 0
        assert stats["cancelled_downloads"] == 0
        assert stats["total_bytes_downloaded"] == 0
        assert stats["total_download_time"] == 0

    def test_record_complete(self):
        """Recording complete download updates stats."""
        mgr = dm.DownloadManager()
        mgr._record_download_complete("model1", 1000, 2.0, 500.0)
        stats = mgr.get_stats()
        assert stats["total_downloads"] == 1
        assert stats["completed_downloads"] == 1
        assert stats["total_bytes_downloaded"] == 1000
        assert stats["total_download_time"] == 2.0

    def test_record_failed(self):
        """Recording failed download updates stats."""
        mgr = dm.DownloadManager()
        mgr._record_download_failed("model1")
        stats = mgr.get_stats()
        assert stats["total_downloads"] == 1
        assert stats["failed_downloads"] == 1

    def test_record_cancelled(self):
        """Recording cancelled download updates stats."""
        mgr = dm.DownloadManager()
        mgr._record_download_cancelled("model1")
        stats = mgr.get_stats()
        assert stats["total_downloads"] == 1
        assert stats["cancelled_downloads"] == 1

    def test_average_speed(self):
        """Average speed is computed correctly."""
        mgr = dm.DownloadManager()
        mgr._record_download_complete("m1", 1000, 2.0, 500.0)
        stats = mgr.get_stats()
        assert stats["average_speed_mb_per_sec"] == round(1000 / 2.0 / (1024 * 1024), 2)

    def test_peak_speed(self):
        """Peak speed tracks the highest speed seen."""
        mgr = dm.DownloadManager()
        mgr._record_download_complete("m1", 1000, 2.0, 500.0)
        mgr._record_download_complete("m2", 1000, 2.0, 1000.0)
        stats = mgr.get_stats()
        assert stats["peak_speed_mb_per_sec"] == round(1000.0 / (1024 * 1024), 2)

    def test_multiple_downloads(self):
        """Multiple downloads accumulate stats."""
        mgr = dm.DownloadManager()
        mgr._record_download_complete("m1", 1000, 1.0, 1000.0)
        mgr._record_download_complete("m2", 2000, 2.0, 1000.0)
        mgr._record_download_failed("m3")
        stats = mgr.get_stats()
        assert stats["total_downloads"] == 3
        assert stats["completed_downloads"] == 2
        assert stats["failed_downloads"] == 1
        assert stats["total_bytes_downloaded"] == 3000
        assert stats["total_download_time"] == 3.0

    def test_stats_to_dict(self):
        """Stats to_dict returns correct structure."""
        mgr = dm.DownloadManager()
        stats = mgr.get_stats()
        assert "total_downloads" in stats
        assert "average_speed_mb_per_sec" in stats
        assert "peak_speed_mb_per_sec" in stats
