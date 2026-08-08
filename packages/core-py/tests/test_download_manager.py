"""Tests for download_manager — HF download orchestration, cache health helpers."""

import asyncio
import sys
import time
import types
from pathlib import Path

import pytest

import domains.infrastructure.download_manager as dm


# ── Fake hf_hub module for worker tests ───────────────────────────────────


def _make_fake_downcraft():
    fake = types.ModuleType("domains.infrastructure.hf_hub")
    calls = {"progress": [], "files": []}

    def download_hf_model(model_id, on_progress=None, on_file_complete=None):
        calls["progress"].append((model_id, 500, 1000, 250.0))
        calls["files"].append((model_id, "model.safetensors"))
        if on_progress:
            on_progress(model_id, 500, 1000, 250.0)
        if on_file_complete:
            on_file_complete(model_id, "model.safetensors")

    fake.download_hf_model = download_hf_model
    return fake, calls


class TestDownloadStatus:
    def test_enum_values(self):
        assert dm.DownloadStatus.QUEUED.value == "queued"
        assert dm.DownloadStatus.DOWNLOADING.value == "downloading"
        assert dm.DownloadStatus.COMPLETE.value == "complete"
        assert dm.DownloadStatus.FAILED.value == "failed"
        assert dm.DownloadStatus.CANCELLED.value == "cancelled"


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


class TestCacheHelpers:
    def test_cache_dir_dashes_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dm, "get_cache_dir",
                            lambda mid: str(tmp_path / f"models--{mid.replace('/', '--')}"))
        assert dm._cache_dir("org/model").name == "models--org--model"

    def test_has_weight_files_true_over_1kb(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "snapshots" / "x").mkdir(parents=True)
        (cache / "snapshots" / "x" / "model.safetensors").write_bytes(b"0" * 2000)
        assert dm._has_weight_files(cache) is True

    def test_has_weight_files_false_small_or_missing(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "snapshots" / "x").mkdir(parents=True)
        (cache / "snapshots" / "x" / "tiny.bin").write_bytes(b"0" * 100)
        assert dm._has_weight_files(cache) is False
        assert dm._has_weight_files(tmp_path / "nope") is False

    def test_has_weight_files_skips_stat_errors(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "snapshots" / "x").mkdir(parents=True)
        (cache / "snapshots" / "x" / "model.safetensors").symlink_to(tmp_path / "missing")
        assert dm._has_weight_files(cache) is False

    def test_has_incomplete_downloads_incomplete_marker(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "blob").mkdir(parents=True)
        (cache / "blob" / "f.incomplete").touch()
        assert dm._has_incomplete_downloads(cache) is True

    def test_has_incomplete_downloads_lock(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "blob").mkdir(parents=True)
        (cache / "blob" / "f.lock").touch()
        assert dm._has_incomplete_downloads(cache) is True

    def test_has_incomplete_downloads_clean(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        assert dm._has_incomplete_downloads(cache) is False

    def test_get_snapshot_ref(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "refs").mkdir(parents=True)
        (cache / "refs" / "main").write_text("abc123\n")
        assert dm._get_snapshot_ref(cache) == "abc123"

    def test_get_snapshot_ref_missing(self, tmp_path):
        assert dm._get_snapshot_ref(tmp_path / "nope") is None

    def test_get_snapshot_ref_read_error(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "refs" / "main").mkdir(parents=True)
        assert dm._get_snapshot_ref(cache) is None

    def test_has_complete_snapshot(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "refs").mkdir(parents=True)
        (cache / "refs" / "main").write_text("abc123")
        (cache / "snapshots" / "abc123").mkdir(parents=True)
        (cache / "snapshots" / "abc123" / "model.safetensors").write_bytes(b"0" * 2000)
        assert dm._has_complete_snapshot(cache) is True

    def test_has_complete_snapshot_no_weights(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "refs").mkdir(parents=True)
        (cache / "refs" / "main").write_text("abc123")
        (cache / "snapshots" / "abc123").mkdir(parents=True)
        assert dm._has_complete_snapshot(cache) is False

    def test_has_complete_snapshot_no_ref(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "snapshots" / "abc123").mkdir(parents=True)
        assert dm._has_complete_snapshot(cache) is False

    def test_has_complete_snapshot_missing_snapshot_dir(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "refs").mkdir(parents=True)
        (cache / "refs" / "main").write_text("abc123")
        assert dm._has_complete_snapshot(cache) is False


class TestDelegatedFunctions:
    def test_is_download_complete_delegates(self, monkeypatch):
        monkeypatch.setattr(dm, "hf_is_download_complete", lambda mid, deep_check=False: True)
        assert dm.is_download_complete("gpt2") is True
        assert dm.is_download_complete("gpt2", deep_check=True) is True

    def test_cleanup_incomplete_removes_dir(self, tmp_path, monkeypatch):
        cache = tmp_path / "models--gpt2"
        cache.mkdir()
        (cache / "blob").mkdir()
        monkeypatch.setattr(dm, "_cache_dir", lambda mid: cache)
        assert dm.cleanup_incomplete("gpt2") is True
        assert not cache.exists()

    def test_cleanup_incomplete_missing_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dm, "_cache_dir", lambda mid: tmp_path / "nope")
        assert dm.cleanup_incomplete("gpt2") is False


class TestListIncompleteModels:
    def test_empty_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert dm.list_incomplete_models() == []

    def test_lists_incomplete_and_partial(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        base = tmp_path / ".cache" / "huggingface" / "hub"
        # incomplete marker
        inc = base / "models--org--bad"
        (inc / "blob").mkdir(parents=True)
        (inc / "blob" / "f.incomplete").touch()
        # weights but no complete snapshot
        partial = base / "models--org--partial"
        (partial / "snapshots" / "abc").mkdir(parents=True)
        (partial / "snapshots" / "abc" / "model.safetensors").write_bytes(b"0" * 2000)
        # complete snapshot -> not listed
        done = base / "models--org--done"
        (done / "refs").mkdir(parents=True)
        (done / "refs" / "main").write_text("commit1")
        (done / "snapshots" / "commit1").mkdir(parents=True)
        (done / "snapshots" / "commit1" / "model.safetensors").write_bytes(b"0" * 2000)
        # non-model dir skipped
        (base / "other").mkdir()
        result = dm.list_incomplete_models()
        assert "org/bad" in result
        assert "org/partial" in result
        assert "org/done" not in result


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

    def test_cancel_queued(self):
        mgr = dm.DownloadManager()
        mgr._set_progress("gpt2", status=dm.DownloadStatus.QUEUED)
        assert mgr.cancel("gpt2") is True
        assert mgr.get_progress("gpt2")["status"] == "cancelled"

    def test_cancel_complete_returns_false(self):
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
        mgr = dm.DownloadManager()
        monkeypatch.setattr(dm, "hf_is_download_complete", lambda mid, deep_check=False: True)
        result = asyncio.run(mgr.download("gpt2"))
        assert result == {"status": "already_cached", "model_id": "gpt2"}

    def test_download_already_downloading(self, monkeypatch):
        mgr = dm.DownloadManager()
        monkeypatch.setattr(dm, "hf_is_download_complete", lambda mid, deep_check=False: False)
        mgr._set_progress("gpt2", status=dm.DownloadStatus.DOWNLOADING)
        result = asyncio.run(mgr.download("gpt2"))
        assert result["status"] == "already_downloading"

    def test_download_completes(self, monkeypatch):
        mgr = dm.DownloadManager()
        monkeypatch.setattr(dm, "hf_is_download_complete", lambda mid, deep_check=False: False)
        monkeypatch.setattr(dm, "list_model_files", lambda mid: [])

        async def fake_worker(self, model_id, hint):
            self._set_progress(model_id, status=dm.DownloadStatus.COMPLETE)
            return {"status": "complete", "model_id": model_id}

        monkeypatch.setattr(dm.DownloadManager, "_download_worker", fake_worker)
        result = asyncio.run(mgr.download("gpt2"))
        assert result["status"] == "complete"
        assert mgr.is_downloading("gpt2") is False

    def test_download_failure_reported(self, monkeypatch):
        mgr = dm.DownloadManager()
        monkeypatch.setattr(dm, "hf_is_download_complete", lambda mid, deep_check=False: False)

        async def bad_worker(self, model_id, hint):
            raise ValueError("network down")

        monkeypatch.setattr(dm.DownloadManager, "_download_worker", bad_worker)
        result = asyncio.run(mgr.download("gpt2"))
        assert result["status"] == "failed"
        assert "network down" in result["error"]

    def test_download_cancelled(self, monkeypatch):
        mgr = dm.DownloadManager()
        monkeypatch.setattr(dm, "hf_is_download_complete", lambda mid, deep_check=False: False)

        async def cancel_worker(self, model_id, hint):
            raise asyncio.CancelledError()

        monkeypatch.setattr(dm.DownloadManager, "_download_worker", cancel_worker)
        result = asyncio.run(mgr.download("gpt2"))
        assert result["status"] == "cancelled"

    def test_is_cached(self, monkeypatch):
        mgr = dm.DownloadManager()
        monkeypatch.setattr(dm, "hf_is_download_complete", lambda mid, deep_check=False: True)
        assert mgr.is_cached("gpt2") is True

    def test_cancel_downloading_cancels_task(self):
        mgr = dm.DownloadManager()
        mgr._set_progress("gpt2", status=dm.DownloadStatus.DOWNLOADING)

        async def scenario():
            t = asyncio.create_task(asyncio.sleep(10))
            mgr._tasks["gpt2"] = t
            ok = mgr.cancel("gpt2")
            try:
                await t
            except asyncio.CancelledError:
                pass
            return ok

        assert asyncio.run(scenario()) is True
        assert mgr.get_progress("gpt2")["status"] == "cancelled"

    def test_download_cleans_incomplete_markers(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        (cache / "blob").mkdir(parents=True)
        (cache / "blob" / "f.incomplete").write_bytes(b"x")
        (cache / "blob" / "f.lock").write_bytes(b"x")
        monkeypatch.setattr(dm, "hf_is_download_complete", lambda mid, deep_check=False: False)
        monkeypatch.setattr(dm, "list_model_files", lambda mid: [])
        monkeypatch.setattr(dm, "_cache_dir", lambda mid: cache)
        monkeypatch.setattr(dm, "sg_state", None)

        async def fake_worker(self, model_id, hint):
            return {"status": "complete", "model_id": model_id}

        monkeypatch.setattr(dm.DownloadManager, "_download_worker", fake_worker)
        mgr = dm.DownloadManager()
        result = asyncio.run(mgr.download("gpt2"))
        assert result["status"] == "complete"
        assert not (cache / "blob" / "f.incomplete").exists()
        assert not (cache / "blob" / "f.lock").exists()

    def test_download_ignores_unlink_errors(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        (cache / "blob").mkdir(parents=True)
        (cache / "blob" / "f.incomplete").write_bytes(b"x")
        monkeypatch.setattr(dm, "hf_is_download_complete", lambda mid, deep_check=False: False)
        monkeypatch.setattr(dm, "list_model_files", lambda mid: [])
        monkeypatch.setattr(dm, "_cache_dir", lambda mid: cache)
        monkeypatch.setattr(dm, "sg_state", None)

        def broken_unlink(self):
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "unlink", broken_unlink)

        async def fake_worker(self, model_id, hint):
            return {"status": "complete", "model_id": model_id}

        monkeypatch.setattr(dm.DownloadManager, "_download_worker", fake_worker)
        mgr = dm.DownloadManager()
        result = asyncio.run(mgr.download("gpt2"))
        assert result["status"] == "complete"

    def test_download_estimate_fallback_on_error(self, monkeypatch):
        monkeypatch.setattr(dm, "hf_is_download_complete", lambda mid, deep_check=False: False)

        def hub_down(mid):
            raise RuntimeError("hub down")

        monkeypatch.setattr(dm, "list_model_files", hub_down)
        monkeypatch.setattr(dm, "_cache_dir", lambda mid: Path("/tmp/nonexistent-cache-xyz"))
        monkeypatch.setattr(dm, "sg_state", None)

        async def fake_worker(self, model_id, hint):
            return {"status": "complete", "model_id": model_id, "hint": hint}

        monkeypatch.setattr(dm.DownloadManager, "_download_worker", fake_worker)
        mgr = dm.DownloadManager()
        result = asyncio.run(mgr.download("gpt2", total_bytes_hint=1234))
        assert result["status"] == "complete"
        assert mgr.get_progress("gpt2")["total_bytes"] == 1234

    def test_worker_updates_progress_and_result(self, monkeypatch):
        fake, calls = _make_fake_downcraft()
        monkeypatch.setitem(sys.modules, "domains.infrastructure.hf_hub", fake)
        monkeypatch.setattr(dm, "_cache_dir", lambda mid: Path("/tmp/cache"))

        mgr = dm.DownloadManager()
        result = asyncio.run(mgr._download_worker("org/model", 1000))
        assert result["status"] == "complete"
        assert "cache_dir" in result
        assert mgr.get_progress("org/model")["status"] == "complete"
        assert mgr.get_progress("org/model")["percentage"] == 100.0
        assert calls["files"] == [("org/model", "model.safetensors")]

    def test_cleanup_stale(self, monkeypatch):
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


class TestSingleton:
    def test_get_download_manager_singleton(self, monkeypatch):
        monkeypatch.setattr(dm, "_download_manager", None)
        a = dm.get_download_manager()
        b = dm.get_download_manager()
        assert a is b


class TestImportFallback:
    def test_downcraft_missing_falls_back(self, monkeypatch):
        import importlib

        monkeypatch.setitem(sys.modules, "downcraft", None)
        monkeypatch.setitem(sys.modules, "downcraft.downloader", None)
        monkeypatch.setitem(sys.modules, "downcraft.state", None)
        mod = importlib.reload(dm)
        assert mod.sg_downloader is None
        assert mod.sg_state is None
        assert mod.get_cache_dir("org/model").endswith("models--org--model")
        assert mod.hf_is_download_complete("gpt2") is False
        assert mod.list_model_files("gpt2") == []
        sys.modules.pop("downcraft", None)
        sys.modules.pop("downcraft.downloader", None)
        sys.modules.pop("downcraft.state", None)
        importlib.reload(dm)
