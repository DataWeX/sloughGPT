"""Tests for the model download seam — ``DownloadManager.download_model()``.

Two download paths meet at this seam:

* **generic** — an explicit URL, handled by ``DownloadManager.download()``
  (downcraft, resume/integrity/cancel), and
* **model-id** — no URL, handled by the HF backend
  (``HFDownloadBackend`` → ``download_hf_model``).

These tests pin the wiring that regressed on 2026-09-16 (commit
``a170a7694`` added ``url`` to ``download()`` without migrating callers, so
``POST /models/download`` passed ``total_bytes_hint`` positionally into
``url`` → ``Invalid URL '0'``) and the failure status that never reached a
terminal state (a failed download stayed ``downloading`` forever, which
both spun the UI card at 0% and locked the model out of any retry).
"""

import asyncio

import domain.infrastructure._internal.download_manager as dm
import domain.infrastructure._internal.hf_hub as hf_hub


class _StubDowncraft:
    """Stand-in for the downcraft module: fails fast, no network."""

    def __init__(self, error: Exception | None = None):
        self.error = error or RuntimeError("Invalid URL 'stub'")
        self.calls: list[dict] = []

    def download(self, **kwargs):
        self.calls.append(kwargs)
        raise self.error


def _stub_downcraft(monkeypatch) -> _StubDowncraft:
    stub = _StubDowncraft()
    monkeypatch.setattr(dm, "downcraft", stub, raising=False)
    monkeypatch.setattr(dm, "_downcraft", stub, raising=False)
    return stub


class TestFailureStatus:
    """An exhausted download must reach a terminal state."""

    def test_exhausted_retries_mark_failed(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SLO_CACHE_DIR", str(tmp_path))
        stub = _stub_downcraft(monkeypatch)
        mgr = dm.DownloadManager()

        result = asyncio.run(
            mgr.download("bad-url-model", url="http://host/file.bin", max_retries=0)
        )

        assert result["status"] == "failed"
        assert len(stub.calls) == 1
        progress = mgr.get_progress("bad-url-model")
        assert progress is not None, "failed download must leave a progress entry"
        assert progress["status"] == "failed"
        assert "Invalid URL" in progress["error"]

    def test_failed_download_can_be_retried(self, monkeypatch, tmp_path):
        """A failed entry must not keep the model locked as 'downloading'."""
        monkeypatch.setenv("SLO_CACHE_DIR", str(tmp_path))
        _stub_downcraft(monkeypatch)
        mgr = dm.DownloadManager()

        asyncio.run(mgr.download("retry-me", url="http://host/file.bin", max_retries=0))

        assert mgr.is_downloading("retry-me") is False
        assert mgr.get_progress("retry-me")["status"] == "failed"

    def test_failure_notifies_progress_callbacks(self, monkeypatch, tmp_path):
        """The UI subscribes via on_progress — failures must be pushed too."""
        monkeypatch.setenv("SLO_CACHE_DIR", str(tmp_path))
        _stub_downcraft(monkeypatch)
        mgr = dm.DownloadManager()
        seen: list[str] = []
        mgr.on_progress("notify-me", lambda entry: seen.append(entry.get("status", "")))

        asyncio.run(mgr.download("notify-me", url="http://host/file.bin", max_retries=0))

        assert "failed" in seen


class TestDownloadModelUrlBranch:
    """A URL must reach ``download()`` as a keyword, never positionally."""

    def test_url_wins_and_is_passed_as_keyword(self, monkeypatch):
        mgr = dm.DownloadManager()
        seen: dict = {}

        async def fake_download(model_id, url="", dest="", total_bytes_hint=0, **kw):
            seen.update(
                model_id=model_id, url=url, dest=dest, total_bytes_hint=total_bytes_hint
            )
            return {"status": "complete", "model_id": model_id}

        monkeypatch.setattr(mgr, "download", fake_download)

        def _no_hf():
            raise AssertionError("HF path must not be taken when a URL is given")

        monkeypatch.setattr(hf_hub, "HFDownloadBackend", _no_hf, raising=False)

        result = asyncio.run(
            mgr.download_model(
                "artifact", url="https://host/x.pdf", dest="/tmp/x.pdf", total_bytes_hint=42
            )
        )

        assert result["status"] == "complete"
        assert seen == {
            "model_id": "artifact",
            "url": "https://host/x.pdf",
            "dest": "/tmp/x.pdf",
            "total_bytes_hint": 42,
        }

    def test_already_downloading_short_circuits(self):
        mgr = dm.DownloadManager()
        mgr._set_progress("busy-model", status=dm.DownloadStatus.DOWNLOADING)

        result = asyncio.run(mgr.download_model("busy-model", url="https://host/x.bin"))

        assert result["status"] == "already_downloading"


class TestDownloadModelHfBranch:
    """No URL → the HF backend does the work, registry reflects it."""

    @staticmethod
    def _fake_backend(monkeypatch, on_download):
        class _FakeHFBackend:
            def download(self, resource_id, on_progress, on_file_complete):
                return on_download(resource_id, on_progress, on_file_complete)

        monkeypatch.setattr(hf_hub, "HFDownloadBackend", _FakeHFBackend, raising=False)

    def test_no_url_delegates_to_hf_backend(self, monkeypatch):
        mgr = dm.DownloadManager()
        seen: dict = {}

        def _download(resource_id, on_progress, on_file_complete):
            seen["id"] = resource_id
            if on_progress:
                on_progress(resource_id, 50, 100, 250.0)
            return {"status": "complete", "cache_dir": "/hf/cache", "total_bytes": 100}

        self._fake_backend(monkeypatch, _download)

        result = asyncio.run(mgr.download_model("gpt2"))

        assert seen["id"] == "gpt2"
        assert result["status"] == "complete"
        progress = mgr.get_progress("gpt2")
        assert progress["status"] == "complete"
        assert progress["percentage"] == 100.0
        assert progress["total_bytes"] == 100
        assert mgr.is_downloading("gpt2") is False

    def test_hf_progress_is_published(self, monkeypatch):
        """Callbacks registered with on_progress must see HF progress."""
        mgr = dm.DownloadManager()

        def _download(resource_id, on_progress, on_file_complete):
            on_progress(resource_id, 25, 100, 128.0)
            return {"status": "complete", "cache_dir": "/hf/cache", "total_bytes": 100}

        self._fake_backend(monkeypatch, _download)
        seen: list[dict] = []
        mgr.on_progress("hf-progress", lambda entry: seen.append(dict(entry)))

        asyncio.run(mgr.download_model("hf-progress"))

        assert any(e.get("percentage") == 25.0 for e in seen)
        assert any(e.get("status") == "complete" for e in seen)

    def test_hf_failure_marks_failed(self, monkeypatch):
        mgr = dm.DownloadManager()

        def _download(resource_id, on_progress, on_file_complete):
            raise RuntimeError("hub unreachable")

        self._fake_backend(monkeypatch, _download)

        result = asyncio.run(mgr.download_model("gpt2"))

        assert result["status"] == "failed"
        assert "hub unreachable" in result["error"]
        progress = mgr.get_progress("gpt2")
        assert progress["status"] == "failed"
        assert mgr.is_downloading("gpt2") is False

    def test_already_cached_hf_model_returns_cached(self, monkeypatch):
        mgr = dm.DownloadManager()

        def _download(resource_id, on_progress, on_file_complete):
            return {"status": "already_cached", "cache_dir": "/hf/cache"}

        self._fake_backend(monkeypatch, _download)

        result = asyncio.run(mgr.download_model("gpt2"))

        assert result["status"] == "already_cached"
        assert mgr.get_progress("gpt2")["status"] == "complete"
        assert mgr.is_downloading("gpt2") is False
