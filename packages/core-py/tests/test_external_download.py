"""Tests for external_download — ExternalDownloadBackend."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domains.infrastructure.download_backend import DownloadBackend, FileEstimate
from domains.infrastructure.external_download import ExternalDownloadBackend, _get_cache_root


class TestExternalDownloadBackend:
    def test_implements_protocol(self):
        """ExternalDownloadBackend satisfies DownloadBackend protocol."""
        backend = ExternalDownloadBackend("http://localhost:8000")
        assert isinstance(backend, DownloadBackend)

    def test_is_cached_false_when_empty(self, tmp_path):
        """Not cached when cache dir doesn't exist."""
        backend = ExternalDownloadBackend("http://localhost:8000")
        with patch.object(backend, "_cache_dir", return_value=tmp_path / "empty"):
            assert backend.is_cached("model-x") is False

    def test_is_cached_true_with_manifest(self, tmp_path):
        """Cached when manifest + files exist."""
        backend = ExternalDownloadBackend("http://localhost:8000")
        cache = tmp_path / "model"
        cache.mkdir()
        manifest = {"files": [{"path": "model.bin", "size": 100, "sha256": "abc"}]}
        (cache / ".manifest.json").write_text(json.dumps(manifest))
        (cache / "model.bin").write_bytes(b"x" * 100)

        with patch.object(backend, "_cache_dir", return_value=cache):
            assert backend.is_cached("model") is True

    def test_is_cached_false_when_file_missing(self, tmp_path):
        """Not cached when a file is missing."""
        backend = ExternalDownloadBackend("http://localhost:8000")
        cache = tmp_path / "model"
        cache.mkdir()
        manifest = {"files": [{"path": "model.bin", "size": 100, "sha256": "abc"}]}
        (cache / ".manifest.json").write_text(json.dumps(manifest))

        with patch.object(backend, "_cache_dir", return_value=cache):
            assert backend.is_cached("model") is False

    def test_get_cache_dir(self):
        """Cache dir resolves to expected path."""
        backend = ExternalDownloadBackend("http://localhost:8000")
        d = backend.get_cache_dir("org/model-name")
        assert "org__model-name" in d
        assert "external" in d

    def test_estimate_total(self, tmp_path):
        """Estimate sums file sizes from manifest."""
        backend = ExternalDownloadBackend("http://localhost:8000")
        cache = tmp_path / "model"
        cache.mkdir()
        manifest = {"files": [
            {"path": "a.bin", "size": 500},
            {"path": "b.bin", "size": 300},
        ]}
        (cache / ".manifest.json").write_text(json.dumps(manifest))

        with patch.object(backend, "_cache_dir", return_value=cache):
            assert backend.estimate_total("model") == 800

    def test_list_files(self, tmp_path):
        """list_files returns FileEstimate objects."""
        backend = ExternalDownloadBackend("http://localhost:8000")
        cache = tmp_path / "model"
        cache.mkdir()
        manifest = {"files": [
            {"path": "model.bin", "size": 1024, "sha256": "abc123"},
        ]}
        (cache / ".manifest.json").write_text(json.dumps(manifest))

        with patch.object(backend, "_cache_dir", return_value=cache):
            files = backend.list_files("model")
            assert len(files) == 1
            assert files[0].path == "model.bin"
            assert files[0].size == 1024
            assert files[0].checksum == "abc123"
            assert "file/model.bin" in files[0].download_url

    def test_cleanup(self, tmp_path):
        """cleanup removes cache directory."""
        backend = ExternalDownloadBackend("http://localhost:8000")
        cache = tmp_path / "model"
        cache.mkdir()
        (cache / "file.bin").write_bytes(b"x")

        with patch.object(backend, "_cache_dir", return_value=cache):
            assert backend.cleanup("model") is True
            assert not cache.exists()

    def test_cleanup_returns_false_when_missing(self, tmp_path):
        """cleanup returns False when dir doesn't exist."""
        backend = ExternalDownloadBackend("http://localhost:8000")
        with patch.object(backend, "_cache_dir", return_value=tmp_path / "nope"):
            assert backend.cleanup("model") is False

    def test_list_incomplete(self, tmp_path):
        """list_incomplete returns resources with missing files."""
        backend = ExternalDownloadBackend("http://localhost:8000")
        root = tmp_path / "external"
        root.mkdir()

        # Complete resource
        complete = root / "complete-model"
        complete.mkdir()
        manifest = {"files": [{"path": "ok.bin", "size": 100}]}
        (complete / ".manifest.json").write_text(json.dumps(manifest))
        (complete / "ok.bin").write_bytes(b"x" * 100)

        # Incomplete resource
        incomplete = root / "incomplete-model"
        incomplete.mkdir()
        manifest2 = {"files": [{"path": "missing.bin", "size": 100}]}
        (incomplete / ".manifest.json").write_text(json.dumps(manifest2))

        with patch("domains.infrastructure.external_download._get_cache_root", return_value=root):
            result = backend.list_incomplete()
            assert "incomplete-model" in result
            assert "complete-model" not in result

    def test_supports_compression_default(self):
        """Compression enabled by default."""
        backend = ExternalDownloadBackend("http://localhost:8000")
        assert backend.supports_compression("any") is True

    def test_supports_compression_disabled(self):
        """Compression can be disabled."""
        backend = ExternalDownloadBackend("http://localhost:8000", compressed=False)
        assert backend.supports_compression("any") is False

    def test_supports_compressed_serve_false(self):
        """External backend cannot serve files."""
        backend = ExternalDownloadBackend("http://localhost:8000")
        assert backend.supports_compressed_serve() is False

    def test_model_url(self):
        """URL construction."""
        backend = ExternalDownloadBackend("http://192.168.1.100:8000")
        assert backend._model_url("my-model") == "http://192.168.1.100:8000/models/my-model"
        assert backend._file_url("my-model", "model.bin") == "http://192.168.1.100:8000/models/my-model/file/model.bin"
        assert backend._manifest_url("my-model") == "http://192.168.1.100:8000/models/my-model/manifest.json"

    def test_model_url_strips_trailing_slash(self):
        """Trailing slash in base URL is handled."""
        backend = ExternalDownloadBackend("http://localhost:8000/")
        assert backend._model_url("m") == "http://localhost:8000/models/m"


class TestExternalDownloadDownload:
    def test_download_uses_compressed_when_enabled(self, tmp_path):
        """download() uses CompressedDownloader when compressed=True."""
        backend = ExternalDownloadBackend("http://localhost:8000", compressed=True)

        manifest = {"files": [{"path": "model.bin", "size": 100, "sha256": "abc"}]}

        with patch.object(backend, "_fetch_manifest", return_value=manifest), \
             patch.object(backend, "_cache_dir", return_value=tmp_path / "out"), \
             patch("domains.infrastructure.compressed_transfer.CompressedDownloader") as MockDL:
            mock_inst = MockDL.return_value
            mock_inst.download_from_url.return_value = MagicMock(success=True)

            progress = []
            files = []
            result = backend.download(
                "model",
                on_progress=lambda mid, d, t, s: progress.append((d, t)),
                on_file_complete=lambda mid, p: files.append(p),
            )

            assert result["status"] == "completed"
            mock_inst.download_from_url.assert_called_once()

    def test_download_fetch_manifest_failure(self):
        """download() returns error when manifest fetch fails."""
        backend = ExternalDownloadBackend("http://localhost:8000")

        with patch.object(backend, "_fetch_manifest", return_value=None):
            result = backend.download("model", None, None)
            assert result["status"] == "error"
            assert "Could not fetch manifest" in result["error"]

    def test_download_empty_manifest(self):
        """download() returns error when manifest has no files."""
        backend = ExternalDownloadBackend("http://localhost:8000")

        with patch.object(backend, "_fetch_manifest", return_value={"files": []}):
            result = backend.download("model", None, None)
            assert result["status"] == "error"
            assert "no files" in result["error"]
