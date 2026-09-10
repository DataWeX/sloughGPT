"""Tests for local_download — LocalFileBackend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from domains.infrastructure.download_backend import DownloadBackend, FileEstimate
from domains.infrastructure.local_download import LocalFileBackend, _get_cache_root


class TestLocalFileBackend:
    def test_implements_protocol(self):
        """LocalFileBackend satisfies DownloadBackend protocol."""
        backend = LocalFileBackend("/tmp/models")
        assert isinstance(backend, DownloadBackend)

    def test_is_cached_false_when_empty(self, tmp_path):
        """Not cached when cache dir doesn't exist."""
        backend = LocalFileBackend(str(tmp_path))
        with patch.object(backend, "_cache_dir", return_value=tmp_path / "empty"):
            assert backend.is_cached("model-x") is False

    def test_is_cached_true_with_manifest(self, tmp_path):
        """Cached when manifest + files exist."""
        backend = LocalFileBackend(str(tmp_path))
        cache = tmp_path / "cache" / "model"
        cache.mkdir(parents=True)
        manifest = {"files": [{"path": "model.bin", "size": 100, "sha256": "abc"}]}
        (cache / ".manifest.json").write_text(json.dumps(manifest))
        (cache / "model.bin").write_bytes(b"x" * 100)

        with patch.object(backend, "_cache_dir", return_value=cache):
            assert backend.is_cached("model") is True

    def test_get_cache_dir(self):
        """Cache dir resolves to expected path."""
        backend = LocalFileBackend("/tmp/models")
        d = backend.get_cache_dir("org/model-name")
        assert "org__model-name" in d
        assert "local" in d

    def test_estimate_total_from_source(self, tmp_path):
        """Estimate sums file sizes from source directory."""
        source = tmp_path / "source" / "model"
        source.mkdir(parents=True)
        (source / "a.bin").write_bytes(b"x" * 500)
        (source / "b.bin").write_bytes(b"y" * 300)

        backend = LocalFileBackend(str(tmp_path / "source"))
        assert backend.estimate_total("model") == 800

    def test_list_files(self, tmp_path):
        """list_files returns FileEstimate objects."""
        source = tmp_path / "source" / "model"
        source.mkdir(parents=True)
        (source / "model.bin").write_bytes(b"x" * 1024)

        backend = LocalFileBackend(str(tmp_path / "source"))
        files = backend.list_files("model")
        assert len(files) == 1
        assert files[0].path == "model.bin"
        assert files[0].size == 1024
        assert len(files[0].checksum) == 64  # SHA-256
        assert files[0].download_url.startswith("file://")

    def test_cleanup(self, tmp_path):
        """cleanup removes cache directory."""
        backend = LocalFileBackend(str(tmp_path))
        cache = tmp_path / "cache" / "model"
        cache.mkdir(parents=True)
        (cache / "file.bin").write_bytes(b"x")

        with patch.object(backend, "_cache_dir", return_value=cache):
            assert backend.cleanup("model") is True
            assert not cache.exists()

    def test_cleanup_returns_false_when_missing(self, tmp_path):
        """cleanup returns False when dir doesn't exist."""
        backend = LocalFileBackend(str(tmp_path))
        with patch.object(backend, "_cache_dir", return_value=tmp_path / "nope"):
            assert backend.cleanup("model") is False

    def test_supports_compression_false(self):
        """Local backend doesn't use compression."""
        backend = LocalFileBackend("/tmp/models")
        assert backend.supports_compression("any") is False

    def test_supports_compressed_serve_false(self):
        """Local backend cannot serve files."""
        backend = LocalFileBackend("/tmp/models")
        assert backend.supports_compressed_serve() is False


class TestLocalFileBackendDownload:
    def test_download_copies_files(self, tmp_path):
        """download() copies files from source to cache."""
        source = tmp_path / "source" / "model"
        source.mkdir(parents=True)
        (source / "model.bin").write_bytes(b"hello world")
        (source / "config.json").write_text('{"key": "value"}')

        cache = tmp_path / "cache"
        backend = LocalFileBackend(str(tmp_path / "source"))

        with patch.object(backend, "_cache_dir", return_value=cache / "model"):
            progress = []
            files = []
            result = backend.download(
                "model",
                on_progress=lambda mid, d, t, s: progress.append((d, t)),
                on_file_complete=lambda mid, p: files.append(p),
            )

            assert result["status"] == "completed"
            assert result["total_bytes"] > 0
            assert len(files) == 2
            assert (cache / "model" / "model.bin").read_bytes() == b"hello world"
            assert (cache / "model" / "config.json").read_text() == '{"key": "value"}'

    def test_download_source_not_found(self, tmp_path):
        """download() returns error when source doesn't exist."""
        backend = LocalFileBackend(str(tmp_path / "nonexistent"))
        result = backend.download("model", None, None)
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_download_empty_source(self, tmp_path):
        """download() returns error when source is empty."""
        source = tmp_path / "source" / "model"
        source.mkdir(parents=True)

        backend = LocalFileBackend(str(tmp_path / "source"))
        result = backend.download("model", None, None)
        assert result["status"] == "error"
        assert "no files" in result["error"].lower()

    def test_download_preserves_directory_structure(self, tmp_path):
        """download() preserves subdirectory structure."""
        source = tmp_path / "source" / "model"
        (source / "dir").mkdir(parents=True)
        (source / "dir" / "nested.bin").write_bytes(b"nested")

        cache = tmp_path / "cache"
        backend = LocalFileBackend(str(tmp_path / "source"))

        with patch.object(backend, "_cache_dir", return_value=cache / "model"):
            result = backend.download("model", lambda *a: None, lambda *a: None)
            assert result["status"] == "completed"
            assert (cache / "model" / "dir" / "nested.bin").read_bytes() == b"nested"


from unittest.mock import patch
