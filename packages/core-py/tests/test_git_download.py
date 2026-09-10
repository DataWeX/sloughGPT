"""Tests for git_download — GitBackend."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domains.infrastructure.download_backend import DownloadBackend, FileEstimate
from domains.infrastructure.git_download import GitBackend, _get_cache_root


class TestGitBackend:
    def test_implements_protocol(self):
        """GitBackend satisfies DownloadBackend protocol."""
        backend = GitBackend()
        assert isinstance(backend, DownloadBackend)

    def test_parse_resource(self):
        """_parse_resource parses org/repo@branch."""
        backend = GitBackend()
        org, repo, branch = backend._parse_resource("org/huggingface@main")
        assert org == "org"
        assert repo == "huggingface"
        assert branch == "main"

    def test_parse_resource_no_branch(self):
        """_parse_resource without branch."""
        backend = GitBackend()
        org, repo, branch = backend._parse_resource("org/repo")
        assert org == "org"
        assert repo == "repo"
        assert branch is None

    def test_parse_resource_invalid(self):
        """_parse_resource raises on invalid format."""
        backend = GitBackend()
        with pytest.raises(ValueError, match="Invalid git resource format"):
            backend._parse_resource("invalid")

    def test_clone_url_https(self):
        """Clone URL is standard HTTPS."""
        backend = GitBackend()
        url = backend._clone_url("org", "repo")
        assert url == "https://github.com/org/repo.git"

    def test_clone_url_with_token(self):
        """Clone URL includes token when provided."""
        backend = GitBackend(token="abc123")
        url = backend._clone_url("org", "repo")
        assert "x-access-token:abc123" in url
        assert "github.com/org/repo.git" in url

    def test_get_cache_dir(self):
        """Cache dir resolves to expected path."""
        backend = GitBackend()
        d = backend.get_cache_dir("org/repo@main")
        assert "org__repo_at_main" in d
        assert "git" in d

    def test_is_cached_false(self, tmp_path):
        """Not cached when cache dir doesn't exist."""
        backend = GitBackend()
        with patch.object(backend, "_cache_dir", return_value=tmp_path / "empty"):
            assert backend.is_cached("org/repo") is False

    def test_is_cached_true(self, tmp_path):
        """Cached when manifest + files exist."""
        backend = GitBackend()
        cache = tmp_path / "cache"
        cache.mkdir()
        manifest = {"files": [{"path": "model.bin", "size": 100, "sha256": "abc"}]}
        (cache / ".manifest.json").write_text(json.dumps(manifest))
        (cache / "model.bin").write_bytes(b"x" * 100)

        with patch.object(backend, "_cache_dir", return_value=cache):
            assert backend.is_cached("org/repo") is True

    def test_cleanup(self, tmp_path):
        """cleanup removes cache directory."""
        backend = GitBackend()
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "file.txt").write_bytes(b"x")

        with patch.object(backend, "_cache_dir", return_value=cache):
            assert backend.cleanup("org/repo") is True
            assert not cache.exists()

    def test_cleanup_returns_false_when_missing(self, tmp_path):
        """cleanup returns False when dir doesn't exist."""
        backend = GitBackend()
        with patch.object(backend, "_cache_dir", return_value=tmp_path / "nope"):
            assert backend.cleanup("org/repo") is False

    def test_supports_compression_false(self):
        """Git backend doesn't use compression."""
        backend = GitBackend()
        assert backend.supports_compression("any") is False

    def test_supports_compressed_serve_false(self):
        """Git backend cannot serve files."""
        backend = GitBackend()
        assert backend.supports_compressed_serve() is False


class TestGitBackendDownload:
    @patch("domains.infrastructure.git_download.subprocess.run")
    def test_download_clones_repo(self, mock_run, tmp_path):
        """download() clones repo and scans files."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cache = tmp_path / "cache" / "org__repo"
        cache.mkdir(parents=True)
        (cache / "model.bin").write_bytes(b"model data")
        (cache / ".git").mkdir()

        backend = GitBackend()

        with patch.object(backend, "_cache_dir", return_value=cache):
            with patch.object(backend, "_scan_repo", return_value=[{"path": "model.bin", "size": 10, "sha256": "abc"}]):
                progress = []
                files = []
                result = backend.download(
                    "org/repo",
                    on_progress=lambda mid, d, t, s: progress.append((d, t)),
                    on_file_complete=lambda mid, p: files.append(p),
                )

                assert result["status"] == "completed"
                assert len(files) == 1
                assert "model.bin" in files[0]

    @patch("domains.infrastructure.git_download.subprocess.run")
    def test_download_clone_fails(self, mock_run, tmp_path):
        """download() returns error when clone fails."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="clone failed")

        backend = GitBackend()

        with patch.object(backend, "_cache_dir", return_value=tmp_path / "cache"):
            result = backend.download("org/repo", lambda *a: None, lambda *a: None)
            assert result["status"] == "error"
            assert "clone failed" in result["error"]

    @patch("domains.infrastructure.git_download.subprocess.run")
    def test_download_updates_existing(self, mock_run, tmp_path):
        """download() updates existing clone."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cache = tmp_path / "cache" / "org__repo"
        cache.mkdir(parents=True)

        backend = GitBackend()

        with patch.object(backend, "_cache_dir", return_value=cache):
            with patch.object(backend, "_scan_repo", return_value=[]):
                result = backend.download(
                    "org/repo",
                    on_progress=lambda *a: None,
                    on_file_complete=lambda *a: None,
                )
                assert result["status"] == "completed"
                # Should have called fetch/pull (update), not clone
                assert mock_run.call_count >= 2

    def test_list_files_from_manifest(self, tmp_path):
        """list_files reads from manifest."""
        backend = GitBackend()
        cache = tmp_path / "cache"
        cache.mkdir()
        manifest = {"files": [{"path": "model.bin", "size": 100, "sha256": "abc"}]}
        (cache / ".manifest.json").write_text(json.dumps(manifest))

        with patch.object(backend, "_cache_dir", return_value=cache):
            files = backend.list_files("org/repo")
            assert len(files) == 1
            assert files[0].path == "model.bin"
            assert files[0].size == 100

    def test_list_incomplete(self, tmp_path):
        """list_incomplete finds repos with missing files."""
        backend = GitBackend()
        root = _get_cache_root()
        root.mkdir(parents=True, exist_ok=True)

        # Create incomplete repo
        repo_dir = root / "org__repo"
        repo_dir.mkdir()
        manifest = {"files": [{"path": "model.bin", "size": 100, "sha256": "abc"}]}
        (repo_dir / ".manifest.json").write_text(json.dumps(manifest))
        # model.bin is missing

        try:
            result = backend.list_incomplete()
            # list_incomplete returns directory names, not resource IDs
            assert len(result) > 0
            assert any("org" in r for r in result)
        finally:
            import shutil
            shutil.rmtree(str(root), ignore_errors=True)
