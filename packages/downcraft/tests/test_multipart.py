"""Tests for downcraft.download.multipart — multi-part group download."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from downcraft.download.multipart import (
    GroupResult,
    PartResult,
    _filename_from_url,
    download_parts,
    parse_urls_file,
)

# ---------------------------------------------------------------------------
# _filename_from_url
# ---------------------------------------------------------------------------


class TestFilenameFromUrl:
    def test_simple(self):
        assert _filename_from_url("https://example.com/file.zip") == "file.zip"

    def test_nested_path(self):
        assert _filename_from_url("https://cdn.example.com/path/to/part01.rar") == "part01.rar"

    def test_trailing_slash(self):
        assert _filename_from_url("https://example.com/file.zip/") == "file.zip"

    def test_query_params(self):
        assert _filename_from_url("https://example.com/file.zip?token=abc") == "file.zip"

    def test_empty_path(self):
        assert _filename_from_url("https://example.com") == "download"


# ---------------------------------------------------------------------------
# parse_urls_file
# ---------------------------------------------------------------------------


class TestParseUrlsFile:
    def test_basic(self, tmp_path):
        f = tmp_path / "urls.txt"
        f.write_text("https://example.com/a.zip\nhttps://example.com/b.zip\n")
        urls = parse_urls_file(f)
        assert urls == ["https://example.com/a.zip", "https://example.com/b.zip"]

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "urls.txt"
        f.write_text("https://example.com/a.zip\n\n\nhttps://example.com/b.zip\n")
        urls = parse_urls_file(f)
        assert len(urls) == 2

    def test_skips_comments(self, tmp_path):
        f = tmp_path / "urls.txt"
        f.write_text("# this is a comment\nhttps://example.com/a.zip\n# another\n")
        urls = parse_urls_file(f)
        assert urls == ["https://example.com/a.zip"]

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "urls.txt"
        f.write_text("  https://example.com/a.zip  \n")
        urls = parse_urls_file(f)
        assert urls == ["https://example.com/a.zip"]

    def test_empty_file(self, tmp_path):
        f = tmp_path / "urls.txt"
        f.write_text("")
        urls = parse_urls_file(f)
        assert urls == []


# ---------------------------------------------------------------------------
# PartResult
# ---------------------------------------------------------------------------


class TestPartResult:
    def test_defaults(self):
        r = PartResult(url="https://x.com/f.zip", dest=Path("/tmp/f.zip"), status="complete")
        assert r.bytes_downloaded == 0
        assert r.elapsed == 0.0
        assert r.error == ""


# ---------------------------------------------------------------------------
# GroupResult
# ---------------------------------------------------------------------------


class TestGroupResult:
    def test_all_complete(self):
        parts = [
            PartResult(url="a", dest=Path("a"), status="complete"),
            PartResult(url="b", dest=Path("b"), status="complete"),
        ]
        g = GroupResult(group_key="g", dest_dir="/tmp", status="complete", parts=parts)
        assert g.completed_count == 2
        assert g.failed_count == 0
        assert g.total_count == 2

    def test_partial(self):
        parts = [
            PartResult(url="a", dest=Path("a"), status="complete"),
            PartResult(url="b", dest=Path("b"), status="failed", error="404"),
        ]
        g = GroupResult(group_key="g", dest_dir="/tmp", status="partial", parts=parts)
        assert g.completed_count == 1
        assert g.failed_count == 1
        assert g.total_count == 2

    def test_empty(self):
        g = GroupResult(group_key="g", dest_dir="/tmp", status="complete")
        assert g.completed_count == 0
        assert g.total_count == 0


# ---------------------------------------------------------------------------
# download_parts (integration with range_server)
# ---------------------------------------------------------------------------


class TestDownloadParts:
    def _mock_download(self, files: dict[str, bytes]):
        """Return a mock for download_file that writes files and calls on_chunk."""

        def _download(url, dest, on_chunk=None, **kwargs):
            path = url.rsplit("/", 1)[-1] if "/" in url else url
            if path in files:
                dest.parent.mkdir(parents=True, exist_ok=True)
                data = files[path]
                dest.write_bytes(data)
                if on_chunk:
                    on_chunk(len(data), len(data))
                return dest
            from downcraft.download.http import DownloadError

            raise DownloadError(f"Not found: {url}")

        return _download

    def test_download_three_parts(self, tmp_path):
        """Download 3 parts with mocked HTTP."""
        files = {
            "part01.zip": b"hello-part1",
            "part02.zip": b"hello-part2",
            "part03.zip": b"hello-part3",
        }
        urls = [
            "https://cdn.example.com/part01.zip",
            "https://cdn.example.com/part02.zip",
            "https://cdn.example.com/part03.zip",
        ]
        dest = tmp_path / "downloaded"

        with patch(
            "downcraft.download.multipart.download_file", side_effect=self._mock_download(files)
        ):
            result = download_parts(urls, dest, group_key="test-3parts")

        assert result.status == "complete"
        assert result.completed_count == 3
        assert result.failed_count == 0
        assert result.total_count == 3
        assert result.total_bytes == 33
        assert (dest / "part01.zip").read_bytes() == b"hello-part1"
        assert (dest / "part02.zip").read_bytes() == b"hello-part2"
        assert (dest / "part03.zip").read_bytes() == b"hello-part3"

    def test_custom_filenames(self, tmp_path):
        """Download with explicit filenames."""
        files = {"a.bin": b"data-a", "b.bin": b"data-b"}
        urls = ["https://cdn.example.com/a.bin", "https://cdn.example.com/b.bin"]
        dest = tmp_path / "named"

        with patch(
            "downcraft.download.multipart.download_file", side_effect=self._mock_download(files)
        ):
            result = download_parts(urls, dest, filenames=["first.bin", "second.bin"])

        assert result.status == "complete"
        assert (dest / "first.bin").read_bytes() == b"data-a"
        assert (dest / "second.bin").read_bytes() == b"data-b"

    def test_partial_failure(self, tmp_path):
        """Handle mix of success and failure."""
        from downcraft.download.http import DownloadError

        def _download_partial(url, dest, on_chunk=None, **kwargs):
            if "missing" in url:
                raise DownloadError("Not found")
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = b"good-data"
            dest.write_bytes(data)
            if on_chunk:
                on_chunk(len(data), len(data))
            return dest

        urls = ["https://cdn.example.com/ok.bin", "https://cdn.example.com/missing.bin"]
        dest = tmp_path / "partial"

        with patch("downcraft.download.multipart.download_file", side_effect=_download_partial):
            result = download_parts(urls, dest, group_key="test-partial")

        assert result.status == "partial"
        assert result.completed_count == 1
        assert result.failed_count == 1

    def test_progress_callback(self, tmp_path):
        """Progress callback is called."""
        files = {"file.bin": b"content-here"}
        urls = ["https://cdn.example.com/file.bin"]
        progress_calls = []

        def on_progress(part_idx, done, total, speed):
            progress_calls.append((part_idx, done, total))

        with patch(
            "downcraft.download.multipart.download_file", side_effect=self._mock_download(files)
        ):
            result = download_parts(urls, tmp_path, on_progress=on_progress)

        assert result.status == "complete"
        assert len(progress_calls) >= 1

    def test_part_complete_callback(self, tmp_path):
        """Part complete callback is called for each part."""
        files = {"x.bin": b"xdata", "y.bin": b"ydata"}
        urls = ["https://cdn.example.com/x.bin", "https://cdn.example.com/y.bin"]
        completed = []

        def on_part(part):
            completed.append(part.status)

        with patch(
            "downcraft.download.multipart.download_file", side_effect=self._mock_download(files)
        ):
            result = download_parts(urls, tmp_path, on_part_complete=on_part)

        assert result.status == "complete"
        assert completed == ["complete", "complete"]

    def test_creates_dest_dir(self, tmp_path):
        """Creates destination directory if it doesn't exist."""
        files = {"f.bin": b"data"}
        urls = ["https://cdn.example.com/f.bin"]
        dest = tmp_path / "nested" / "deep" / "dir"

        with patch(
            "downcraft.download.multipart.download_file", side_effect=self._mock_download(files)
        ):
            result = download_parts(urls, dest)

        assert result.status == "complete"
        assert dest.exists()

    def test_empty_urls(self, tmp_path):
        """Empty URL list produces empty result."""
        result = download_parts([], tmp_path / "empty")
        assert result.status == "complete"
        assert result.total_count == 0
