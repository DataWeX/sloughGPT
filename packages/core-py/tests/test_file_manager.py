"""Tests for FileManager — unified path resolution (VFS → host FS)."""

import os
import pytest
import tempfile
from domains.shell.file_manager import FileManager, get_file_manager, reset_file_manager


class TestFileManager:
    def setup_method(self):
        reset_file_manager()
        self._tmpdir = tempfile.mkdtemp()
        self._fm = FileManager()
        # Write a test file to host FS
        self._test_file = os.path.join(self._tmpdir, "hello.txt")
        with open(self._test_file, "w") as f:
            f.write("Hello from host")

    def teardown_method(self):
        reset_file_manager()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_resolve_absolute(self):
        path = self._fm.resolve(self._test_file)
        assert path == self._test_file

    def test_resolve_relative(self):
        path = self._fm.resolve("hello.txt")
        # Relative path not in CWD returns None (not found on host FS either)
        # But if it resolves, it should be an absolute path
        if path is not None:
            assert os.path.isabs(path)

    def test_read_text_host(self):
        text = self._fm.read_text(self._test_file)
        assert text == "Hello from host"

    def test_read_text_nonexistent_returns_none(self):
        assert self._fm.read_text("/nonexistent/file.txt") is None

    def test_read_bytes_host(self):
        data = self._fm.read_bytes(self._test_file)
        assert data == b"Hello from host"

    def test_read_bytes_nonexistent_returns_none(self):
        assert self._fm.read_bytes("/nonexistent/file.txt") is None

    def test_write_text_host(self):
        out = os.path.join(self._tmpdir, "written.txt")
        result = self._fm.write_text(out, "written content")
        # write_text returns None on success, error string on failure
        assert result is None
        with open(out) as f:
            assert f.read() == "written content"

    def test_exists(self):
        assert self._fm.exists(self._test_file)
        assert not self._fm.exists("/nonexistent/file.txt")

    def test_isfile(self):
        assert self._fm.isfile(self._test_file)
        assert not self._fm.isfile(self._tmpdir)

    def test_isdir(self):
        assert self._fm.isdir(self._tmpdir)
        assert not self._fm.isdir(self._test_file)

    def test_listdir(self):
        entries = self._fm.listdir(self._tmpdir)
        assert "hello.txt" in entries

    def test_listdir_nonexistent(self):
        # listdir returns None when directory not found
        assert self._fm.listdir("/nonexistent/dir") is None

    def test_singleton(self):
        fm1 = get_file_manager()
        fm2 = get_file_manager()
        assert fm1 is fm2

    def test_reset_singleton(self):
        fm1 = get_file_manager()
        reset_file_manager()
        fm2 = get_file_manager()
        assert fm1 is not fm2
