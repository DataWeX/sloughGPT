"""Tests for VirtualFS — inode-based filesystem on VirtualDisk."""

import os
import pytest
import tempfile
from domains.shell.virtual_disk import VirtualDisk
from domains.shell.virtual_fs import VirtualFS, ROOT_INODE


class TestVirtualFS:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._disk_path = os.path.join(self._tmpdir, "test.dsk")
        self._disk = VirtualDisk(self._disk_path, size_mb=2, create=True)
        self._vfs = VirtualFS(self._disk)

    def teardown_method(self):
        self._disk.close()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_root_inode_exists(self):
        assert self._vfs.exists("/")

    def test_mkdir(self):
        self._vfs.mkdir("/data")
        assert self._vfs.exists("/data")
        assert self._vfs.isdir("/data") if hasattr(self._vfs, 'isdir') else True

    def test_mkdir_nested(self):
        self._vfs.mkdir("/data")
        self._vfs.mkdir("/data/models")
        assert self._vfs.exists("/data/models")

    def test_mkdir_duplicate_raises(self):
        self._vfs.mkdir("/data")
        with pytest.raises(FileExistsError):
            self._vfs.mkdir("/data")

    def test_mkdir_nonexistent_parent_raises(self):
        with pytest.raises(FileNotFoundError):
            self._vfs.mkdir("/nonexistent/file.txt")

    def test_create_file(self):
        inode_num = self._vfs.create("/hello.txt")
        assert inode_num > 0
        assert self._vfs.exists("/hello.txt")

    def test_create_file_duplicate_raises(self):
        self._vfs.create("/hello.txt")
        with pytest.raises(FileExistsError):
            self._vfs.create("/hello.txt")

    def test_write_and_read(self):
        self._vfs.create("/hello.txt")
        self._vfs.write("/hello.txt", b"Hello, World!")
        data = self._vfs.read("/hello.txt")
        assert data == b"Hello, World!"

    def test_write_creates_if_not_exists(self):
        self._vfs.write("/new.txt", b"auto-created")
        assert self._vfs.read("/new.txt") == b"auto-created"

    def test_read_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            self._vfs.read("/nope.txt")

    def test_read_directory_raises(self):
        self._vfs.mkdir("/dir")
        with pytest.raises(IsADirectoryError):
            self._vfs.read("/dir")

    def test_delete_file(self):
        self._vfs.create("/temp.txt")
        self._vfs.write("/temp.txt", b"delete me")
        assert self._vfs.delete("/temp.txt")
        assert not self._vfs.exists("/temp.txt")

    def test_delete_nonexistent_returns_false(self):
        assert not self._vfs.delete("/nope.txt")

    def test_delete_empty_directory(self):
        self._vfs.mkdir("/empty")
        assert self._vfs.delete("/empty")
        assert not self._vfs.exists("/empty")

    def test_listdir(self):
        self._vfs.create("/a.txt")
        self._vfs.create("/b.txt")
        self._vfs.mkdir("/dir")
        entries = self._vfs.listdir("/")
        assert "a.txt" in entries
        assert "b.txt" in entries
        assert "dir" in entries

    def test_listdir_empty(self):
        entries = self._vfs.listdir("/")
        assert entries == []

    def test_listdir_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            self._vfs.listdir("/nope")

    def test_rename(self):
        self._vfs.create("/old.txt")
        self._vfs.write("/old.txt", b"content")
        self._vfs.rename("/old.txt", "/new.txt")
        assert not self._vfs.exists("/old.txt")
        assert self._vfs.read("/new.txt") == b"content"

    def test_rename_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            self._vfs.rename("/nope.txt", "/new.txt")

    def test_rename_to_existing_raises(self):
        self._vfs.create("/a.txt")
        self._vfs.create("/b.txt")
        with pytest.raises(FileExistsError):
            self._vfs.rename("/a.txt", "/b.txt")

    def test_stat(self):
        self._vfs.create("/file.txt")
        self._vfs.write("/file.txt", b"hello")
        st = self._vfs.stat("/file.txt")
        assert st["is_file"]
        assert not st["is_dir"]
        assert st["size"] == 5
        assert st["inode"] > 0

    def test_stat_directory(self):
        self._vfs.mkdir("/dir")
        st = self._vfs.stat("/dir")
        assert st["is_dir"]
        assert not st["is_file"]

    def test_stat_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            self._vfs.stat("/nope.txt")

    def test_large_file(self):
        """Test multi-block file (larger than block size)."""
        big_data = b"x" * 10000  # larger than 4096 block size
        self._vfs.write("/big.txt", big_data)
        read_back = self._vfs.read("/big.txt")
        assert read_back == big_data

    def test_write_overwrite(self):
        self._vfs.write("/file.txt", b"original")
        self._vfs.write("/file.txt", b"overwritten")
        assert self._vfs.read("/file.txt") == b"overwritten"

    def test_read_text(self):
        self._vfs.write("/utf8.txt", "Hello, Unicode: café".encode("utf-8"))
        text = self._vfs.read_text("/utf8.txt")
        assert text == "Hello, Unicode: café"

    def test_write_text(self):
        self._vfs.write_text("/text.txt", "Hello, World!")
        data = self._vfs.read("/text.txt")
        assert data == b"Hello, World!"

    def test_directory_in_directory(self):
        self._vfs.mkdir("/a")
        self._vfs.mkdir("/a/b")
        self._vfs.mkdir("/a/b/c")
        self._vfs.create("/a/b/c/file.txt")
        assert self._vfs.exists("/a/b/c/file.txt")
        entries = self._vfs.listdir("/a/b/c")
        assert "file.txt" in entries
