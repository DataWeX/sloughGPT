"""Tests for shell.storage_device — StorageDevice file operations and ioctl."""

from __future__ import annotations

import os
import tempfile

import pytest

from domains.shell.storage_device import StorageDevice
from domains.shell.kernel_syscall import SyscallResult


@pytest.fixture
def dev(tmp_path):
    return StorageDevice(name="test-storage", base_path=str(tmp_path))


@pytest.fixture
def dev_with_file(dev, tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    return dev


# ── Basics ────────────────────────────────────────────────────────────────


class TestStorageDeviceBasics:

    def test_name(self, dev):
        assert dev.name == "test-storage"

    def test_default_name(self, tmp_path):
        d = StorageDevice(base_path=str(tmp_path))
        assert d.name == "storage"

    def test_info(self, dev):
        info = dev.info()
        assert info["name"] == "test-storage"
        assert info["type"] == "storage"
        assert info["open_files"] == 0

    def test_list_commands(self, dev):
        cmds = dev.list_commands()
        assert "READ" in cmds
        assert "WRITE" in cmds
        assert "OPEN" in cmds
        assert "CLOSE" in cmds
        assert "STAT" in cmds
        assert "MKDIR" in cmds
        assert "REMOVE" in cmds
        assert "INFO" in cmds
        assert cmds == sorted(cmds)


# ── ioctl ─────────────────────────────────────────────────────────────────


class TestStorageDeviceIoctl:

    def test_ioctl_unknown_command(self, dev):
        result = dev.ioctl("NONEXISTENT")
        assert isinstance(result, SyscallResult)
        assert not result.success
        assert "unknown command" in result.error

    def test_ioctl_open(self, dev_with_file):
        result = dev_with_file.ioctl("OPEN", "test.txt", "rb")
        assert result.success
        dev_with_file.close_file(result.value)

    def test_ioctl_read(self, dev_with_file):
        fd = dev_with_file.open_file("test.txt", "rb")
        result = dev_with_file.ioctl("READ", fd, 5)
        assert result.success
        assert result.value == b"hello"
        dev_with_file.close_file(fd)

    def test_ioctl_write(self, dev_with_file):
        fd = dev_with_file.open_file("test.txt", "wb")
        result = dev_with_file.ioctl("WRITE", fd, b"new data")
        assert result.success
        dev_with_file.close_file(fd)

    def test_ioctl_close(self, dev_with_file):
        fd = dev_with_file.open_file("test.txt", "rb")
        result = dev_with_file.ioctl("CLOSE", fd)
        assert result.success

    def test_ioctl_stat(self, dev_with_file):
        result = dev_with_file.ioctl("STAT", "test.txt")
        assert result.success
        assert result.value["size"] > 0

    def test_ioctl_list(self, dev):
        result = dev.ioctl("LIST", ".")
        assert result.success

    def test_ioctl_mkdir(self, dev):
        result = dev.ioctl("MKDIR", "subdir")
        assert result.success

    def test_ioctl_remove(self, dev_with_file):
        result = dev_with_file.ioctl("REMOVE", "test.txt")
        assert result.success

    def test_ioctl_rename(self, dev_with_file):
        result = dev_with_file.ioctl("RENAME", "test.txt", "renamed.txt")
        assert result.success

    def test_ioctl_exists(self, dev_with_file):
        result = dev_with_file.ioctl("EXISTS", "test.txt")
        assert result.success
        assert result.value is True

    def test_ioctl_info(self, dev):
        result = dev.ioctl("INFO")
        assert result.success

    def test_ioctl_exception(self, dev):
        result = dev.ioctl("READ", 999)
        assert not result.success
        assert "ioctl error" in result.error


# ── call interface ────────────────────────────────────────────────────────


class TestStorageDeviceCall:

    def test_call_success(self, dev_with_file):
        result = dev_with_file.call("EXISTS", "test.txt")
        assert result is True

    def test_call_failure_raises(self, dev):
        with pytest.raises(Exception, match="unknown command"):
            dev.call("NONEXISTENT")


# ── File operations ───────────────────────────────────────────────────────


class TestStorageDeviceFileOps:

    def test_open_and_read(self, dev_with_file):
        fd = dev_with_file.open_file("test.txt", "rb")
        data = dev_with_file.read(fd)
        assert data == b"hello world"
        dev_with_file.close_file(fd)

    def test_read_size(self, dev_with_file):
        fd = dev_with_file.open_file("test.txt", "rb")
        data = dev_with_file.read(fd, 5)
        assert data == b"hello"
        dev_with_file.close_file(fd)

    def test_write(self, dev):
        fd = dev.open_file("new.txt", "wb")
        dev.write(fd, b"content")
        dev.close_file(fd)
        assert dev.exists("new.txt")

    def test_read_bad_fd(self, dev):
        with pytest.raises(ValueError, match="bad fd"):
            dev.read(999)

    def test_write_bad_fd(self, dev):
        with pytest.raises(ValueError, match="bad fd"):
            dev.write(999, b"data")

    def test_seek_and_tell(self, dev_with_file):
        fd = dev_with_file.open_file("test.txt", "rb")
        dev_with_file.seek(fd, 5)
        pos = dev_with_file.tell(fd)
        assert pos == 5
        dev_with_file.close_file(fd)

    def test_seek_bad_fd(self, dev):
        with pytest.raises(ValueError, match="bad fd"):
            dev.seek(999, 0)

    def test_tell_bad_fd(self, dev):
        with pytest.raises(ValueError, match="bad fd"):
            dev.tell(999)

    def test_close_bad_fd(self, dev):
        assert dev.close_file(999) is False

    def test_stat(self, dev_with_file):
        s = dev_with_file.stat("test.txt")
        assert s["size"] > 0
        assert "mode" in s
        assert "mtime" in s

    def test_list_dir(self, dev_with_file):
        files = dev_with_file.list_dir(".")
        assert "test.txt" in files

    def test_mkdir(self, dev):
        assert dev.mkdir("subdir") is True
        assert os.path.isdir(os.path.join(dev._base_path, "subdir"))

    def test_remove(self, dev_with_file):
        assert dev_with_file.remove("test.txt") is True
        assert not dev_with_file.exists("test.txt")

    def test_rename(self, dev_with_file):
        assert dev_with_file.rename("test.txt", "new.txt") is True
        assert dev_with_file.exists("new.txt")
        assert not dev_with_file.exists("test.txt")

    def test_exists(self, dev_with_file):
        assert dev_with_file.exists("test.txt") is True
        assert dev_with_file.exists("nope.txt") is False
