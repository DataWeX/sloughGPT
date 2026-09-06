"""Tests for shell.display_device — DisplayDevice output operations and ioctl."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from domains.shell.display_device import DisplayDevice
from domains.shell.kernel_syscall import SyscallResult


@pytest.fixture
def dev():
    return DisplayDevice(name="test-display")


# ── Basics ────────────────────────────────────────────────────────────────


class TestDisplayDeviceBasics:

    def test_name(self, dev):
        assert dev.name == "test-display"

    def test_default_name(self):
        d = DisplayDevice()
        assert d.name == "display"

    def test_info(self, dev):
        info = dev.info()
        assert info["name"] == "test-display"
        assert info["type"] == "display"
        assert "buffer_size" in info
        assert "cursor" in info

    def test_list_commands(self, dev):
        cmds = dev.list_commands()
        assert "PRINT" in cmds
        assert "PRINTLN" in cmds
        assert "CLEAR" in cmds
        assert "MOVE" in cmds
        assert "COLOR" in cmds
        assert "STYLE" in cmds
        assert "FLUSH" in cmds
        assert "WRITE" in cmds
        assert "INFO" in cmds
        assert cmds == sorted(cmds)


# ── ioctl ─────────────────────────────────────────────────────────────────


class TestDisplayDeviceIoctl:

    def test_ioctl_unknown_command(self, dev):
        result = dev.ioctl("NONEXISTENT")
        assert isinstance(result, SyscallResult)
        assert not result.success
        assert "unknown command" in result.error

    def test_ioctl_print(self, dev):
        with patch.object(dev, "print_text", return_value=5):
            result = dev.ioctl("PRINT", "hello")
            assert result.success
            assert result.value == 5

    def test_ioctl_println(self, dev):
        with patch.object(dev, "println", return_value=6):
            result = dev.ioctl("PRINTLN", "hello")
            assert result.success

    def test_ioctl_clear(self, dev):
        with patch.object(dev, "clear", return_value=True):
            result = dev.ioctl("CLEAR")
            assert result.success

    def test_ioctl_move(self, dev):
        with patch.object(dev, "move", return_value=True):
            result = dev.ioctl("MOVE", 5, 10)
            assert result.success

    def test_ioctl_color(self, dev):
        with patch.object(dev, "color", return_value=True):
            result = dev.ioctl("COLOR", "red", "blue")
            assert result.success

    def test_ioctl_style(self, dev):
        with patch.object(dev, "style", return_value=True):
            result = dev.ioctl("STYLE", True, False)
            assert result.success

    def test_ioctl_flush(self, dev):
        with patch.object(dev, "flush", return_value=True):
            result = dev.ioctl("FLUSH")
            assert result.success

    def test_ioctl_write(self, dev):
        with patch.object(dev, "write", return_value=3):
            result = dev.ioctl("WRITE", "abc")
            assert result.success
            assert result.value == 3

    def test_ioctl_info(self, dev):
        result = dev.ioctl("INFO")
        assert result.success
        assert result.value["type"] == "display"

    def test_ioctl_exception(self, dev):
        with patch.object(dev, "print_text", side_effect=RuntimeError("boom")):
            result = dev.ioctl("PRINT", "x")
            assert not result.success
            assert "ioctl error" in result.error


# ── call interface ────────────────────────────────────────────────────────


class TestDisplayDeviceCall:

    def test_call_success(self, dev):
        with patch.object(dev, "print_text", return_value=5):
            assert dev.call("PRINT", "hello") == 5

    def test_call_failure_raises(self, dev):
        with pytest.raises(Exception, match="unknown command"):
            dev.call("NONEXISTENT")


# ── Direct function calls ─────────────────────────────────────────────────


class TestDisplayDeviceFunctions:

    def test_print_text(self, dev, capsys):
        n = dev.print_text("hello")
        assert n == 5
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_println(self, dev, capsys):
        n = dev.println("hello")
        assert n == 6
        captured = capsys.readouterr()
        assert "hello\n" in captured.out

    def test_println_empty(self, dev, capsys):
        n = dev.println()
        assert n == 1

    def test_clear(self, dev, capsys):
        assert dev.clear() is True
        assert dev._cursor_x == 0
        assert dev._cursor_y == 0
        captured = capsys.readouterr()
        assert "\033[2J" in captured.out

    def test_move(self, dev, capsys):
        assert dev.move(5, 10) is True
        assert dev._cursor_x == 5
        assert dev._cursor_y == 10
        captured = capsys.readouterr()
        assert "\033[11;6H" in captured.out

    def test_color_fg(self, dev, capsys):
        assert dev.color(fg="red") is True
        captured = capsys.readouterr()
        assert "\033[31m" in captured.out

    def test_color_bg(self, dev, capsys):
        assert dev.color(bg="blue") is True
        captured = capsys.readouterr()
        # Code uses same codes dict for fg and bg; bg "blue" -> "34"
        assert "\033[34m" in captured.out

    def test_color_unknown(self, dev, capsys):
        assert dev.color(fg="neon") is True

    def test_style_bold(self, dev, capsys):
        assert dev.style(bold=True) is True
        captured = capsys.readouterr()
        assert "\033[1m" in captured.out

    def test_style_underline(self, dev, capsys):
        assert dev.style(underline=True) is True
        captured = capsys.readouterr()
        assert "\033[4m" in captured.out

    def test_flush(self, dev):
        assert dev.flush() is True

    def test_write(self, dev, capsys):
        n = dev.write("abc")
        assert n == 3
        captured = capsys.readouterr()
        assert "abc" in captured.out
