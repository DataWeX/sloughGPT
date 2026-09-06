"""Tests for shell.input_device — InputDevice input operations and ioctl."""

from __future__ import annotations

from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from domains.shell.input_device import InputDevice
from domains.shell.kernel_syscall import SyscallResult


@pytest.fixture
def dev():
    return InputDevice(name="test-input")


# ── Basics ────────────────────────────────────────────────────────────────


class TestInputDeviceBasics:

    def test_name(self, dev):
        assert dev.name == "test-input"

    def test_default_name(self):
        d = InputDevice()
        assert d.name == "input"

    def test_info(self, dev):
        info = dev.info()
        assert info["name"] == "test-input"
        assert info["type"] == "input"

    def test_list_commands(self, dev):
        cmds = dev.list_commands()
        assert "READLINE" in cmds
        assert "READCHAR" in cmds
        assert "READKEY" in cmds
        assert "READLINE_ECHO" in cmds
        assert "POLL" in cmds
        assert "FLUSH" in cmds
        assert "INFO" in cmds
        assert cmds == sorted(cmds)


# ── ioctl ─────────────────────────────────────────────────────────────────


class TestInputDeviceIoctl:

    def test_ioctl_unknown_command(self, dev):
        result = dev.ioctl("NONEXISTENT")
        assert isinstance(result, SyscallResult)
        assert not result.success
        assert "unknown command" in result.error

    def test_ioctl_readline(self, dev):
        with patch.object(dev, "readline", return_value="hello"):
            result = dev.ioctl("READLINE")
            assert result.success
            assert result.value == "hello"

    def test_ioctl_readchar(self, dev):
        with patch.object(dev, "readchar", return_value="a"):
            result = dev.ioctl("READCHAR")
            assert result.success
            assert result.value == "a"

    def test_ioctl_readkey(self, dev):
        with patch.object(dev, "readkey", return_value="\x1b[A"):
            result = dev.ioctl("READKEY")
            assert result.success
            assert result.value == "\x1b[A"

    def test_ioctl_readline_echo(self, dev):
        with patch.object(dev, "readline_echo", return_value="password"):
            result = dev.ioctl("READLINE_ECHO")
            assert result.success
            assert result.value == "password"

    def test_ioctl_poll(self, dev):
        with patch.object(dev, "poll", return_value=True):
            result = dev.ioctl("POLL", 1.0)
            assert result.success
            assert result.value is True

    def test_ioctl_flush(self, dev):
        with patch.object(dev, "flush", return_value=True):
            result = dev.ioctl("FLUSH")
            assert result.success

    def test_ioctl_info(self, dev):
        result = dev.ioctl("INFO")
        assert result.success
        assert result.value["type"] == "input"

    def test_ioctl_exception(self, dev):
        with patch.object(dev, "readline", side_effect=RuntimeError("boom")):
            result = dev.ioctl("READLINE")
            assert not result.success
            assert "ioctl error" in result.error


# ── call interface ────────────────────────────────────────────────────────


class TestInputDeviceCall:

    def test_call_success(self, dev):
        with patch.object(dev, "readline", return_value="hello"):
            assert dev.call("READLINE") == "hello"

    def test_call_failure_raises(self, dev):
        with pytest.raises(Exception, match="unknown command"):
            dev.call("NONEXISTENT")


# ── Direct function calls ─────────────────────────────────────────────────


class TestInputDeviceFunctions:

    def test_readline(self, dev):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.readline.return_value = "hello\n"
            assert dev.readline() == "hello"

    def test_readchar(self, dev):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = "a"
            assert dev.readchar() == "a"

    def test_readkey_simple(self, dev):
        with patch("sys.stdin") as mock_stdin, \
             patch("termios.tcgetattr", return_value=[]), \
             patch("termios.tcsetattr"), \
             patch("tty.setraw"):
            mock_stdin.read.side_effect = ["a"]
            assert dev.readkey() == "a"

    def test_readkey_escape(self, dev):
        with patch("sys.stdin") as mock_stdin, \
             patch("termios.tcgetattr", return_value=[]), \
             patch("termios.tcsetattr"), \
             patch("tty.setraw"):
            mock_stdin.read.side_effect = ["\x1b", "[", "A"]
            assert dev.readkey() == "\x1b[A"

    def test_readkey_escape_unknown(self, dev):
        with patch("sys.stdin") as mock_stdin, \
             patch("termios.tcgetattr", return_value=[]), \
             patch("termios.tcsetattr"), \
             patch("tty.setraw"):
            mock_stdin.read.side_effect = ["\x1b", "x"]
            assert dev.readkey() == "\x1bx"

    def test_readline_echo_normal(self, dev):
        with patch("sys.stdin") as mock_stdin, \
             patch("builtins.print"):
            mock_stdin.read.side_effect = ["a", "b", "\n"]
            assert dev.readline_echo() == "ab"

    def test_readline_echo_backspace(self, dev):
        with patch("sys.stdin") as mock_stdin, \
             patch("builtins.print"), \
             patch("sys.stdout"):
            mock_stdin.read.side_effect = ["a", "\x7f", "\n"]
            assert dev.readline_echo() == ""

    def test_readline_echo_ctrl_c(self, dev):
        with patch("sys.stdin") as mock_stdin, \
             patch("builtins.print"):
            mock_stdin.read.side_effect = ["\x03"]
            with pytest.raises(KeyboardInterrupt):
                dev.readline_echo()

    def test_poll_true(self, dev):
        with patch("select.select", return_value=([True], [], [])):
            assert dev.poll(0.0) is True

    def test_poll_false(self, dev):
        with patch("select.select", return_value=([], [], [])):
            assert dev.poll(0.0) is False

    def test_flush(self, dev):
        with patch("termios.tcflush"):
            assert dev.flush() is True
