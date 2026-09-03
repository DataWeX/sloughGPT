"""Tests for device decorators — adds ioctl to any class."""
from __future__ import annotations

import pytest

from domains.shell.device_decorators import add_ioctl_command, with_ioctl


class TestWithIoctl:
    def test_adds_ioctl_method(self):
        @with_ioctl({"MATMUL": "matmul"})
        class Device:
            def matmul(self, a, b):
                return a + b

        dev = Device()
        assert hasattr(dev, "ioctl")

    def test_ioctl_dispatches(self):
        @with_ioctl({"MATMUL": "matmul", "RELU": "relu"})
        class Device:
            def matmul(self, a, b):
                return a + b

            def relu(self, x):
                return max(0, x)

        dev = Device()
        assert dev.ioctl("MATMUL", 1, 2) == 3
        assert dev.ioctl("RELU", -5) == 0
        assert dev.ioctl("RELU", 5) == 5

    def test_unknown_command_raises(self):
        @with_ioctl({})
        class Device:
            pass

        dev = Device()
        with pytest.raises(ValueError, match="unknown command"):
            dev.ioctl("BAD")

    def test_list_commands(self):
        @with_ioctl({"MATMUL": "matmul", "RELU": "relu"})
        class Device:
            def matmul(self, a, b):
                return a + b

            def relu(self, x):
                return x

        dev = Device()
        cmds = dev.list_commands()
        assert cmds == ["MATMUL", "RELU"]

    def test_list_commands_sorted(self):
        @with_ioctl({"Z": "z", "A": "a", "M": "m"})
        class Device:
            def z(self): pass
            def a(self): pass
            def m(self): pass

        dev = Device()
        assert dev.list_commands() == ["A", "M", "Z"]


class TestAddIoctlCommand:
    def test_adds_command(self):
        class Device:
            def matmul(self, a, b):
                return a * b

        add_ioctl_command(Device, "MATMUL", "matmul")
        dev = Device()
        assert dev.ioctl("MATMUL", 3, 4) == 12

    def test_adds_multiple_commands(self):
        class Device:
            def matmul(self, a, b):
                return a + b

            def relu(self, x):
                return max(0, x)

        add_ioctl_command(Device, "MATMUL", "matmul")
        add_ioctl_command(Device, "RELU", "relu")
        dev = Device()
        assert dev.ioctl("MATMUL", 1, 2) == 3
        assert dev.ioctl("RELU", -1) == 0

    def test_list_commands(self):
        class Device:
            def a(self): pass
            def b(self): pass

        add_ioctl_command(Device, "CMD_A", "a")
        add_ioctl_command(Device, "CMD_B", "b")
        dev = Device()
        assert sorted(dev.list_commands()) == ["CMD_A", "CMD_B"]

    def test_unknown_command_raises(self):
        class Device:
            pass

        add_ioctl_command(Device, "CMD", "method")
        dev = Device()
        with pytest.raises(ValueError, match="unknown command"):
            dev.ioctl("BAD")

    def test_method_not_found_raises(self):
        class Device:
            pass

        add_ioctl_command(Device, "CMD", "nonexistent")
        dev = Device()
        with pytest.raises(ValueError, match="method not found"):
            dev.ioctl("CMD")
