"""Tests for shell.device_decorators — ioctl decorator pattern."""

from __future__ import annotations

import pytest

from domains.shell.device_decorators import with_ioctl, add_ioctl_command


# ── with_ioctl decorator ──────────────────────────────────────────────────


class TestWithIoctl:

    def test_adds_ioctl_method(self):
        @with_ioctl({"MATMUL": "matmul"})
        class Dev:
            def matmul(self, a, b):
                return a + b

        dev = Dev()
        assert hasattr(dev, "ioctl")

    def test_adds_list_commands(self):
        @with_ioctl({"MATMUL": "matmul", "RELU": "relu"})
        class Dev:
            def matmul(self, a, b):
                return a + b
            def relu(self, a):
                return max(0, a)

        dev = Dev()
        assert dev.list_commands() == ["MATMUL", "RELU"]

    def test_ioctl_dispatches_to_method(self):
        @with_ioctl({"MATMUL": "matmul"})
        class Dev:
            def matmul(self, a, b):
                return a * b

        dev = Dev()
        assert dev.ioctl("MATMUL", 3, 4) == 12

    def test_ioctl_with_no_args(self):
        @with_ioctl({"INFO": "info"})
        class Dev:
            def info(self):
                return "device-v1"

        dev = Dev()
        assert dev.ioctl("INFO") == "device-v1"

    def test_ioctl_unknown_command_raises(self):
        @with_ioctl({"MATMUL": "matmul"})
        class Dev:
            def matmul(self, a, b):
                return a + b

        dev = Dev()
        with pytest.raises(ValueError, match="unknown command"):
            dev.ioctl("UNKNOWN")

    def test_ioctl_missing_method_raises(self):
        @with_ioctl({"MATMUL": "nonexistent"})
        class Dev:
            pass

        dev = Dev()
        with pytest.raises(ValueError, match="method not found"):
            dev.ioctl("MATMUL")

    def test_list_commands_sorted(self):
        @with_ioctl({"Z": "z", "A": "a", "M": "m"})
        class Dev:
            def z(self): pass
            def a(self): pass
            def m(self): pass

        dev = Dev()
        assert dev.list_commands() == ["A", "M", "Z"]


# ── add_ioctl_command ─────────────────────────────────────────────────────


class TestAddIoctlCommand:

    def test_adds_command_to_existing_class(self):
        class Dev:
            def matmul(self, a, b):
                return a + b

        add_ioctl_command(Dev, "MATMUL", "matmul")
        dev = Dev()
        assert dev.ioctl("MATMUL", 2, 3) == 5

    def test_adds_multiple_commands(self):
        class Dev:
            def matmul(self, a, b):
                return a + b
            def relu(self, a):
                return max(0, a)

        add_ioctl_command(Dev, "MATMUL", "matmul")
        add_ioctl_command(Dev, "RELU", "relu")
        dev = Dev()
        assert dev.ioctl("MATMUL", 1, 2) == 3
        assert dev.ioctl("RELU", -1) == 0

    def test_list_commands_returns_sorted(self):
        class Dev:
            def matmul(self, a, b): return a
            def relu(self, a): return a

        add_ioctl_command(Dev, "RELU", "relu")
        add_ioctl_command(Dev, "MATMUL", "matmul")
        dev = Dev()
        assert dev.list_commands() == ["MATMUL", "RELU"]

    def test_unknown_command_raises(self):
        class Dev:
            pass

        add_ioctl_command(Dev, "MATMUL", "matmul")
        dev = Dev()
        with pytest.raises(ValueError, match="unknown command"):
            dev.ioctl("UNKNOWN")

    def test_missing_method_raises(self):
        class Dev:
            pass

        add_ioctl_command(Dev, "MATMUL", "nonexistent")
        dev = Dev()
        with pytest.raises(ValueError, match="method not found"):
            dev.ioctl("MATMUL")

    def test_no_commands_registered_no_ioctl(self):
        class Dev:
            pass

        dev = Dev()
        assert not hasattr(dev, "ioctl")

    def test_decorator_returns_class(self):
        @with_ioctl({"A": "a"})
        class Dev:
            def a(self): return 1

        assert Dev.__name__ == "Dev"
        dev = Dev()
        assert dev.ioctl("A") == 1
