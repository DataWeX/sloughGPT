"""Tests for IoctlCommand enum — type-safe command identifiers."""
from __future__ import annotations

from domains.shell.ioctl import IoctlCommand


class TestIoctlCommand:
    def test_common_commands(self):
        assert IoctlCommand.INFO.value == "INFO"
        assert IoctlCommand.LIST_COMMANDS.value == "LIST_COMMANDS"

    def test_tensor_commands(self):
        assert IoctlCommand.MATMUL.value == "MATMUL"
        assert IoctlCommand.ADD.value == "ADD"
        assert IoctlCommand.MUL.value == "MUL"

    def test_activation_commands(self):
        assert IoctlCommand.RELU.value == "RELU"
        assert IoctlCommand.SIGMOID.value == "SIGMOID"
        assert IoctlCommand.TANH.value == "TANH"
        assert IoctlCommand.GELU.value == "GELU"

    def test_shape_commands(self):
        assert IoctlCommand.RESHAPE.value == "RESHAPE"
        assert IoctlCommand.TRANSPOSE.value == "TRANSPOSE"
        assert IoctlCommand.FLATTEN.value == "FLATTEN"

    def test_all_unique_values(self):
        values = [cmd.value for cmd in IoctlCommand]
        assert len(values) == len(set(values))

    def test_enum_members_count(self):
        assert len(IoctlCommand) >= 40
