"""Tests for domains.shell.cmds — CmdModule and discover()."""

import pytest
from unittest.mock import patch, MagicMock
from domains.shell.cmds import CmdModule, discover, _MODULE_NAMES


class TestCmdModule:
    def test_loaded_initially_false(self):
        mod = CmdModule("nonexistent_module")
        assert mod.loaded is False

    def test_load_sets_loaded(self):
        mod = CmdModule("health")
        _ = mod.run  # triggers _load
        assert mod.loaded is True

    def test_help_property(self):
        mod = CmdModule("health")
        help_text = mod.help
        assert isinstance(help_text, str)

    def test_run_property(self):
        mod = CmdModule("health")
        fn = mod.run
        assert callable(fn)


class TestDiscover:
    def test_returns_dict(self):
        result = discover()
        assert isinstance(result, dict)

    def test_contains_mapped_commands(self):
        result = discover()
        # data_cmds maps to multiple commands
        assert "datasets" in result
        assert "checkpoints" in result
        assert "knowledge" in result
        assert "health" in result
        assert "models" in result
        assert "souls" in result

    def test_values_are_cmd_modules(self):
        result = discover()
        for key, val in result.items():
            assert isinstance(val, CmdModule)

    def test_module_names_keys_match(self):
        result = discover()
        for mod_name, cmd_names in _MODULE_NAMES.items():
            for cmd_name in cmd_names:
                assert cmd_name in result
