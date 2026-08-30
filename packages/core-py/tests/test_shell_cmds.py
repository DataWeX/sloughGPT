"""Tests for domains.shell.cmds — CmdModule, discover(), and pure utility functions."""

import pytest
from domains.shell.cmds import CmdModule, discover, _MODULE_NAMES


class TestCmdModule:
    def test_loaded_initially_false(self):
        mod = CmdModule("nonexistent_module")
        assert mod.loaded is False

    def test_load_sets_loaded(self):
        mod = CmdModule("health")
        _ = mod.run
        assert mod.loaded is True

    def test_help_property(self):
        mod = CmdModule("health")
        help_text = mod.help
        assert isinstance(help_text, str)

    def test_run_property(self):
        mod = CmdModule("health")
        fn = mod.run
        assert callable(fn)

    def test_load_idempotent(self):
        mod = CmdModule("health")
        _ = mod.run
        mod._load()
        assert mod.loaded is True

    def test_help_loaded_triggers_load(self):
        mod = CmdModule("health")
        assert mod.loaded is False
        _ = mod.help
        assert mod.loaded is True

    def test_loaded_stays_false_without_access(self):
        mod = CmdModule("health")
        assert mod.loaded is False
        assert mod.loaded is False

    def test_name_stored(self):
        mod = CmdModule("health")
        assert mod._name == "health"

    def test_mod_initially_none(self):
        mod = CmdModule("health")
        assert mod._mod is None

    def test_mod_set_after_load(self):
        mod = CmdModule("health")
        _ = mod.run
        assert mod._mod is not None

    def test_multiple_instances_independent(self):
        m1 = CmdModule("health")
        m2 = CmdModule("health")
        _ = m1.run
        assert m1.loaded is True
        assert m2.loaded is False

    def test_dashboard_module_loads(self):
        mod = CmdModule("dashboard")
        _ = mod.run
        assert mod.loaded is True

    def test_data_cmds_module_loads(self):
        mod = CmdModule("data_cmds")
        _ = mod.run
        assert mod.loaded is True

    def test_models_cmd_module_loads(self):
        mod = CmdModule("models_cmd")
        _ = mod.run
        assert mod.loaded is True

    def test_souls_cmd_module_loads(self):
        mod = CmdModule("souls_cmd")
        _ = mod.run
        assert mod.loaded is True

    def test_status_module_loads(self):
        mod = CmdModule("status")
        _ = mod.run
        assert mod.loaded is True

    def test_health_help_nonempty(self):
        mod = CmdModule("health")
        assert len(mod.help) > 0

    def test_dashboard_help_nonempty(self):
        mod = CmdModule("dashboard")
        assert len(mod.help) > 0

    def test_data_cmds_help_nonempty(self):
        mod = CmdModule("data_cmds")
        assert len(mod.help) > 0

    def test_models_cmd_help_nonempty(self):
        mod = CmdModule("models_cmd")
        assert len(mod.help) > 0

    def test_souls_cmd_help_nonempty(self):
        mod = CmdModule("souls_cmd")
        assert len(mod.help) > 0

    def test_status_help_nonempty(self):
        mod = CmdModule("status")
        assert len(mod.help) > 0


class TestDiscover:
    def test_returns_dict(self):
        result = discover()
        assert isinstance(result, dict)

    def test_contains_mapped_commands(self):
        result = discover()
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

    def test_dashboard_in_result(self):
        result = discover()
        assert "dashboard" in result

    def test_status_in_result(self):
        result = discover()
        assert "status" in result

    def test_all_module_names_covered(self):
        result = discover()
        all_cmd_names = []
        for cmd_names in _MODULE_NAMES.values():
            all_cmd_names.extend(cmd_names)
        for name in all_cmd_names:
            assert name in result

    def test_discover_unique_modules(self):
        result = discover()
        seen_modules = set()
        for cmd_name, mod in result.items():
            seen_modules.add(id(mod))
        assert len(seen_modules) >= 5

    def test_health_module_loaded_lazily(self):
        result = discover()
        assert result["health"].loaded is False

    def test_models_module_loaded_lazily(self):
        result = discover()
        assert result["models"].loaded is False

    def test_souls_module_loaded_lazily(self):
        result = discover()
        assert result["souls"].loaded is False

    def test_multiple_commands_same_module(self):
        result = discover()
        mod_datasets = result["datasets"]
        mod_checkpoints = result["checkpoints"]
        assert mod_datasets is mod_checkpoints

    def test_models_module_shared(self):
        result = discover()
        mod_models = result["models"]
        mod_unload = result["unload"]
        assert mod_models is mod_unload

    def test_souls_module_shared(self):
        result = discover()
        mod_souls = result["souls"]
        mod_switch = result["switch"]
        mod_whoami = result["whoami"]
        assert mod_souls is mod_switch
        assert mod_switch is mod_whoami

    def test_module_names_completeness(self):
        assert "health" in _MODULE_NAMES
        assert "dashboard" in _MODULE_NAMES
        assert "data_cmds" in _MODULE_NAMES
        assert "models_cmd" in _MODULE_NAMES
        assert "souls_cmd" in _MODULE_NAMES

    def test_data_cmds_names(self):
        expected = {"datasets", "checkpoints", "finetuned", "knowledge", "remember", "recall", "tokenizer"}
        assert set(_MODULE_NAMES["data_cmds"]) == expected

    def test_models_cmd_names(self):
        expected = {"models", "unload", "precision", "quantize", "dequantize"}
        assert set(_MODULE_NAMES["models_cmd"]) == expected

    def test_souls_cmd_names(self):
        expected = {"souls", "switch", "whoami"}
        assert set(_MODULE_NAMES["souls_cmd"]) == expected

    def test_health_single_name(self):
        assert _MODULE_NAMES["health"] == ["health"]

    def test_dashboard_single_name(self):
        assert _MODULE_NAMES["dashboard"] == ["dashboard"]


class TestDashboardUtils:
    def test_format_uptime_seconds(self):
        from domains.shell.cmds.dashboard import _format_uptime
        assert _format_uptime(30) == "30s"

    def test_format_uptime_minutes(self):
        from domains.shell.cmds.dashboard import _format_uptime
        assert _format_uptime(120) == "2m 00s"

    def test_format_uptime_hours(self):
        from domains.shell.cmds.dashboard import _format_uptime
        assert _format_uptime(3660) == "1h 01m"

    def test_format_uptime_zero(self):
        from domains.shell.cmds.dashboard import _format_uptime
        assert _format_uptime(0) == "0s"

    def test_format_ts(self):
        from domains.shell.cmds.dashboard import _format_ts
        result = _format_ts(0.0)
        assert isinstance(result, str)
        assert ":" in result

    def test_sparkline_empty(self):
        from domains.shell.cmds.dashboard import _sparkline
        assert _sparkline([]) == ""

    def test_sparkline_single_value(self):
        from domains.shell.cmds.dashboard import _sparkline
        result = _sparkline([5.0])
        assert len(result) >= 1

    def test_sparkline_multiple_values(self):
        from domains.shell.cmds.dashboard import _sparkline
        result = _sparkline([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(result) > 0

    def test_sparkline_with_width(self):
        from domains.shell.cmds.dashboard import _sparkline
        result = _sparkline([1.0, 2.0, 3.0], width=2)
        assert len(result) <= 3

    def test_status_icon_running(self):
        from domains.shell.cmds.dashboard import _status_icon
        icon = _status_icon("running")
        assert isinstance(icon, str)

    def test_status_icon_error(self):
        from domains.shell.cmds.dashboard import _status_icon
        icon = _status_icon("error")
        assert isinstance(icon, str)

    def test_status_icon_unknown(self):
        from domains.shell.cmds.dashboard import _status_icon
        icon = _status_icon("unknown_status")
        assert icon == "?"

    def test_progress_bar(self):
        from domains.shell.cmds.dashboard import _progress_bar
        bar = _progress_bar(50.0)
        assert isinstance(bar, str)
        assert len(bar) > 0

    def test_progress_bar_full(self):
        from domains.shell.cmds.dashboard import _progress_bar
        bar = _progress_bar(100.0)
        assert isinstance(bar, str)

    def test_progress_bar_empty(self):
        from domains.shell.cmds.dashboard import _progress_bar
        bar = _progress_bar(0.0)
        assert isinstance(bar, str)


class TestStatusUtils:
    def test_fmt_uptime_seconds(self):
        from domains.shell.cmds.status import _fmt_uptime
        assert _fmt_uptime(30) == "30s"

    def test_fmt_uptime_minutes(self):
        from domains.shell.cmds.status import _fmt_uptime
        assert _fmt_uptime(120) == "2m 0s"

    def test_fmt_uptime_hours(self):
        from domains.shell.cmds.status import _fmt_uptime
        result = _fmt_uptime(3660)
        assert "1h" in result
        assert "01m" in result

    def test_fmt_uptime_zero(self):
        from domains.shell.cmds.status import _fmt_uptime
        assert _fmt_uptime(0) == "0s"

    def test_fmt_uptime_exact_hour(self):
        from domains.shell.cmds.status import _fmt_uptime
        assert _fmt_uptime(3600) == "1h 00m"


class TestLinuxCmdUtils:
    def test_format_size_bytes(self):
        from domains.shell.cmds.linux import LinuxCommandsMixin
        result = LinuxCommandsMixin._format_size(100, human=False)
        assert "100" in result

    def test_format_size_human_bytes(self):
        from domains.shell.cmds.linux import LinuxCommandsMixin
        result = LinuxCommandsMixin._format_size(500, human=True)
        assert "B" in result

    def test_format_size_human_kb(self):
        from domains.shell.cmds.linux import LinuxCommandsMixin
        result = LinuxCommandsMixin._format_size(2048, human=True)
        assert "K" in result

    def test_format_size_human_mb(self):
        from domains.shell.cmds.linux import LinuxCommandsMixin
        result = LinuxCommandsMixin._format_size(2097152, human=True)
        assert "M" in result

    def test_fmt_error_file_not_found(self):
        from domains.shell.cmds.linux import LinuxCommandsMixin
        result = LinuxCommandsMixin._fmt_error(FileNotFoundError("test.txt"), "cat")
        assert "not found" in result.lower() or "cat" in result

    def test_fmt_error_permission(self):
        from domains.shell.cmds.linux import LinuxCommandsMixin
        result = LinuxCommandsMixin._fmt_error(PermissionError("denied"), "rm")
        assert "permission" in result.lower() or "rm" in result

    def test_fmt_error_generic(self):
        from domains.shell.cmds.linux import LinuxCommandsMixin
        result = LinuxCommandsMixin._fmt_error(ValueError("bad value"), "test")
        assert "ValueError" in result

    def test_fmt_error_no_cmd(self):
        from domains.shell.cmds.linux import LinuxCommandsMixin
        result = LinuxCommandsMixin._fmt_error(RuntimeError("oops"))
        assert "RuntimeError" in result


class TestErrorFormatting:
    def test_format_error_connection(self):
        from domains.shell.error import format_error
        result = format_error(ConnectionError("refused"), "health", color=False)
        assert "Connection failed" in result

    def test_format_error_timeout(self):
        from domains.shell.error import format_error
        result = format_error(TimeoutError("timed out"), "models", color=False)
        assert "timed out" in result.lower() or "Timeout" in result

    def test_format_error_permission(self):
        from domains.shell.error import format_error
        result = format_error(PermissionError("denied"), "rm", color=False)
        assert "Permission denied" in result

    def test_format_error_file_not_found(self):
        from domains.shell.error import format_error
        result = format_error(FileNotFoundError("nope"), "cat", color=False)
        assert "File not found" in result

    def test_format_error_generic(self):
        from domains.shell.error import format_error
        result = format_error(ValueError("oops"), "test", color=False)
        assert "ValueError" in result

    def test_format_error_no_cmd(self):
        from domains.shell.error import format_error
        result = format_error(RuntimeError("fail"), color=False)
        assert "RuntimeError" in result

    def test_format_error_with_cmd_prefix(self):
        from domains.shell.error import format_error
        result = format_error(RuntimeError("fail"), "mycmd", color=False)
        assert "[mycmd]" in result


class TestCmdModuleProtocol:
    def test_all_modules_have_run(self):
        result = discover()
        for name, mod in result.items():
            if name == "linux":
                continue
            fn = mod.run
            assert callable(fn), f"{name} module run is not callable"

    def test_all_modules_have_help(self):
        result = discover()
        for name, mod in result.items():
            if name == "linux":
                continue
            h = mod.help
            assert isinstance(h, str), f"{name} module help is not a string"

    def test_module_load_only_once(self):
        mod = CmdModule("health")
        mod._load()
        first_mod = mod._mod
        mod._load()
        assert mod._mod is first_mod

    def test_nonexistent_module_loaded_false(self):
        mod = CmdModule("does_not_exist_xyz")
        assert mod.loaded is False

    def test_mod_name_matches(self):
        for name in ["health", "dashboard", "data_cmds", "models_cmd", "souls_cmd", "status"]:
            mod = CmdModule(name)
            assert mod._name == name
