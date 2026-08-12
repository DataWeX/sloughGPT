"""Tests for the shared feature flag registry."""

import json

import pytest

from domains.shared.feature_flags import (
    FeatureFlag,
    FeatureFlags,
    FlagStatus,
    _register_defaults,
    is_enabled,
)


@pytest.fixture
def fresh_registry():
    original = dict(FeatureFlags._flags)
    original_config = FeatureFlags._config_path
    FeatureFlags._flags.clear()
    yield
    FeatureFlags._flags.clear()
    FeatureFlags._flags.update(original)
    FeatureFlags._config_path = original_config


@pytest.fixture
def default_registry(fresh_registry):
    _register_defaults()
    yield
    FeatureFlags._flags.clear()
    _register_defaults()


class TestFeatureFlag:
    def test_env_var_auto_generated(self):
        flag = FeatureFlag(name="my_feature", description="d")
        assert flag.env_var == "SLO_FF_MY_FEATURE"

    def test_custom_env_var_preserved(self):
        flag = FeatureFlag(name="my_feature", description="d", env_var="CUSTOM_VAR")
        assert flag.env_var == "CUSTOM_VAR"

    def test_enabled_status(self):
        flag = FeatureFlag(name="x", description="d", status=FlagStatus.ENABLED)
        assert flag.is_enabled is True

    def test_disabled_status(self):
        flag = FeatureFlag(name="x", description="d", status=FlagStatus.DISABLED)
        assert flag.is_enabled is False

    def test_experimental_counts_as_enabled(self):
        flag = FeatureFlag(name="x", description="d", status=FlagStatus.EXPERIMENTAL)
        assert flag.is_enabled is True

    def test_env_override_disables(self, monkeypatch):
        flag = FeatureFlag(name="x", description="d", status=FlagStatus.ENABLED)
        monkeypatch.setenv("SLO_FF_X", "0")
        assert flag.is_enabled is False

    def test_env_override_enables(self, monkeypatch):
        flag = FeatureFlag(name="x", description="d", status=FlagStatus.DISABLED)
        monkeypatch.setenv("SLO_FF_X", "1")
        assert flag.is_enabled is True

    def test_env_true_values(self, monkeypatch):
        for value in ("1", "true", "yes", "on", "TRUE", "Yes"):
            flag = FeatureFlag(name="x", description="d", status=FlagStatus.DISABLED)
            monkeypatch.setenv("SLO_FF_X", value)
            assert flag.is_enabled is True, value

    def test_env_unknown_value_disables(self, monkeypatch):
        flag = FeatureFlag(name="x", description="d", status=FlagStatus.ENABLED)
        monkeypatch.setenv("SLO_FF_X", "2")
        assert flag.is_enabled is False


class TestRegister:
    def test_register_new_flag(self, fresh_registry):
        flag = FeatureFlags.register("new_flag", description="desc")
        assert isinstance(flag, FeatureFlag)
        assert flag.name == "new_flag"
        assert flag.description == "desc"
        assert flag.status == FlagStatus.DISABLED
        assert "new_flag" in FeatureFlags._flags

    def test_register_returns_existing(self, fresh_registry):
        first = FeatureFlags.register("dup", description="original")
        second = FeatureFlags.register("dup", description="override")
        assert first is second
        assert second.description == "original"

    def test_register_status_default(self, fresh_registry):
        flag = FeatureFlags.register("st")
        assert flag.status == FlagStatus.DISABLED

    def test_register_custom_status(self, fresh_registry):
        flag = FeatureFlags.register("st", status=FlagStatus.ENABLED)
        assert flag.is_enabled is True


class TestIsEnabled:
    def test_enabled_flag(self, fresh_registry):
        FeatureFlags.register("known_flag", status=FlagStatus.ENABLED)
        assert FeatureFlags.is_enabled("known_flag") is True

    def test_unknown_flag_disabled(self, fresh_registry):
        assert FeatureFlags.is_enabled("no_such_flag") is False

    def test_unknown_flag_warns(self, fresh_registry, caplog):
        with caplog.at_level("WARNING"):
            FeatureFlags.is_enabled("no_such_flag")
        assert any("Unknown feature flag" in r.message for r in caplog.records)

    def test_module_shortcut(self, fresh_registry):
        FeatureFlags.register("shortcut", status=FlagStatus.ENABLED)
        assert is_enabled("shortcut") is True

    def test_env_override_at_registry_level(self, fresh_registry, monkeypatch):
        FeatureFlags.register("env_flag", status=FlagStatus.DISABLED)
        monkeypatch.setenv("SLO_FF_ENV_FLAG", "1")
        assert FeatureFlags.is_enabled("env_flag") is True


class TestSetStatus:
    def test_updates_status(self, fresh_registry):
        FeatureFlags.register("mut", status=FlagStatus.DISABLED)
        FeatureFlags.set_status("mut", FlagStatus.ENABLED)
        assert FeatureFlags.is_enabled("mut") is True

    def test_unknown_raises_keyerror(self, fresh_registry):
        with pytest.raises(KeyError):
            FeatureFlags.set_status("nope", FlagStatus.ENABLED)


class TestListAll:
    def test_structure(self, fresh_registry):
        FeatureFlags.register("listed", description="d", status=FlagStatus.EXPERIMENTAL)
        result = FeatureFlags.list_all()
        assert "listed" in result
        entry = result["listed"]
        assert entry["description"] == "d"
        assert entry["status"] == "experimental"
        assert entry["enabled"] is True
        assert entry["env_var"] == "SLO_FF_LISTED"

    def test_empty_registry(self, fresh_registry):
        assert FeatureFlags.list_all() == {}


class TestConfigPersistence:
    def test_load_missing_file_no_error(self, fresh_registry, tmp_path, caplog):
        missing = tmp_path / "missing.json"
        FeatureFlags.load_config(missing)
        assert FeatureFlags._config_path == missing

    def test_load_updates_statuses(self, fresh_registry, tmp_path):
        FeatureFlags.register("cfg_a", status=FlagStatus.DISABLED)
        FeatureFlags.register("cfg_b", status=FlagStatus.ENABLED)
        config = tmp_path / "flags.json"
        config.write_text(json.dumps({"cfg_a": "enabled", "cfg_b": "disabled"}))
        FeatureFlags.load_config(config)
        assert FeatureFlags.is_enabled("cfg_a") is True
        assert FeatureFlags.is_enabled("cfg_b") is False

    def test_load_ignores_unknown_flags(self, fresh_registry, tmp_path):
        FeatureFlags.register("cfg_c", status=FlagStatus.DISABLED)
        config = tmp_path / "flags.json"
        config.write_text(json.dumps({"not_a_flag": "enabled", "cfg_c": "enabled"}))
        FeatureFlags.load_config(config)
        assert FeatureFlags.is_enabled("cfg_c") is True
        assert "not_a_flag" not in FeatureFlags._flags

    def test_load_invalid_status_warns_and_skips(self, fresh_registry, tmp_path, caplog):
        FeatureFlags.register("cfg_d", status=FlagStatus.DISABLED)
        config = tmp_path / "flags.json"
        config.write_text(json.dumps({"cfg_d": "sometimes"}))
        with caplog.at_level("WARNING"):
            FeatureFlags.load_config(config)
        assert FeatureFlags.is_enabled("cfg_d") is False
        assert any("Invalid status" in r.message for r in caplog.records)

    def test_load_corrupt_json_no_crash(self, fresh_registry, tmp_path, caplog):
        config = tmp_path / "flags.json"
        config.write_text("{not valid json")
        with caplog.at_level("WARNING"):
            FeatureFlags.load_config(config)
        assert any("Failed to load" in r.message for r in caplog.records)

    def test_save_then_load_roundtrip(self, fresh_registry, tmp_path):
        FeatureFlags.register("rt_a", status=FlagStatus.ENABLED)
        FeatureFlags.register("rt_b", status=FlagStatus.EXPERIMENTAL)
        FeatureFlags.register("rt_c", status=FlagStatus.DISABLED)
        config = tmp_path / "flags.json"
        FeatureFlags.save_config(config)
        FeatureFlags.set_status("rt_a", FlagStatus.DISABLED)
        FeatureFlags.load_config(config)
        assert FeatureFlags.is_enabled("rt_a") is True
        assert FeatureFlags.is_enabled("rt_b") is True
        assert FeatureFlags.is_enabled("rt_c") is False

    def test_save_uses_configured_path(self, fresh_registry, tmp_path):
        FeatureFlags.register("path_flag", status=FlagStatus.ENABLED)
        config = tmp_path / "sub" / "flags.json"
        FeatureFlags._config_path = config
        FeatureFlags.save_config()
        assert config.exists()
        data = json.loads(config.read_text())
        assert data["path_flag"] == "enabled"

    def test_save_creates_parent_dirs(self, fresh_registry, tmp_path):
        FeatureFlags.register("deep", status=FlagStatus.DISABLED)
        config = tmp_path / "a" / "b" / "flags.json"
        FeatureFlags.save_config(config)
        assert config.exists()


class TestDefaults:
    def test_default_registration_count(self, fresh_registry):
        _register_defaults()
        assert len(FeatureFlags._flags) == 26

    def test_slonet_provider_enabled(self, fresh_registry):
        _register_defaults()
        assert FeatureFlags.is_enabled("slonet_provider") is True

    def test_native_c_inference_disabled(self, fresh_registry):
        _register_defaults()
        assert FeatureFlags.is_enabled("native_c_inference") is False

    def test_slonet_kernels_experimental_enabled(self, fresh_registry):
        _register_defaults()
        assert FeatureFlags.is_enabled("slonet_kernels") is True

    def test_defaults_survive_rerun(self, default_registry):
        _register_defaults()
        _register_defaults()
        assert len(FeatureFlags._flags) == 26
        assert FeatureFlags.is_enabled("soul_format") is True
        assert FeatureFlags.is_enabled("feature_flags") is True
