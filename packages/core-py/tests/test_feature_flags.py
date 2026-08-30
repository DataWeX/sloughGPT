"""Tests for domains.shared.feature_flags — FeatureFlags, FeatureFlag, FlagStatus."""

import os
import pytest
from domains.shared.feature_flags import (
    FlagStatus, FeatureFlag, FeatureFlags, is_enabled,
)


@pytest.fixture(autouse=True)
def _clean_flags():
    """Snapshot and restore FeatureFlags state around each test."""
    original_flags = dict(FeatureFlags._flags)
    original_config_path = FeatureFlags._config_path
    yield
    FeatureFlags._flags.clear()
    FeatureFlags._flags.update(original_flags)
    FeatureFlags._config_path = original_config_path


class TestFlagStatus:
    def test_all_members(self):
        assert len(FlagStatus) == 3

    def test_values(self):
        assert FlagStatus.ENABLED.value == "enabled"
        assert FlagStatus.DISABLED.value == "disabled"
        assert FlagStatus.EXPERIMENTAL.value == "experimental"

    def test_enum_from_value(self):
        assert FlagStatus("enabled") is FlagStatus.ENABLED
        assert FlagStatus("disabled") is FlagStatus.DISABLED
        assert FlagStatus("experimental") is FlagStatus.EXPERIMENTAL

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            FlagStatus("bogus")

    def test_is_str_subclass(self):
        assert isinstance(FlagStatus.ENABLED, str)


class TestFeatureFlag:
    def test_env_var_auto_generated(self):
        f = FeatureFlag(name="my_flag", description="test")
        assert f.env_var == "SLO_FF_MY_FLAG"

    def test_env_var_auto_generated_uppercase(self):
        f = FeatureFlag(name="lower_case", description="test")
        assert f.env_var == "SLO_FF_LOWER_CASE"

    def test_env_var_custom_preserved(self):
        f = FeatureFlag(name="x", description="t", env_var="CUSTOM_VAR")
        assert f.env_var == "CUSTOM_VAR"

    def test_env_var_custom_not_overridden(self):
        f = FeatureFlag(name="x", description="t", env_var="MY_OVERRIDE")
        os.environ["MY_OVERRIDE"] = "1"
        try:
            assert f.is_enabled is True
        finally:
            del os.environ["MY_OVERRIDE"]

    def test_env_var_override_enabled(self):
        f = FeatureFlag(name="test_flag", description="test", status=FlagStatus.DISABLED)
        os.environ[f.env_var] = "1"
        try:
            assert f.is_enabled is True
        finally:
            del os.environ[f.env_var]

    def test_env_var_override_disabled(self):
        f = FeatureFlag(name="test_flag2", description="test", status=FlagStatus.ENABLED)
        os.environ[f.env_var] = "0"
        try:
            assert f.is_enabled is False
        finally:
            del os.environ[f.env_var]

    def test_status_enabled(self):
        f = FeatureFlag(name="t", description="t", status=FlagStatus.ENABLED)
        assert f.is_enabled is True

    def test_status_experimental(self):
        f = FeatureFlag(name="t", description="t", status=FlagStatus.EXPERIMENTAL)
        assert f.is_enabled is True

    def test_status_disabled(self):
        f = FeatureFlag(name="t", description="t", status=FlagStatus.DISABLED)
        assert f.is_enabled is False

    def test_default_status_is_disabled(self):
        f = FeatureFlag(name="t", description="t")
        assert f.status is FlagStatus.DISABLED
        assert f.is_enabled is False

    def test_empty_env_var_string_not_set_in_environ(self):
        f = FeatureFlag(name="t", description="t")
        assert f.env_var == "SLO_FF_T"
        assert f.env_var not in os.environ


class TestFeatureFlags:
    def test_register(self):
        FeatureFlags.register("test_register_ff", description="test", status=FlagStatus.ENABLED)
        assert FeatureFlags.is_enabled("test_register_ff") is True

    def test_unknown_flag(self):
        assert FeatureFlags.is_enabled("nonexistent_flag_xyz") is False

    def test_set_status(self):
        FeatureFlags.register("test_set_status_ff", status=FlagStatus.DISABLED)
        FeatureFlags.set_status("test_set_status_ff", FlagStatus.ENABLED)
        assert FeatureFlags.is_enabled("test_set_status_ff") is True

    def test_set_status_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown feature flag"):
            FeatureFlags.set_status("unknown_flag_xyz", FlagStatus.ENABLED)

    def test_list_all(self):
        flags = FeatureFlags.list_all()
        assert isinstance(flags, dict)
        assert len(flags) > 0

    def test_register_existing_returns_same(self):
        f1 = FeatureFlags.register("test_idem_ff", description="a")
        f2 = FeatureFlags.register("test_idem_ff", description="b")
        assert f1 is f2
        # description should remain the original since register is idempotent
        assert f1.description == "a"

    def test_list_all_structure(self):
        FeatureFlags.register("struct_ff", description="s", status=FlagStatus.ENABLED)
        result = FeatureFlags.list_all()
        entry = result["struct_ff"]
        assert entry["description"] == "s"
        assert entry["status"] == "enabled"
        assert entry["enabled"] is True
        assert entry["env_var"] == "SLO_FF_STRUCT_FF"

    def test_list_all_disabled_flag(self):
        FeatureFlags.register("struct_disabled_ff", description="d", status=FlagStatus.DISABLED)
        result = FeatureFlags.list_all()
        assert result["struct_disabled_ff"]["enabled"] is False

    def test_set_status_transitions(self):
        FeatureFlags.register("trans_ff", status=FlagStatus.DISABLED)
        FeatureFlags.set_status("trans_ff", FlagStatus.ENABLED)
        assert FeatureFlags.is_enabled("trans_ff") is True
        FeatureFlags.set_status("trans_ff", FlagStatus.EXPERIMENTAL)
        assert FeatureFlags.is_enabled("trans_ff") is True
        FeatureFlags.set_status("trans_ff", FlagStatus.DISABLED)
        assert FeatureFlags.is_enabled("trans_ff") is False


class TestFeatureFlagEnvVariants:
    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "yes", "Yes", "on", "On", "1"])
    def test_env_var_truthy_values(self, value):
        """Env var with various truthy strings should enable the flag."""
        f = FeatureFlag(name="truthy_test", description="test", status=FlagStatus.DISABLED)
        os.environ[f.env_var] = value
        try:
            assert f.is_enabled is True
        finally:
            del os.environ[f.env_var]

    @pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", "anything", "", "  "])
    def test_env_var_falsy_values(self, value):
        """Env var with non-truthy strings should disable the flag."""
        f = FeatureFlag(name="falsy_test", description="test", status=FlagStatus.ENABLED)
        os.environ[f.env_var] = value
        try:
            assert f.is_enabled is False
        finally:
            del os.environ[f.env_var]

    def test_env_var_absent_uses_status(self):
        """When env var is absent, the flag's status determines enabled."""
        f = FeatureFlag(name="absent_test", description="test", status=FlagStatus.EXPERIMENTAL)
        assert f.env_var not in os.environ
        assert f.is_enabled is True  # EXPERIMENTAL counts as enabled

    def test_env_var_whitespace_only_is_falsy(self):
        f = FeatureFlag(name="ws_test", description="test", status=FlagStatus.ENABLED)
        os.environ[f.env_var] = "   "
        try:
            assert f.is_enabled is False
        finally:
            del os.environ[f.env_var]

    def test_env_var_case_insensitive(self):
        f = FeatureFlag(name="case_test", description="test", status=FlagStatus.DISABLED)
        os.environ[f.env_var] = "YES"
        try:
            assert f.is_enabled is True
        finally:
            del os.environ[f.env_var]

    def test_custom_env_var_independence(self):
        """Two flags with different env vars are independent."""
        f1 = FeatureFlag(name="a", description="a", status=FlagStatus.DISABLED, env_var="EVAR_A")
        f2 = FeatureFlag(name="b", description="b", status=FlagStatus.DISABLED, env_var="EVAR_B")
        os.environ["EVAR_A"] = "1"
        try:
            assert f1.is_enabled is True
            assert f2.is_enabled is False
        finally:
            del os.environ["EVAR_A"]


class TestFeatureFlagsConfigIO:
    def test_save_and_load_round_trip(self, tmp_path):
        """save_config then load_config preserves flag statuses."""
        FeatureFlags.register("round_trip_ff", status=FlagStatus.ENABLED)
        path = tmp_path / "flags.json"
        FeatureFlags.save_config(path)
        # Reset the flag
        FeatureFlags._flags["round_trip_ff"].status = FlagStatus.DISABLED
        FeatureFlags.load_config(path)
        assert FeatureFlags._flags["round_trip_ff"].status == FlagStatus.ENABLED

    def test_load_config_nonexistent_path(self, tmp_path):
        """Loading from a non-existent path is a no-op, doesn't raise."""
        FeatureFlags.load_config(tmp_path / "nonexistent.json")  # should not raise

    def test_load_config_invalid_json(self, tmp_path):
        """Loading invalid JSON is a no-op, doesn't raise."""
        path = tmp_path / "bad.json"
        path.write_text("not json {{{")
        FeatureFlags.register("invalid_json_ff", status=FlagStatus.ENABLED)
        FeatureFlags.load_config(path)  # should not raise
        # Flag should keep its original status
        assert FeatureFlags._flags["invalid_json_ff"].status == FlagStatus.ENABLED

    def test_load_config_invalid_status_value(self, tmp_path):
        """Loading a config with an invalid status value skips that flag."""
        FeatureFlags.register("invalid_status_ff", status=FlagStatus.ENABLED)
        path = tmp_path / "bad_status.json"
        path.write_text('{"invalid_status_ff": "bogus"}')
        FeatureFlags.load_config(path)  # should not raise
        # Flag should keep its original status
        assert FeatureFlags._flags["invalid_status_ff"].status == FlagStatus.ENABLED

    def test_save_config_creates_parent_dirs(self, tmp_path):
        """save_config creates parent directories if they don't exist."""
        FeatureFlags.register("mkdir_ff", status=FlagStatus.ENABLED)
        path = tmp_path / "deep" / "nested" / "flags.json"
        FeatureFlags.save_config(path)
        assert path.exists()

    def test_save_and_load_multiple_flags(self, tmp_path):
        """Multiple flags are preserved through save/load."""
        FeatureFlags.register("multi_a", status=FlagStatus.ENABLED)
        FeatureFlags.register("multi_b", status=FlagStatus.EXPERIMENTAL)
        FeatureFlags.register("multi_c", status=FlagStatus.DISABLED)
        path = tmp_path / "multi.json"
        FeatureFlags.save_config(path)
        FeatureFlags._flags["multi_a"].status = FlagStatus.DISABLED
        FeatureFlags._flags["multi_b"].status = FlagStatus.DISABLED
        FeatureFlags._flags["multi_c"].status = FlagStatus.ENABLED
        FeatureFlags.load_config(path)
        assert FeatureFlags._flags["multi_a"].status == FlagStatus.ENABLED
        assert FeatureFlags._flags["multi_b"].status == FlagStatus.EXPERIMENTAL
        assert FeatureFlags._flags["multi_c"].status == FlagStatus.DISABLED

    def test_load_config_ignores_unregistered_flags(self, tmp_path):
        """Flags in config that aren't registered are silently ignored."""
        path = tmp_path / "extra.json"
        path.write_text('{"ghost_flag": "enabled"}')
        FeatureFlags.load_config(path)
        assert "ghost_flag" not in FeatureFlags._flags

    def test_save_config_does_not_set_config_path(self, tmp_path):
        """save_config does not set _config_path (only load_config does)."""
        FeatureFlags._config_path = None
        path = tmp_path / "flags.json"
        FeatureFlags.save_config(path)
        assert FeatureFlags._config_path is None

    def test_load_config_sets_config_path(self, tmp_path):
        """load_config stores the path in _config_path."""
        path = tmp_path / "flags.json"
        FeatureFlags.load_config(path)
        assert FeatureFlags._config_path == path


class TestRegisterDefaults:
    def test_known_flags_registered(self):
        """_register_defaults registers all known feature flags."""
        known = [
            "slonet_provider", "native_c_inference", "cloud_vector_store",
            "soul_format", "soul_manager", "slonet_kernels", "multimodal",
            "cross_attention", "kv_cache", "session_kv_cache",
            "model_server", "model_registry", "process_isolation",
            "on_device_training", "quantization", "hf_finetune",
            "vlm", "dpo", "context_managers", "knowledge_memory",
            "semantic_cache", "llm_nlp", "feature_flags",
            "slonet_provider_tests", "slonet_provider_wave_i", "slonet_wave_f",
        ]
        for name in known:
            assert name in FeatureFlags._flags, f"Missing known flag: {name}"

    def test_known_flags_default_status(self):
        """Verify default statuses of a few known flags."""
        assert FeatureFlags._flags["slonet_provider"].status is FlagStatus.ENABLED
        assert FeatureFlags._flags["native_c_inference"].status is FlagStatus.DISABLED
        assert FeatureFlags._flags["slonet_kernels"].status is FlagStatus.EXPERIMENTAL
        assert FeatureFlags._flags["feature_flags"].status is FlagStatus.ENABLED
        assert FeatureFlags._flags["hf_finetune"].status is FlagStatus.DISABLED

    def test_is_enabled_convenience_function(self):
        """Module-level is_enabled() delegates to FeatureFlags.is_enabled()."""
        assert is_enabled("slonet_provider") is True
        assert is_enabled("nonexistent_xyz") is False

    def test_idempotent_defaults(self):
        """Calling _register_defaults twice doesn't duplicate flags."""
        from domains.shared.feature_flags import _register_defaults
        count_before = len(FeatureFlags._flags)
        _register_defaults()
        count_after = len(FeatureFlags._flags)
        assert count_before == count_after
