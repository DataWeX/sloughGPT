"""Tests for domains.shared.feature_flags — FeatureFlags, FeatureFlag, FlagStatus."""

import os
import pytest
from domains.shared.feature_flags import (
    FlagStatus, FeatureFlag, FeatureFlags, is_enabled,
)


class TestFlagStatus:
    def test_all_members(self):
        assert len(FlagStatus) == 3

    def test_values(self):
        assert FlagStatus.ENABLED.value == "enabled"
        assert FlagStatus.DISABLED.value == "disabled"
        assert FlagStatus.EXPERIMENTAL.value == "experimental"


class TestFeatureFlag:
    def test_env_var_auto_generated(self):
        f = FeatureFlag(name="my_flag", description="test")
        assert f.env_var == "SLO_FF_MY_FLAG"

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
        try:
            FeatureFlags.set_status("unknown_flag_xyz", FlagStatus.ENABLED)
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_list_all(self):
        flags = FeatureFlags.list_all()
        assert isinstance(flags, dict)
        assert len(flags) > 0

    def test_register_existing_returns_same(self):
        f1 = FeatureFlags.register("test_idem_ff", description="a")
        f2 = FeatureFlags.register("test_idem_ff", description="b")
        assert f1 is f2


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

    @pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", "anything"])
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
