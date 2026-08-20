"""Tests for domains.shared.feature_flags — FeatureFlags, FeatureFlag, FlagStatus."""

import os
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
