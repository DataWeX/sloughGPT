"""Tests for domains.memory.memory_config — MemoryConfig singleton and env parsing."""

import os
import threading
import pytest
from domains.memory.memory_config import MemoryConfig


class TestMemoryConfigDefaults:
    def test_default_enabled(self):
        cfg = MemoryConfig()
        assert cfg.enabled is True

    def test_default_min_chars(self):
        cfg = MemoryConfig()
        assert cfg.min_chars == 80

    def test_default_max_facts(self):
        cfg = MemoryConfig()
        assert cfg.max_facts == 5

    def test_default_store_path(self):
        cfg = MemoryConfig()
        assert cfg.store_path == "data/memory"

    def test_default_sync_remember(self):
        cfg = MemoryConfig()
        assert cfg.sync_remember is False

    def test_default_consolidation_threshold(self):
        cfg = MemoryConfig()
        assert cfg.consolidation_threshold == 0.80

    def test_default_maintenance_interval(self):
        cfg = MemoryConfig()
        assert cfg.maintenance_interval_minutes == 60

    def test_default_archive_retention(self):
        cfg = MemoryConfig()
        assert cfg.archive_retention_days == 30


class TestMemoryConfigOverrides:
    def test_override_enabled(self):
        cfg = MemoryConfig(enabled=False)
        assert cfg.enabled is False

    def test_override_min_chars(self):
        cfg = MemoryConfig(min_chars=200)
        assert cfg.min_chars == 200

    def test_override_max_facts(self):
        cfg = MemoryConfig(max_facts=10)
        assert cfg.max_facts == 10

    def test_override_store_path(self):
        cfg = MemoryConfig(store_path="/tmp/mem")
        assert cfg.store_path == "/tmp/mem"

    def test_override_consolidation_threshold(self):
        cfg = MemoryConfig(consolidation_threshold=0.95)
        assert cfg.consolidation_threshold == 0.95


class TestFromBool:
    def test_true_values(self):
        for val in ("1", "true", "yes", "on", " True ", "YES"):
            assert MemoryConfig._from_bool("NONEXISTENT_VAR", False) is False
            os.environ["_TEST_BOOL"] = val
            assert MemoryConfig._from_bool("_TEST_BOOL", False) is True
            del os.environ["_TEST_BOOL"]

    def test_false_values(self):
        for val in ("0", "false", "no", "off", "nope", ""):
            os.environ["_TEST_BOOL"] = val
            assert MemoryConfig._from_bool("_TEST_BOOL", True) is False
            del os.environ["_TEST_BOOL"]

    def test_unset_returns_default(self):
        assert MemoryConfig._from_bool("_UNSET_VAR_XYZ_", True) is True
        assert MemoryConfig._from_bool("_UNSET_VAR_XYZ_", False) is False


class TestEnvOverrides:
    def test_env_min_chars(self):
        os.environ["SLO_MEMORY_MIN_CHARS"] = "200"
        try:
            cfg = MemoryConfig()
            assert cfg.min_chars == 200
        finally:
            del os.environ["SLO_MEMORY_MIN_CHARS"]

    def test_env_max_facts(self):
        os.environ["SLO_MEMORY_MAX_FACTS"] = "10"
        try:
            cfg = MemoryConfig()
            assert cfg.max_facts == 10
        finally:
            del os.environ["SLO_MEMORY_MAX_FACTS"]


class TestSingleton:
    def test_get_returns_same_instance(self):
        a = MemoryConfig.get()
        b = MemoryConfig.get()
        assert a is b


class TestSetEnabled:
    def test_set_enabled(self):
        cfg = MemoryConfig(enabled=False)
        cfg.set_enabled(True)
        assert cfg.enabled is True

    def test_set_disabled(self):
        cfg = MemoryConfig(enabled=True)
        cfg.set_enabled(False)
        assert cfg.enabled is False


class TestSetArchiveRetention:
    def test_set_retention(self):
        cfg = MemoryConfig()
        cfg.set_archive_retention_days(60)
        assert cfg.archive_retention_days == 60

    def test_set_retention_negative_clamps(self):
        cfg = MemoryConfig()
        cfg.set_archive_retention_days(-5)
        assert cfg.archive_retention_days == 0.0


class TestSnapshot:
    def test_snapshot_keys(self):
        cfg = MemoryConfig()
        snap = cfg.snapshot()
        assert "enabled" in snap
        assert "min_chars" in snap
        assert "max_facts" in snap
        assert "store_path" in snap
        assert "sync_remember" in snap
        assert "consolidation_threshold" in snap
        assert "archive_retention_days" in snap

    def test_snapshot_values_match(self):
        cfg = MemoryConfig(min_chars=123)
        snap = cfg.snapshot()
        assert snap["min_chars"] == 123

    def test_snapshot_returns_dict(self):
        cfg = MemoryConfig()
        assert isinstance(cfg.snapshot(), dict)
