"""Tests for domains.memory.memory_config — MemoryConfig."""

import os
import pytest
from domains.memory.memory_config import MemoryConfig


class TestMemoryConfigDefaults:
    def test_enabled(self):
        mc = MemoryConfig()
        assert mc.enabled is True

    def test_min_chars(self):
        mc = MemoryConfig()
        assert mc.min_chars == 80

    def test_max_facts(self):
        mc = MemoryConfig()
        assert mc.max_facts == 5

    def test_store_path(self):
        mc = MemoryConfig()
        assert mc.store_path == "data/memory"

    def test_sync_remember(self):
        mc = MemoryConfig()
        assert mc.sync_remember is False

    def test_consolidation_threshold(self):
        mc = MemoryConfig()
        assert mc.consolidation_threshold == 0.80

    def test_maintenance_interval_minutes(self):
        mc = MemoryConfig()
        assert mc.maintenance_interval_minutes == 60

    def test_archive_retention_days(self):
        mc = MemoryConfig()
        assert mc.archive_retention_days == 30


class TestMemoryConfigCustom:
    def test_enabled_false(self):
        mc = MemoryConfig(enabled=False)
        assert mc.enabled is False

    def test_min_chars_custom(self):
        mc = MemoryConfig(min_chars=100)
        assert mc.min_chars == 100

    def test_max_facts_custom(self):
        mc = MemoryConfig(max_facts=10)
        assert mc.max_facts == 10

    def test_store_path_custom(self):
        mc = MemoryConfig(store_path="/tmp/mem")
        assert mc.store_path == "/tmp/mem"

    def test_sync_remember_true(self):
        mc = MemoryConfig(sync_remember=True)
        assert mc.sync_remember is True

    def test_consolidation_threshold_custom(self):
        mc = MemoryConfig(consolidation_threshold=0.95)
        assert mc.consolidation_threshold == 0.95

    def test_maintenance_interval_custom(self):
        mc = MemoryConfig(maintenance_interval_minutes=30)
        assert mc.maintenance_interval_minutes == 30

    def test_archive_retention_custom(self):
        mc = MemoryConfig(archive_retention_days=60)
        assert mc.archive_retention_days == 60

    def test_all_custom(self):
        mc = MemoryConfig(
            enabled=False,
            min_chars=200,
            max_facts=20,
            store_path="/data",
            sync_remember=True,
            consolidation_threshold=0.7,
            maintenance_interval_minutes=15,
            archive_retention_days=7,
        )
        assert mc.enabled is False
        assert mc.min_chars == 200
        assert mc.max_facts == 20
        assert mc.store_path == "/data"
        assert mc.sync_remember is True
        assert mc.consolidation_threshold == 0.7
        assert mc.maintenance_interval_minutes == 15
        assert mc.archive_retention_days == 7


class TestMemoryConfigSingleton:
    def test_get_returns_same_instance(self):
        mc1 = MemoryConfig.get()
        mc2 = MemoryConfig.get()
        assert mc1 is mc2

    def test_singleton_is_memory_config(self):
        mc = MemoryConfig.get()
        assert isinstance(mc, MemoryConfig)

    def test_singleton_has_defaults(self):
        mc = MemoryConfig.get()
        assert isinstance(mc.enabled, bool)
        assert isinstance(mc.min_chars, int)
        assert isinstance(mc.max_facts, int)

    def test_singleton_thread_safety(self):
        import threading
        instances = []
        def get_instance():
            instances.append(MemoryConfig.get())
        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(inst is instances[0] for inst in instances)


class TestMemoryConfigSetEnabled:
    def test_set_enabled_false(self):
        mc = MemoryConfig()
        mc.set_enabled(False)
        assert mc.enabled is False

    def test_set_enabled_true(self):
        mc = MemoryConfig()
        mc.set_enabled(False)
        mc.set_enabled(True)
        assert mc.enabled is True

    def test_set_enabled_idempotent(self):
        mc = MemoryConfig()
        mc.set_enabled(True)
        assert mc.enabled is True

    def test_set_enabled_toggle(self):
        mc = MemoryConfig()
        mc.set_enabled(False)
        mc.set_enabled(True)
        mc.set_enabled(False)
        assert mc.enabled is False

    def test_set_enabled_multiple_times(self):
        mc = MemoryConfig()
        for i in range(10):
            mc.set_enabled(i % 2 == 0)
        assert mc.enabled is True


class TestMemoryConfigSetArchiveRetention:
    def test_set_archive_retention(self):
        mc = MemoryConfig()
        mc.set_archive_retention_days(60)
        assert mc.archive_retention_days == 60

    def test_set_archive_retention_zero(self):
        mc = MemoryConfig()
        mc.set_archive_retention_days(0)
        assert mc.archive_retention_days == 0

    def test_set_archive_retention_negative_clamps(self):
        mc = MemoryConfig()
        mc.set_archive_retention_days(-5)
        assert mc.archive_retention_days == 0

    def test_set_archive_retention_float(self):
        mc = MemoryConfig()
        mc.set_archive_retention_days(14.5)
        assert mc.archive_retention_days == 14.5

    def test_set_archive_retention_large(self):
        mc = MemoryConfig()
        mc.set_archive_retention_days(3650)
        assert mc.archive_retention_days == 3650

    def test_set_archive_retention_negative_large(self):
        mc = MemoryConfig()
        mc.set_archive_retention_days(-1000)
        assert mc.archive_retention_days == 0


class TestMemoryConfigSnapshot:
    def test_snapshot_keys(self):
        mc = MemoryConfig()
        snap = mc.snapshot()
        expected_keys = {
            "enabled", "min_chars", "max_facts", "store_path",
            "sync_remember", "consolidation_threshold",
            "maintenance_interval_minutes", "archive_retention_days",
        }
        assert set(snap.keys()) == expected_keys

    def test_snapshot_values(self):
        mc = MemoryConfig()
        snap = mc.snapshot()
        assert snap["enabled"] is True
        assert snap["min_chars"] == 80
        assert snap["max_facts"] == 5
        assert snap["store_path"] == "data/memory"
        assert snap["sync_remember"] is False
        assert snap["consolidation_threshold"] == 0.80
        assert snap["maintenance_interval_minutes"] == 60
        assert snap["archive_retention_days"] == 30

    def test_snapshot_returns_dict(self):
        mc = MemoryConfig()
        assert isinstance(mc.snapshot(), dict)

    def test_snapshot_reflects_changes(self):
        mc = MemoryConfig()
        mc.set_enabled(False)
        mc.set_archive_retention_days(100)
        snap = mc.snapshot()
        assert snap["enabled"] is False
        assert snap["archive_retention_days"] == 100

    def test_snapshot_independent_copy(self):
        mc = MemoryConfig()
        snap = mc.snapshot()
        snap["enabled"] = False
        assert mc.enabled is True

    def test_snapshot_after_multiple_changes(self):
        mc = MemoryConfig()
        mc.set_enabled(False)
        mc.set_archive_retention_days(10)
        mc.set_enabled(True)
        mc.set_archive_retention_days(20)
        snap = mc.snapshot()
        assert snap["enabled"] is True
        assert snap["archive_retention_days"] == 20


class TestMemoryConfigFromBool:
    def test_from_bool_true_values(self):
        assert MemoryConfig._from_bool("X", False) is False
        os.environ["TEST_BOOL_1"] = "1"
        assert MemoryConfig._from_bool("TEST_BOOL_1", False) is True
        os.environ["TEST_BOOL_TRUE"] = "true"
        assert MemoryConfig._from_bool("TEST_BOOL_TRUE", False) is True
        os.environ["TEST_BOOL_YES"] = "yes"
        assert MemoryConfig._from_bool("TEST_BOOL_YES", False) is True
        os.environ["TEST_BOOL_ON"] = "on"
        assert MemoryConfig._from_bool("TEST_BOOL_ON", False) is True
        os.environ["TEST_BOOL_TRUE2"] = "TRUE"
        assert MemoryConfig._from_bool("TEST_BOOL_TRUE2", False) is True
        os.environ["TEST_BOOL_YES2"] = " Yes "
        assert MemoryConfig._from_bool("TEST_BOOL_YES2", False) is True
        del os.environ["TEST_BOOL_1"]
        del os.environ["TEST_BOOL_TRUE"]
        del os.environ["TEST_BOOL_YES"]
        del os.environ["TEST_BOOL_ON"]
        del os.environ["TEST_BOOL_TRUE2"]
        del os.environ["TEST_BOOL_YES2"]

    def test_from_bool_false_values(self):
        os.environ["TEST_BOOL_FALSE"] = "false"
        assert MemoryConfig._from_bool("TEST_BOOL_FALSE", True) is False
        os.environ["TEST_BOOL_NO"] = "no"
        assert MemoryConfig._from_bool("TEST_BOOL_NO", True) is False
        os.environ["TEST_BOOL_OFF"] = "off"
        assert MemoryConfig._from_bool("TEST_BOOL_OFF", True) is False
        os.environ["TEST_BOOL_ZERO"] = "0"
        assert MemoryConfig._from_bool("TEST_BOOL_ZERO", True) is False
        del os.environ["TEST_BOOL_FALSE"]
        del os.environ["TEST_BOOL_NO"]
        del os.environ["TEST_BOOL_OFF"]
        del os.environ["TEST_BOOL_ZERO"]

    def test_from_bool_unset_returns_default(self):
        key = "UNSET_KEY_XYZ_123"
        if key in os.environ:
            del os.environ[key]
        assert MemoryConfig._from_bool(key, True) is True
        assert MemoryConfig._from_bool(key, False) is False

    def test_from_bool_empty_string(self):
        os.environ["TEST_BOOL_EMPTY"] = ""
        assert MemoryConfig._from_bool("TEST_BOOL_EMPTY", True) is False
        del os.environ["TEST_BOOL_EMPTY"]

    def test_from_bool_numeric_one(self):
        os.environ["TEST_BOOL_NUM1"] = "1"
        assert MemoryConfig._from_bool("TEST_BOOL_NUM1", False) is True
        del os.environ["TEST_BOOL_NUM1"]

    def test_from_bool_numeric_zero(self):
        os.environ["TEST_BOOL_NUM0"] = "0"
        assert MemoryConfig._from_bool("TEST_BOOL_NUM0", True) is False
        del os.environ["TEST_BOOL_NUM0"]

    def test_from_bool_case_insensitive(self):
        os.environ["TEST_BOOL_CASE"] = "True"
        assert MemoryConfig._from_bool("TEST_BOOL_CASE", False) is True
        os.environ["TEST_BOOL_CASE"] = "TRUE"
        assert MemoryConfig._from_bool("TEST_BOOL_CASE", False) is True
        os.environ["TEST_BOOL_CASE"] = "true"
        assert MemoryConfig._from_bool("TEST_BOOL_CASE", False) is True
        del os.environ["TEST_BOOL_CASE"]

    def test_from_bool_whitespace(self):
        os.environ["TEST_BOOL_WS"] = "  yes  "
        assert MemoryConfig._from_bool("TEST_BOOL_WS", False) is True
        del os.environ["TEST_BOOL_WS"]


class TestMemoryConfigEdgeCases:
    def test_min_chars_zero(self):
        mc = MemoryConfig(min_chars=0)
        assert mc.min_chars == 0

    def test_max_facts_zero(self):
        mc = MemoryConfig(max_facts=0)
        assert mc.max_facts == 0

    def test_consolidation_threshold_one(self):
        mc = MemoryConfig(consolidation_threshold=1.0)
        assert mc.consolidation_threshold == 1.0

    def test_consolidation_threshold_zero(self):
        mc = MemoryConfig(consolidation_threshold=0.0)
        assert mc.consolidation_threshold == 0.0

    def test_large_values(self):
        mc = MemoryConfig(min_chars=999999, max_facts=999999)
        assert mc.min_chars == 999999
        assert mc.max_facts == 999999

    def test_negative_min_chars(self):
        mc = MemoryConfig(min_chars=-10)
        assert mc.min_chars == -10

    def test_negative_max_facts(self):
        mc = MemoryConfig(max_facts=-1)
        assert mc.max_facts == -1

    def test_store_path_empty(self):
        mc = MemoryConfig(store_path="")
        assert mc.store_path == ""

    def test_store_path_long(self):
        long_path = "/a" * 100
        mc = MemoryConfig(store_path=long_path)
        assert mc.store_path == long_path

    def test_maintenance_interval_zero(self):
        mc = MemoryConfig(maintenance_interval_minutes=0)
        assert mc.maintenance_interval_minutes == 0

    def test_maintenance_interval_negative(self):
        mc = MemoryConfig(maintenance_interval_minutes=-10)
        assert mc.maintenance_interval_minutes == -10

    def test_archive_retention_zero(self):
        mc = MemoryConfig(archive_retention_days=0)
        assert mc.archive_retention_days == 0

    def test_archive_retention_negative(self):
        mc = MemoryConfig(archive_retention_days=-5)
        assert mc.archive_retention_days == -5


class TestMemoryConfigImmutability:
    def test_default_enabled_immutable(self):
        mc1 = MemoryConfig()
        mc2 = MemoryConfig()
        assert mc1.enabled == mc2.enabled

    def test_default_min_chars_immutable(self):
        mc1 = MemoryConfig()
        mc2 = MemoryConfig()
        assert mc1.min_chars == mc2.min_chars

    def test_default_max_facts_immutable(self):
        mc1 = MemoryConfig()
        mc2 = MemoryConfig()
        assert mc1.max_facts == mc2.max_facts

    def test_custom_values_independent(self):
        mc1 = MemoryConfig(min_chars=100)
        mc2 = MemoryConfig(min_chars=200)
        assert mc1.min_chars != mc2.min_chars


class TestMemoryConfigConstants:
    def test_default_enabled_constant(self):
        assert MemoryConfig.DEFAULT_ENABLED is True

    def test_default_min_chars_constant(self):
        assert MemoryConfig.DEFAULT_MIN_CHARS == 80

    def test_default_max_facts_constant(self):
        assert MemoryConfig.DEFAULT_MAX_FACTS == 5

    def test_default_store_path_constant(self):
        assert MemoryConfig.DEFAULT_STORE_PATH == "data/memory"

    def test_default_consolidation_threshold_constant(self):
        assert MemoryConfig.DEFAULT_CONSOLIDATION_THRESHOLD == 0.80

    def test_default_maintenance_interval_constant(self):
        assert MemoryConfig.DEFAULT_MAINTENANCE_INTERVAL_MINUTES == 60

    def test_default_archive_retention_constant(self):
        assert MemoryConfig.DEFAULT_ARCHIVE_RETENTION_DAYS == 30
