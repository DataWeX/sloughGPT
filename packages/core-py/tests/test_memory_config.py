"""Tests for MemoryConfig — env var parsing, singleton, toggle."""

import pytest
from domains.memory.memory_config import MemoryConfig


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before each test."""
    MemoryConfig._instance = None
    yield
    MemoryConfig._instance = None


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
    def test_kwargs_override_enabled(self):
        cfg = MemoryConfig(enabled=False)
        assert cfg.enabled is False

    def test_kwargs_override_min_chars(self):
        cfg = MemoryConfig(min_chars=200)
        assert cfg.min_chars == 200

    def test_kwargs_override_max_facts(self):
        cfg = MemoryConfig(max_facts=10)
        assert cfg.max_facts == 10

    def test_kwargs_override_store_path(self):
        cfg = MemoryConfig(store_path="/tmp/mem")
        assert cfg.store_path == "/tmp/mem"

    def test_kwargs_override_consolidation_threshold(self):
        cfg = MemoryConfig(consolidation_threshold=0.95)
        assert cfg.consolidation_threshold == 0.95


class TestMemoryConfigEnvVars:
    def test_env_enabled_true(self, monkeypatch):
        monkeypatch.setenv("SLO_MEMORY_ENABLED", "true")
        cfg = MemoryConfig()
        assert cfg.enabled is True

    def test_env_enabled_false(self, monkeypatch):
        monkeypatch.setenv("SLO_MEMORY_ENABLED", "false")
        cfg = MemoryConfig()
        assert cfg.enabled is False

    def test_env_enabled_1(self, monkeypatch):
        monkeypatch.setenv("SLO_MEMORY_ENABLED", "1")
        cfg = MemoryConfig()
        assert cfg.enabled is True

    def test_env_enabled_on(self, monkeypatch):
        monkeypatch.setenv("SLO_MEMORY_ENABLED", "on")
        cfg = MemoryConfig()
        assert cfg.enabled is True

    def test_env_enabled_yes(self, monkeypatch):
        monkeypatch.setenv("SLO_MEMORY_ENABLED", "yes")
        cfg = MemoryConfig()
        assert cfg.enabled is True

    def test_env_min_chars(self, monkeypatch):
        monkeypatch.setenv("SLO_MEMORY_MIN_CHARS", "120")
        cfg = MemoryConfig()
        assert cfg.min_chars == 120

    def test_env_max_facts(self, monkeypatch):
        monkeypatch.setenv("SLO_MEMORY_MAX_FACTS", "8")
        cfg = MemoryConfig()
        assert cfg.max_facts == 8

    def test_env_store_path(self, monkeypatch):
        monkeypatch.setenv("SLO_MEMORY_STORE_PATH", "/data/bank")
        cfg = MemoryConfig()
        assert cfg.store_path == "/data/bank"

    def test_env_sync_remember(self, monkeypatch):
        monkeypatch.setenv("SLO_MEMORY_SYNC", "1")
        cfg = MemoryConfig()
        assert cfg.sync_remember is True

    def test_env_consolidation_threshold(self, monkeypatch):
        monkeypatch.setenv("SLO_MEMORY_CONSOLIDATION_THRESHOLD", "0.9")
        cfg = MemoryConfig()
        assert cfg.consolidation_threshold == 0.9

    def test_env_maintenance_interval(self, monkeypatch):
        monkeypatch.setenv("SLO_MEMORY_MAINTENANCE_INTERVAL_MINUTES", "30")
        cfg = MemoryConfig()
        assert cfg.maintenance_interval_minutes == 30.0

    def test_env_archive_retention(self, monkeypatch):
        monkeypatch.setenv("SLO_MEMORY_ARCHIVE_RETENTION_DAYS", "60")
        cfg = MemoryConfig()
        assert cfg.archive_retention_days == 60.0


class TestMemoryConfigSingleton:
    def test_get_returns_same_instance(self):
        a = MemoryConfig.get()
        b = MemoryConfig.get()
        assert a is b

    def test_get_returns_memory_config_type(self):
        cfg = MemoryConfig.get()
        assert isinstance(cfg, MemoryConfig)

    def test_singleton_thread_safe(self):
        import threading
        instances = []
        def get_cfg():
            instances.append(MemoryConfig.get())
        threads = [threading.Thread(target=get_cfg) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(i is instances[0] for i in instances)


class TestMemoryConfigSetEnabled:
    def test_set_enabled_true(self):
        cfg = MemoryConfig()
        cfg.set_enabled(True)
        assert cfg.enabled is True

    def test_set_enabled_false(self):
        cfg = MemoryConfig()
        cfg.set_enabled(False)
        assert cfg.enabled is False

    def test_set_enabled_toggles(self):
        cfg = MemoryConfig()
        cfg.set_enabled(False)
        assert cfg.enabled is False
        cfg.set_enabled(True)
        assert cfg.enabled is True

    def test_set_enabled_affects_singleton(self):
        cfg = MemoryConfig.get()
        cfg.set_enabled(False)
        same = MemoryConfig.get()
        assert same.enabled is False

    def test_set_archive_retention_days(self):
        cfg = MemoryConfig()
        cfg.set_archive_retention_days(45)
        assert cfg.archive_retention_days == 45

    def test_set_archive_retention_clamps_negative(self):
        cfg = MemoryConfig()
        cfg.set_archive_retention_days(-10)
        assert cfg.archive_retention_days == 0

    def test_set_archive_retention_zero_prunes_all(self):
        cfg = MemoryConfig()
        cfg.set_archive_retention_days(0)
        assert cfg.archive_retention_days == 0

    def test_set_archive_retention_affects_singleton(self):
        cfg = MemoryConfig.get()
        cfg.set_archive_retention_days(60)
        assert MemoryConfig.get().archive_retention_days == 60

    def test_snapshot_returns_all_keys(self):
        cfg = MemoryConfig()
        snap = cfg.snapshot()
        for key in ("enabled", "min_chars", "max_facts", "store_path", "sync_remember",
                    "consolidation_threshold", "maintenance_interval_minutes", "archive_retention_days"):
            assert key in snap
        assert snap["archive_retention_days"] == cfg.archive_retention_days

    def test_snapshot_reflects_runtime_mutations(self):
        cfg = MemoryConfig()
        cfg.set_enabled(False)
        cfg.set_archive_retention_days(21)
        snap = cfg.snapshot()
        assert snap["enabled"] is False
        assert snap["archive_retention_days"] == 21
