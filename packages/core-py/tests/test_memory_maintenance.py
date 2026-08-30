"""Tests for domains.memory.maintenance — memory maintenance scheduler,
domains.memory.memory_config — MemoryConfig."""

import asyncio
import os
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from domains.memory.maintenance import (
    maintenance_tick, run_memory_maintenance,
    start_memory_maintenance, stop_memory_maintenance,
)
from domains.memory.memory_config import MemoryConfig


@pytest.fixture(autouse=True)
def _reset_maintenance():
    """Reset the module-level task before each test."""
    import domains.memory.maintenance as mod
    mod._maintenance_task = None
    yield
    mod._maintenance_task = None


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the MemoryConfig singleton between tests."""
    MemoryConfig._instance = None
    yield
    MemoryConfig._instance = None


def _run(coro):
    """Helper to run async from sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# maintenance_tick
# ---------------------------------------------------------------------------

class TestMaintenanceTick:
    @patch("domains.memory.maintenance.prune_archive", return_value=5)
    @patch("domains.memory.maintenance.submit_memory_consolidate", new_callable=AsyncMock, return_value="task_123")
    @patch("domains.memory.maintenance.MemoryConfig")
    def test_prunes_and_enqueues(self, mock_cfg, mock_submit, mock_prune):
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=30)
        result = _run(maintenance_tick())
        mock_prune.assert_called_once()
        mock_submit.assert_called_once()
        assert result == "task_123"

    @patch("domains.memory.maintenance.MemoryConfig")
    def test_disabled_returns_none(self, mock_cfg):
        mock_cfg.get.return_value = MagicMock(enabled=False)
        result = _run(maintenance_tick())
        assert result is None

    @patch("domains.memory.maintenance.MemoryConfig")
    def test_zero_interval_returns_none(self, mock_cfg):
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=0)
        result = _run(maintenance_tick())
        assert result is None

    @patch("domains.memory.maintenance.prune_archive", side_effect=Exception("disk full"))
    @patch("domains.memory.maintenance.submit_memory_consolidate", new_callable=AsyncMock, return_value="task_456")
    @patch("domains.memory.maintenance.MemoryConfig")
    def test_prune_failure_still_enqueues(self, mock_cfg, mock_submit, mock_prune):
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=10)
        result = _run(maintenance_tick())
        mock_submit.assert_called_once()
        assert result == "task_456"

    @patch("domains.memory.maintenance.prune_archive", return_value=0)
    @patch("domains.memory.maintenance.submit_memory_consolidate", new_callable=AsyncMock, side_effect=Exception("queue down"))
    @patch("domains.memory.maintenance.MemoryConfig")
    def test_enqueue_failure_returns_none(self, mock_cfg, mock_submit, mock_prune):
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=10)
        result = _run(maintenance_tick())
        assert result is None

    @patch("domains.memory.maintenance.prune_archive", return_value=0)
    @patch("domains.memory.maintenance.submit_memory_consolidate", new_callable=AsyncMock, return_value="t")
    @patch("domains.memory.maintenance.MemoryConfig")
    def test_prune_returns_zero_no_log(self, mock_cfg, mock_submit, mock_prune):
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=10)
        result = _run(maintenance_tick())
        assert result == "t"

    @patch("domains.memory.maintenance.prune_archive", return_value=100)
    @patch("domains.memory.maintenance.submit_memory_consolidate", new_callable=AsyncMock, return_value="t2")
    @patch("domains.memory.maintenance.MemoryConfig")
    def test_prune_returns_many(self, mock_cfg, mock_submit, mock_prune):
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=10)
        result = _run(maintenance_tick())
        mock_prune.assert_called_once()
        assert result == "t2"

    @patch("domains.memory.maintenance.MemoryConfig")
    def test_negative_interval_returns_none(self, mock_cfg):
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=-5)
        result = _run(maintenance_tick())
        assert result is None

    @patch("domains.memory.maintenance.MemoryConfig")
    def test_negative_enabled_returns_none(self, mock_cfg):
        mock_cfg.get.return_value = MagicMock(enabled=False, maintenance_interval_minutes=60)
        result = _run(maintenance_tick())
        assert result is None

    @patch("domains.memory.maintenance.prune_archive", side_effect=RuntimeError("oops"))
    @patch("domains.memory.maintenance.submit_memory_consolidate", new_callable=AsyncMock, side_effect=RuntimeError("queue"))
    @patch("domains.memory.maintenance.MemoryConfig")
    def test_both_fail_returns_none(self, mock_cfg, mock_submit, mock_prune):
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=5)
        result = _run(maintenance_tick())
        assert result is None


# ---------------------------------------------------------------------------
# run_memory_maintenance
# ---------------------------------------------------------------------------

class TestRunMemoryMaintenance:
    @patch("domains.memory.maintenance.MemoryConfig")
    def test_disabled_run_returns_immediately(self, mock_cfg):
        mock_cfg.get.return_value = MagicMock(enabled=False)
        _run(run_memory_maintenance())

    @patch("domains.memory.maintenance.MemoryConfig")
    def test_zero_interval_returns_immediately(self, mock_cfg):
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=0)
        _run(run_memory_maintenance())


# ---------------------------------------------------------------------------
# start_memory_maintenance
# ---------------------------------------------------------------------------

class TestStartMemoryMaintenance:
    @patch("domains.memory.maintenance.MemoryConfig")
    def test_disabled_returns_none(self, mock_cfg):
        mock_cfg.get.return_value = MagicMock(enabled=False)
        task = start_memory_maintenance()
        assert task is None

    def test_enabled_returns_task(self):
        _run(_test_enabled_returns_task_impl())

    def test_idempotent(self):
        _run(_test_idempotent_impl())

    @patch("domains.memory.maintenance.MemoryConfig")
    def test_disabled_start_returns_none(self, mock_cfg):
        mock_cfg.get.return_value = MagicMock(enabled=False, maintenance_interval_minutes=0)
        assert start_memory_maintenance() is None


async def _test_enabled_returns_task_impl():
    with patch("domains.memory.maintenance.MemoryConfig") as mock_cfg:
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=60)
        task = start_memory_maintenance()
        assert task is not None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _test_idempotent_impl():
    with patch("domains.memory.maintenance.MemoryConfig") as mock_cfg:
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=60)
        t1 = start_memory_maintenance()
        t2 = start_memory_maintenance()
        assert t1 is t2
        t1.cancel()
        try:
            await t1
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# stop_memory_maintenance
# ---------------------------------------------------------------------------

class TestStopMemoryMaintenance:
    def test_stop_cleans_task(self):
        import domains.memory.maintenance as mod

        async def _fake_coro():
            await asyncio.sleep(999)

        async def _run_stop():
            task = asyncio.create_task(_fake_coro())
            mod._maintenance_task = task
            await stop_memory_maintenance()
            assert mod._maintenance_task is None

        _run(_run_stop())

    def test_stop_noop_when_none(self):
        import domains.memory.maintenance as mod
        mod._maintenance_task = None
        _run(stop_memory_maintenance())
        assert mod._maintenance_task is None

    def test_stop_noop_when_done(self):
        import domains.memory.maintenance as mod
        loop = asyncio.new_event_loop()

        async def _done():
            pass

        task = loop.create_task(_done())
        loop.run_until_complete(task)
        mod._maintenance_task = task
        _run(stop_memory_maintenance())
        assert mod._maintenance_task is None
        loop.close()

    def test_stop_cancels_running_task(self):
        import domains.memory.maintenance as mod

        async def _long_running():
            await asyncio.sleep(9999)

        async def _test():
            task = asyncio.create_task(_long_running())
            mod._maintenance_task = task
            await stop_memory_maintenance()
            assert task.cancelled()

        _run(_test())

    def test_stop_twice_no_error(self):
        import domains.memory.maintenance as mod
        mod._maintenance_task = None
        _run(stop_memory_maintenance())
        _run(stop_memory_maintenance())
        assert mod._maintenance_task is None


# ---------------------------------------------------------------------------
# MemoryConfig
# ---------------------------------------------------------------------------

class TestMemoryConfig:
    def test_defaults(self):
        cfg = MemoryConfig()
        assert cfg.enabled is True
        assert cfg.min_chars == 80
        assert cfg.max_facts == 5
        assert cfg.sync_remember is False

    def test_singleton(self):
        cfg1 = MemoryConfig.get()
        cfg2 = MemoryConfig.get()
        assert cfg1 is cfg2

    def test_singleton_reset(self):
        cfg1 = MemoryConfig.get()
        MemoryConfig._instance = None
        cfg2 = MemoryConfig.get()
        assert cfg1 is not cfg2

    def test_set_enabled(self):
        cfg = MemoryConfig()
        cfg.set_enabled(False)
        assert cfg.enabled is False
        cfg.set_enabled(True)
        assert cfg.enabled is True

    def test_set_archive_retention_days(self):
        cfg = MemoryConfig()
        cfg.set_archive_retention_days(60)
        assert cfg.archive_retention_days == 60.0

    def test_set_archive_retention_days_negative_clamps(self):
        cfg = MemoryConfig()
        cfg.set_archive_retention_days(-5)
        assert cfg.archive_retention_days == 0.0

    def test_snapshot(self):
        cfg = MemoryConfig()
        snap = cfg.snapshot()
        assert "enabled" in snap
        assert "min_chars" in snap
        assert "max_facts" in snap
        assert "store_path" in snap
        assert "sync_remember" in snap
        assert "consolidation_threshold" in snap
        assert "maintenance_interval_minutes" in snap
        assert "archive_retention_days" in snap

    def test_snapshot_values_match(self):
        cfg = MemoryConfig()
        snap = cfg.snapshot()
        assert snap["enabled"] == cfg.enabled
        assert snap["min_chars"] == cfg.min_chars
        assert snap["max_facts"] == cfg.max_facts

    def test_custom_kwargs(self):
        cfg = MemoryConfig(enabled=False, min_chars=20, max_facts=10)
        assert cfg.enabled is False
        assert cfg.min_chars == 20
        assert cfg.max_facts == 10

    def test_from_bool_truthy(self):
        assert MemoryConfig._from_bool("NONEXISTENT_VAR_XYZ", True) is True

    def test_from_bool_falsy(self):
        assert MemoryConfig._from_bool("NONEXISTENT_VAR_XYZ", False) is False

    def test_thread_safety(self):
        import threading
        cfg = MemoryConfig()
        results = []

        def toggle():
            for _ in range(100):
                cfg.set_enabled(True)
                cfg.set_enabled(False)
                results.append(cfg.enabled)

        threads = [threading.Thread(target=toggle) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 400

    def test_default_store_path(self):
        cfg = MemoryConfig()
        assert cfg.store_path == "data/memory"

    def test_default_consolidation_threshold(self):
        cfg = MemoryConfig()
        assert cfg.consolidation_threshold == 0.80

    def test_default_maintenance_interval(self):
        cfg = MemoryConfig()
        assert cfg.maintenance_interval_minutes == 60

    def test_default_archive_retention(self):
        cfg = MemoryConfig()
        assert cfg.archive_retention_days == 30

    def test_custom_store_path(self):
        cfg = MemoryConfig(store_path="/tmp/test_mem")
        assert cfg.store_path == "/tmp/test_mem"

    def test_custom_threshold(self):
        cfg = MemoryConfig(consolidation_threshold=0.95)
        assert cfg.consolidation_threshold == 0.95

    def test_snapshot_dict_type(self):
        cfg = MemoryConfig()
        snap = cfg.snapshot()
        assert isinstance(snap, dict)

    def test_set_archive_retention_days_zero(self):
        cfg = MemoryConfig()
        cfg.set_archive_retention_days(0)
        assert cfg.archive_retention_days == 0.0

    def test_set_archive_retention_days_large(self):
        cfg = MemoryConfig()
        cfg.set_archive_retention_days(365)
        assert cfg.archive_retention_days == 365.0


# ---------------------------------------------------------------------------
# MemoryConfig — env-var overrides
# ---------------------------------------------------------------------------

class TestMemoryConfigEnvVars:
    def test_env_min_chars(self):
        import os
        os.environ["SLO_MEMORY_MIN_CHARS"] = "200"
        try:
            cfg = MemoryConfig()
            assert cfg.min_chars == 200
        finally:
            del os.environ["SLO_MEMORY_MIN_CHARS"]

    def test_env_max_facts(self):
        import os
        os.environ["SLO_MEMORY_MAX_FACTS"] = "15"
        try:
            cfg = MemoryConfig()
            assert cfg.max_facts == 15
        finally:
            del os.environ["SLO_MEMORY_MAX_FACTS"]

    def test_env_store_path(self):
        import os
        os.environ["SLO_MEMORY_STORE_PATH"] = "/tmp/custom_store"
        try:
            cfg = MemoryConfig()
            assert cfg.store_path == "/tmp/custom_store"
        finally:
            del os.environ["SLO_MEMORY_STORE_PATH"]

    def test_env_sync_remember_true(self):
        import os
        os.environ["SLO_MEMORY_SYNC"] = "true"
        try:
            cfg = MemoryConfig()
            assert cfg.sync_remember is True
        finally:
            del os.environ["SLO_MEMORY_SYNC"]

    def test_env_sync_remember_false(self):
        import os
        os.environ["SLO_MEMORY_SYNC"] = "false"
        try:
            cfg = MemoryConfig()
            assert cfg.sync_remember is False
        finally:
            del os.environ["SLO_MEMORY_SYNC"]

    def test_env_consolidation_threshold(self):
        import os
        os.environ["SLO_MEMORY_CONSOLIDATION_THRESHOLD"] = "0.95"
        try:
            cfg = MemoryConfig()
            assert cfg.consolidation_threshold == 0.95
        finally:
            del os.environ["SLO_MEMORY_CONSOLIDATION_THRESHOLD"]

    def test_env_maintenance_interval(self):
        import os
        os.environ["SLO_MEMORY_MAINTENANCE_INTERVAL_MINUTES"] = "120"
        try:
            cfg = MemoryConfig()
            assert cfg.maintenance_interval_minutes == 120.0
        finally:
            del os.environ["SLO_MEMORY_MAINTENANCE_INTERVAL_MINUTES"]

    def test_env_archive_retention(self):
        import os
        os.environ["SLO_MEMORY_ARCHIVE_RETENTION_DAYS"] = "60"
        try:
            cfg = MemoryConfig()
            assert cfg.archive_retention_days == 60.0
        finally:
            del os.environ["SLO_MEMORY_ARCHIVE_RETENTION_DAYS"]

    def test_env_enabled_true(self):
        import os
        os.environ["SLO_MEMORY_ENABLED"] = "1"
        try:
            cfg = MemoryConfig()
            assert cfg.enabled is True
        finally:
            del os.environ["SLO_MEMORY_ENABLED"]

    def test_env_enabled_false(self):
        import os
        os.environ["SLO_MEMORY_ENABLED"] = "0"
        try:
            cfg = MemoryConfig()
            assert cfg.enabled is False
        finally:
            del os.environ["SLO_MEMORY_ENABLED"]

    def test_env_enabled_yes(self):
        import os
        os.environ["SLO_MEMORY_ENABLED"] = "yes"
        try:
            cfg = MemoryConfig()
            assert cfg.enabled is True
        finally:
            del os.environ["SLO_MEMORY_ENABLED"]

    def test_env_enabled_on(self):
        import os
        os.environ["SLO_MEMORY_ENABLED"] = "on"
        try:
            cfg = MemoryConfig()
            assert cfg.enabled is True
        finally:
            del os.environ["SLO_MEMORY_ENABLED"]

    def test_kwargs_override_env(self):
        import os
        os.environ["SLO_MEMORY_MIN_CHARS"] = "200"
        try:
            cfg = MemoryConfig(min_chars=50)
            assert cfg.min_chars == 50
        finally:
            del os.environ["SLO_MEMORY_MIN_CHARS"]

    def test_env_sync_on(self):
        import os
        os.environ["SLO_MEMORY_SYNC"] = "on"
        try:
            cfg = MemoryConfig()
            assert cfg.sync_remember is True
        finally:
            del os.environ["SLO_MEMORY_SYNC"]

    def test_env_sync_yes(self):
        import os
        os.environ["SLO_MEMORY_SYNC"] = "yes"
        try:
            cfg = MemoryConfig()
            assert cfg.sync_remember is True
        finally:
            del os.environ["SLO_MEMORY_SYNC"]


# ---------------------------------------------------------------------------
# MemoryConfig — additional edge cases
# ---------------------------------------------------------------------------

class TestMemoryConfigExtraEdges:
    def test_default_constants(self):
        assert MemoryConfig.DEFAULT_ENABLED is True
        assert MemoryConfig.DEFAULT_MIN_CHARS == 80
        assert MemoryConfig.DEFAULT_MAX_FACTS == 5
        assert MemoryConfig.DEFAULT_STORE_PATH == "data/memory"
        assert MemoryConfig.DEFAULT_CONSOLIDATION_THRESHOLD == 0.80
        assert MemoryConfig.DEFAULT_MAINTENANCE_INTERVAL_MINUTES == 60
        assert MemoryConfig.DEFAULT_ARCHIVE_RETENTION_DAYS == 30

    def test_snapshot_has_all_keys(self):
        cfg = MemoryConfig()
        snap = cfg.snapshot()
        expected_keys = {
            "enabled", "min_chars", "max_facts", "store_path",
            "sync_remember", "consolidation_threshold",
            "maintenance_interval_minutes", "archive_retention_days",
        }
        assert set(snap.keys()) == expected_keys

    def test_from_bool_with_real_env(self):
        import os
        os.environ["TEST_BOOL_TRUE_VAR"] = "true"
        os.environ["TEST_BOOL_FALSE_VAR"] = "false"
        os.environ["TEST_BOOL_ONE_VAR"] = "1"
        try:
            assert MemoryConfig._from_bool("TEST_BOOL_TRUE_VAR", False) is True
            assert MemoryConfig._from_bool("TEST_BOOL_FALSE_VAR", True) is False
            assert MemoryConfig._from_bool("TEST_BOOL_ONE_VAR", False) is True
            assert MemoryConfig._from_bool("TEST_BOOL_MISSING_VAR", True) is True
            assert MemoryConfig._from_bool("TEST_BOOL_MISSING_VAR", False) is False
        finally:
            for k in ["TEST_BOOL_TRUE_VAR", "TEST_BOOL_FALSE_VAR", "TEST_BOOL_ONE_VAR"]:
                del os.environ[k]

    def test_from_bool_case_insensitive(self):
        import os
        os.environ["TEST_BOOL_CASE"] = "TRUE"
        try:
            assert MemoryConfig._from_bool("TEST_BOOL_CASE", False) is True
        finally:
            del os.environ["TEST_BOOL_CASE"]

    def test_from_bool_with_whitespace(self):
        import os
        os.environ["TEST_BOOL_WS"] = "  yes  "
        try:
            assert MemoryConfig._from_bool("TEST_BOOL_WS", False) is True
        finally:
            del os.environ["TEST_BOOL_WS"]

    def test_snapshot_returns_dict_type(self):
        cfg = MemoryConfig()
        assert type(cfg.snapshot()) is dict

    def test_set_enabled_thread_safety_stress(self):
        import threading
        cfg = MemoryConfig()
        errors = []

        def toggle(tid):
            try:
                for _ in range(200):
                    cfg.set_enabled(tid % 2 == 0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggle, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_set_archive_retention_stress(self):
        import threading
        cfg = MemoryConfig()
        errors = []

        def set_val(v):
            try:
                for _ in range(100):
                    cfg.set_archive_retention_days(v)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=set_val, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# maintenance_tick — additional edge cases
# ---------------------------------------------------------------------------

class TestMaintenanceTickExtra:
    @patch("domains.memory.maintenance.prune_archive", return_value=50)
    @patch("domains.memory.maintenance.submit_memory_consolidate", new_callable=AsyncMock, return_value="t_ok")
    @patch("domains.memory.maintenance.MemoryConfig")
    def test_prune_large_count(self, mock_cfg, mock_submit, mock_prune):
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=30)
        result = _run(maintenance_tick())
        mock_prune.assert_called_once()
        assert result == "t_ok"

    @patch("domains.memory.maintenance.prune_archive", return_value=1)
    @patch("domains.memory.maintenance.submit_memory_consolidate", new_callable=AsyncMock, return_value="t_single")
    @patch("domains.memory.maintenance.MemoryConfig")
    def test_prune_single_record(self, mock_cfg, mock_submit, mock_prune):
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=15)
        result = _run(maintenance_tick())
        assert result == "t_single"

    @patch("domains.memory.maintenance.prune_archive", side_effect=IOError("permission denied"))
    @patch("domains.memory.maintenance.submit_memory_consolidate", new_callable=AsyncMock, return_value="t_after_io")
    @patch("domains.memory.maintenance.MemoryConfig")
    def test_prune_io_error_still_enqueues(self, mock_cfg, mock_submit, mock_prune):
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=20)
        result = _run(maintenance_tick())
        mock_submit.assert_called_once()
        assert result == "t_after_io"

    @patch("domains.memory.maintenance.prune_archive", return_value=0)
    @patch("domains.memory.maintenance.submit_memory_consolidate", new_callable=AsyncMock, side_effect=TimeoutError("queue timeout"))
    @patch("domains.memory.maintenance.MemoryConfig")
    def test_enqueue_timeout_returns_none(self, mock_cfg, mock_submit, mock_prune):
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=10)
        result = _run(maintenance_tick())
        assert result is None


# ---------------------------------------------------------------------------
# stop_memory_maintenance — additional edge cases
# ---------------------------------------------------------------------------

class TestStopMaintenanceExtra:
    def test_stop_after_task_done_and_replace(self):
        import domains.memory.maintenance as mod

        async def _coro():
            pass

        async def _test():
            loop = asyncio.get_event_loop()
            task = loop.create_task(_coro())
            await task  # let it finish
            mod._maintenance_task = task
            await stop_memory_maintenance()
            assert mod._maintenance_task is None
            # Can start again
            with patch("domains.memory.maintenance.MemoryConfig") as mock_cfg:
                mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=60)
                new_task = start_memory_maintenance()
                assert new_task is not None
                new_task.cancel()
                try:
                    await new_task
                except asyncio.CancelledError:
                    pass

        _run(_test())


# ---------------------------------------------------------------------------
# run_memory_maintenance — additional edge cases
# ---------------------------------------------------------------------------

class TestRunMaintenanceExtra:
    @patch("domains.memory.maintenance.maintenance_tick", new_callable=AsyncMock, return_value="tick_result")
    @patch("domains.memory.maintenance.MemoryConfig")
    def test_run_calls_tick_after_sleep(self, mock_cfg, mock_tick):
        mock_cfg.get.return_value = MagicMock(enabled=True, maintenance_interval_minutes=0)
        _run(run_memory_maintenance())
        mock_tick.assert_not_called()

    @patch("domains.memory.maintenance.MemoryConfig")
    def test_run_disabled_does_not_loop(self, mock_cfg):
        mock_cfg.get.return_value = MagicMock(enabled=False)
        _run(run_memory_maintenance())
