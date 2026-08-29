"""Tests for domains.memory.maintenance — memory maintenance scheduler."""

import asyncio
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


def _run(coro):
    """Helper to run async from sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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


class TestStartStopMemoryMaintenance:
    @patch("domains.memory.maintenance.MemoryConfig")
    def test_disabled_returns_none(self, mock_cfg):
        mock_cfg.get.return_value = MagicMock(enabled=False)
        task = start_memory_maintenance()
        assert task is None

    @patch("domains.memory.maintenance.MemoryConfig")
    def test_disabled_run_returns_immediately(self, mock_cfg):
        mock_cfg.get.return_value = MagicMock(enabled=False)
        _run(run_memory_maintenance())

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
        loop.run_until_complete(task)  # task finishes
        mod._maintenance_task = task
        _run(stop_memory_maintenance())
        assert mod._maintenance_task is None
        loop.close()
