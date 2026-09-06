"""Tests for memory.maintenance — periodic memory maintenance scheduler."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domains.memory.maintenance import (
    maintenance_tick,
    run_memory_maintenance,
    start_memory_maintenance,
    stop_memory_maintenance,
    _maintenance_task,
)
import domains.memory.maintenance as mod


@pytest.fixture(autouse=True)
def _reset_maintenance_task():
    """Ensure _maintenance_task is clean between tests."""
    mod._maintenance_task = None
    yield
    mod._maintenance_task = None


@pytest.fixture
def enabled_cfg():
    cfg = AsyncMock()
    cfg.enabled = True
    cfg.maintenance_interval_minutes = 30
    return cfg


@pytest.fixture
def disabled_cfg():
    cfg = AsyncMock()
    cfg.enabled = False
    cfg.maintenance_interval_minutes = 0
    return cfg


# ── maintenance_tick ──────────────────────────────────────────────────────


class TestMaintenanceTick:

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self, disabled_cfg):
        with patch.object(mod, "MemoryConfig") as mock_cfg:
            mock_cfg.get.return_value = disabled_cfg
            result = await maintenance_tick()
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_interval_zero(self):
        cfg = AsyncMock()
        cfg.enabled = True
        cfg.maintenance_interval_minutes = 0
        with patch.object(mod, "MemoryConfig") as mock_cfg:
            mock_cfg.get.return_value = cfg
            result = await maintenance_tick()
            assert result is None

    @pytest.mark.asyncio
    async def test_prunes_and_enqueues(self, enabled_cfg):
        with (
            patch.object(mod, "MemoryConfig") as mock_cfg,
            patch.object(mod, "prune_archive", return_value=2),
            patch.object(mod, "submit_memory_consolidate", new_callable=AsyncMock) as mock_submit,
        ):
            mock_cfg.get.return_value = enabled_cfg
            mock_submit.return_value = "task-123"
            result = await maintenance_tick()
            assert result == "task-123"
            mock_submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_on_enqueue_failure(self, enabled_cfg):
        with (
            patch.object(mod, "MemoryConfig") as mock_cfg,
            patch.object(mod, "prune_archive", return_value=0),
            patch.object(mod, "submit_memory_consolidate", new_callable=AsyncMock) as mock_submit,
        ):
            mock_cfg.get.return_value = enabled_cfg
            mock_submit.side_effect = RuntimeError("queue full")
            result = await maintenance_tick()
            assert result is None

    @pytest.mark.asyncio
    async def test_prune_failure_does_not_stop_enqueue(self, enabled_cfg):
        with (
            patch.object(mod, "MemoryConfig") as mock_cfg,
            patch.object(mod, "prune_archive", side_effect=RuntimeError("disk")),
            patch.object(mod, "submit_memory_consolidate", new_callable=AsyncMock) as mock_submit,
        ):
            mock_cfg.get.return_value = enabled_cfg
            mock_submit.return_value = "task-456"
            result = await maintenance_tick()
            assert result == "task-456"


# ── start_memory_maintenance ──────────────────────────────────────────────


class TestStartMemoryMaintenance:

    def test_returns_none_when_disabled(self, disabled_cfg):
        with patch.object(mod, "MemoryConfig") as mock_cfg:
            mock_cfg.get.return_value = disabled_cfg
            result = start_memory_maintenance()
            assert result is None

    def test_returns_none_when_interval_zero(self):
        cfg = AsyncMock()
        cfg.enabled = True
        cfg.maintenance_interval_minutes = 0
        with patch.object(mod, "MemoryConfig") as mock_cfg:
            mock_cfg.get.return_value = cfg
            result = start_memory_maintenance()
            assert result is None

    def test_returns_task_when_enabled(self, enabled_cfg):
        with patch.object(mod, "MemoryConfig") as mock_cfg, \
             patch("asyncio.create_task") as mock_create:
            mock_cfg.get.return_value = enabled_cfg
            mock_create.return_value = AsyncMock()
            result = start_memory_maintenance()
            assert result is not None

    def test_idempotent(self, enabled_cfg):
        with patch.object(mod, "MemoryConfig") as mock_cfg, \
             patch("asyncio.create_task") as mock_create:
            mock_cfg.get.return_value = enabled_cfg
            mock_create.return_value = AsyncMock()
            t1 = start_memory_maintenance()
            t2 = start_memory_maintenance()
            assert t1 is t2


# ── stop_memory_maintenance ───────────────────────────────────────────────


class TestStopMaintenance:

    @pytest.mark.asyncio
    async def test_noop_when_no_task(self):
        await stop_memory_maintenance()

    @pytest.mark.asyncio
    async def test_cancels_running_task(self):
        async def _noop():
            await asyncio.sleep(999)

        task = asyncio.create_task(_noop())
        mod._maintenance_task = task
        await stop_memory_maintenance()
        assert mod._maintenance_task is None
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_noop_when_task_already_done(self):
        async def _noop():
            pass

        task = asyncio.create_task(_noop())
        await task
        mod._maintenance_task = task
        await stop_memory_maintenance()
        assert mod._maintenance_task is None
