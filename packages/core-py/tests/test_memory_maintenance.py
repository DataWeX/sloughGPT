"""Tests for the periodic memory maintenance scheduler (domains/memory/maintenance.py)."""

import asyncio

import pytest

from domains.memory import maintenance
from domains.memory.memory_config import MemoryConfig


@pytest.fixture(autouse=True)
async def reset_scheduler():
    """Ensure no scheduler task leaks between tests."""
    await maintenance.stop_memory_maintenance()
    yield
    await maintenance.stop_memory_maintenance()


@pytest.fixture(autouse=True)
def enabled_config(monkeypatch):
    """Default config state: memory enabled with a normal interval."""
    cfg = MemoryConfig.get()
    monkeypatch.setattr(cfg, "enabled", True)
    monkeypatch.setattr(cfg, "maintenance_interval_minutes", 60)
    yield cfg


@pytest.fixture
def fake_submit(monkeypatch):
    """Replace submit_memory_consolidate with a recording async fake."""
    calls = []

    async def _fake(**kwargs):
        calls.append(kwargs)
        return "task-1"

    monkeypatch.setattr(maintenance, "submit_memory_consolidate", _fake)
    return calls


@pytest.fixture(autouse=True)
def fake_prune(monkeypatch):
    """Keep tick() hermetic: replace prune_archive with a recording fake.

    Returns a call log; each entry records the retain_days kwarg actually
    passed (None means "use the config default retention window").
    """
    calls = []

    def _fake(retain_days=None):
        calls.append(retain_days)
        return 0

    monkeypatch.setattr(maintenance, "prune_archive", _fake)
    return calls


def test_tick_enqueues_consolidate(fake_submit, monkeypatch):
    monkeypatch.setattr(
        "domains.infrastructure.task_queue.get_task_queue", lambda: None)
    task_id = asyncio.run(maintenance.maintenance_tick())
    assert task_id == "task-1"
    assert len(fake_submit) == 1
    assert fake_submit[0]["queue"] is None


def test_tick_prunes_archive_with_default_retention(fake_submit, monkeypatch,
                                                    fake_prune):
    monkeypatch.setattr(
        "domains.infrastructure.task_queue.get_task_queue", lambda: None)
    asyncio.run(maintenance.maintenance_tick())
    assert fake_prune == [None]


def test_tick_does_not_prune_when_disabled(enabled_config, monkeypatch,
                                           fake_submit, fake_prune):
    monkeypatch.setattr(enabled_config, "enabled", False)
    assert asyncio.run(maintenance.maintenance_tick()) is None
    assert fake_submit == []
    assert fake_prune == []


def test_tick_does_not_prune_when_interval_zero(enabled_config, monkeypatch,
                                                fake_submit, fake_prune):
    monkeypatch.setattr(enabled_config, "maintenance_interval_minutes", 0)
    assert asyncio.run(maintenance.maintenance_tick()) is None
    assert fake_submit == []
    assert fake_prune == []


def test_tick_prunes_before_enqueue(fake_submit, monkeypatch, fake_prune):
    order = []
    monkeypatch.setattr(
        "domains.infrastructure.task_queue.get_task_queue", lambda: None)

    def _recording_prune(retain_days=None):
        order.append("prune")
        return 0

    async def _recording_submit(**kwargs):
        order.append("enqueue")
        return "task-1"

    monkeypatch.setattr(maintenance, "prune_archive", _recording_prune)
    monkeypatch.setattr(maintenance, "submit_memory_consolidate", _recording_submit)
    asyncio.run(maintenance.maintenance_tick())
    assert order == ["prune", "enqueue"]


def test_tick_still_enqueues_when_prune_raises(fake_submit, monkeypatch):
    monkeypatch.setattr(
        "domains.infrastructure.task_queue.get_task_queue", lambda: None)

    def _broken_prune(retain_days=None):
        raise OSError("disk full")

    monkeypatch.setattr(maintenance, "prune_archive", _broken_prune)
    task_id = asyncio.run(maintenance.maintenance_tick())
    assert task_id == "task-1"
    assert len(fake_submit) == 1


def test_tick_swallows_enqueue_error(monkeypatch, fake_submit):
    async def _broken(**kwargs):
        raise RuntimeError("queue down")

    monkeypatch.setattr(maintenance, "submit_memory_consolidate", _broken)
    assert asyncio.run(maintenance.maintenance_tick()) is None


def test_run_returns_immediately_when_interval_zero(enabled_config, monkeypatch):
    monkeypatch.setattr(enabled_config, "maintenance_interval_minutes", 0)
    asyncio.run(maintenance.run_memory_maintenance())


def test_run_returns_immediately_when_disabled(enabled_config, monkeypatch):
    monkeypatch.setattr(enabled_config, "enabled", False)
    asyncio.run(maintenance.run_memory_maintenance())


def test_start_and_stop_loop_ticks(enabled_config, monkeypatch, fake_submit):
    """A started loop enqueues at least one consolidate within its interval."""
    monkeypatch.setattr(enabled_config, "maintenance_interval_minutes", 0.001)

    async def scenario():
        task = maintenance.start_memory_maintenance()
        assert task is not None
        assert maintenance.start_memory_maintenance() is task
        await asyncio.sleep(0.15)
        await maintenance.stop_memory_maintenance()

    asyncio.run(scenario())
    assert len(fake_submit) >= 1


def test_start_returns_none_when_disabled(enabled_config, monkeypatch):
    monkeypatch.setattr(enabled_config, "enabled", False)
    assert maintenance.start_memory_maintenance() is None


def test_memory_config_exposes_interval():
    cfg = MemoryConfig(maintenance_interval_minutes=120)
    assert cfg.maintenance_interval_minutes == 120


def test_memory_config_interval_default():
    cfg = MemoryConfig()
    assert cfg.maintenance_interval_minutes == MemoryConfig.DEFAULT_MAINTENANCE_INTERVAL_MINUTES


def test_memory_config_consolidation_threshold_default():
    cfg = MemoryConfig()
    assert cfg.consolidation_threshold == MemoryConfig.DEFAULT_CONSOLIDATION_THRESHOLD
    assert cfg.consolidation_threshold == 0.80


def test_memory_config_archive_retention_exposes_value():
    cfg = MemoryConfig(archive_retention_days=90)
    assert cfg.archive_retention_days == 90


def test_memory_config_archive_retention_default():
    cfg = MemoryConfig()
    assert cfg.archive_retention_days == MemoryConfig.DEFAULT_ARCHIVE_RETENTION_DAYS
