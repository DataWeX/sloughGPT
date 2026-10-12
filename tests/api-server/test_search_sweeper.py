"""Periodic search sweeper — env config, fan-out, lifecycle, failure isolation.

The sweeper is the "after a while" half of staleness handling: it wakes on an
interval and calls ``sweep()`` on every registered store that advertises the
Sweepable capability, so indexed stores catch up without a query forcing it.
"""

from __future__ import annotations

import domain.search.sweeper as sweeper
from domain.search.registry import SearchRegistry
from domain.search.types import SearchContext

CTX = SearchContext(user_id="u1")


class SweepStore:
    """Structurally Sweepable — has sweep()."""

    capability = "indexed"
    workspace_scoped = False

    def __init__(self, store, counts=None, boom=False):
        self.store = store
        self._counts = counts or {"new": 1}
        self._boom = boom
        self.sweeps = 0

    async def search(self, q, limit, ctx):
        return []

    def sweep(self):
        if self._boom:
            raise RuntimeError("disk on fire")
        self.sweeps += 1
        return dict(self._counts)


class LiveStore:
    """Live stores answer from the source — nothing to sweep."""

    store = "live_store"
    capability = "live"
    workspace_scoped = False

    async def search(self, q, limit, ctx):
        return []


def _registry(*adapters) -> SearchRegistry:
    reg = SearchRegistry()
    for a in adapters:
        reg.register(a)
    return reg


def test_interval_reads_env(monkeypatch):
    monkeypatch.delenv(sweeper.INTERVAL_ENV, raising=False)
    assert sweeper.sweep_interval_minutes() == sweeper.DEFAULT_SWEEP_INTERVAL_MINUTES
    monkeypatch.setenv(sweeper.INTERVAL_ENV, "2.5")
    assert sweeper.sweep_interval_minutes() == 2.5
    monkeypatch.setenv(sweeper.INTERVAL_ENV, "0")
    assert sweeper.sweep_interval_minutes() == 0.0
    monkeypatch.setenv(sweeper.INTERVAL_ENV, "nonsense")
    assert sweeper.sweep_interval_minutes() == sweeper.DEFAULT_SWEEP_INTERVAL_MINUTES
    monkeypatch.setenv(sweeper.INTERVAL_ENV, "-3")
    assert sweeper.sweep_interval_minutes() == 0.0  # negative clamps to disabled


def test_sweep_once_fans_out_and_isolates_failures(monkeypatch):
    good = SweepStore("good", {"new": 2, "refreshed": 1})
    bad = SweepStore("bad", boom=True)
    monkeypatch.setattr(sweeper, "get_registry", lambda: _registry(good, bad, LiveStore()))
    results = sweeper.sweep_once()
    assert results == {"good": {"new": 2, "refreshed": 1}, "bad": {"error": 1}}
    assert good.sweeps == 1


async def test_sweep_tick_runs_off_the_loop(monkeypatch):
    good = SweepStore("good", {"new": 1})
    monkeypatch.setattr(sweeper, "get_registry", lambda: _registry(good))
    assert await sweeper.sweep_tick() == {"good": {"new": 1}}


async def test_start_stop_lifecycle(monkeypatch):
    monkeypatch.setattr(sweeper, "get_registry", lambda: _registry())
    await sweeper.stop_search_sweeper()  # reset module state

    monkeypatch.setattr(sweeper, "sweep_interval_minutes", lambda: 0.0)
    assert sweeper.start_search_sweeper() is None, "interval 0 must not start a task"

    monkeypatch.setattr(sweeper, "sweep_interval_minutes", lambda: 1e-9)  # wake ~immediately
    try:
        first = sweeper.start_search_sweeper()
        assert first is not None and not first.done()
        assert sweeper.start_search_sweeper() is first, "second start must be idempotent"
    finally:
        await sweeper.stop_search_sweeper()
    await sweeper.stop_search_sweeper()  # stopping again is safe
