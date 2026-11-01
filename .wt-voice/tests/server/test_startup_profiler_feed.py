"""Tests for the ⑩ startup-profiler feed (kanban card b753cad5).

Review 10 found the profiler fed *stages* but never *hooks*: ``profile_hook``
had zero callers, so every stage reported ``hook_count=0``/``duration=0``
and both health consumers (``health.startup_profile``,
``health.startup_export``) served zeros. These tests pin the feed:

1. ``StagedLoader.run_stage`` records per-hook timings (success, failure,
   timeout) into the profile.
2. ``get_summary()`` is non-mutating — a health call mid-boot must not
   close the open stage and drop in-flight hook timings.
3. The module-level ``profile_hook`` advertised in the docstring exists.
4. A bare ``TimeoutError`` records a readable error, not ``""``.
5. ``_finalize_startup``'s ⑩ finalizer computes the profile totals.
6. The ``health.startup_profile`` JSON keys stay unchanged.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = REPO_ROOT / "apps/api/server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from infrastructure.staged_loader import Stage, StagedLoader  # noqa: E402
from infrastructure.startup_profiler import (  # noqa: E402
    StartupProfiler,
    profile_hook,
    reset_profiler,
)

SUMMARY_KEYS = {"total_ms", "total_hooks", "stages"}
STAGE_SUMMARY_KEYS = {"name", "ms", "hooks", "slowest"}
PROFILE_KEYS = {
    "total_duration_ms",
    "total_hooks",
    "total_success",
    "total_failure",
    "timestamp",
    "stages",
}


@pytest.fixture(autouse=True)
def _clean_profiler():
    """The profiler is a global singleton — isolate every test."""
    reset_profiler()
    yield
    reset_profiler()


async def _sleep(ms: float) -> None:
    await asyncio.sleep(ms / 1000)


async def _boom() -> None:
    raise RuntimeError("kaput")


class TestRunStageFeedsProfile:
    """run_stage must record per-hook timings into the profile."""

    def test_success_and_failure_hooks_recorded(self):
        from infrastructure.startup_profiler import get_profiler

        loader = StagedLoader()
        loader.on(Stage.CRITICAL, "prof_fast", lambda: asyncio.sleep(0))
        loader.on(Stage.CRITICAL, "prof_slow", lambda: _sleep(20.0))
        loader.on(Stage.CRITICAL, "prof_boom", _boom)

        asyncio.run(loader.run_stage(Stage.CRITICAL))

        profile = get_profiler().get_profile()
        [stage] = profile.stages
        assert stage.name == "CRITICAL"
        assert stage.hook_count == 3
        assert stage.success_count == 2
        assert stage.failure_count == 1

        by_name = {h.name: h for h in stage.hooks}
        assert set(by_name) == {"prof_fast", "prof_slow", "prof_boom"}
        assert by_name["prof_boom"].success is False
        assert by_name["prof_boom"].error == "kaput"
        assert by_name["prof_slow"].duration_ms >= 5.0  # actually timed, not zeros
        assert stage.slowest_hook == "prof_slow"
        assert stage.total_duration_ms > 0.0

    def test_timeout_hook_recorded_as_failure(self):
        from infrastructure.startup_profiler import get_profiler

        loader = StagedLoader()
        loader.on(Stage.CRITICAL, "prof_hang", lambda: _sleep(5000.0), timeout=0.05)

        asyncio.run(loader.run_stage(Stage.CRITICAL))  # _run_hook swallows timeouts

        [stage] = get_profiler().get_profile().stages
        [hook] = stage.hooks
        assert hook.name == "prof_hang"
        assert hook.success is False
        assert "TimeoutError" in (hook.error or "")
        assert stage.failure_count == 1

    def test_empty_stage_still_recorded(self):
        from infrastructure.startup_profiler import get_profiler

        asyncio.run(StagedLoader().run_stage(Stage.BACKGROUND))
        [stage] = get_profiler().get_profile().stages
        assert stage.name == "BACKGROUND"
        assert stage.hook_count == 0


class TestGetSummaryIsNonMutating:
    """A health call mid-boot must never drop in-flight hook timings."""

    def test_open_stage_survives_a_summary_call(self):
        p = StartupProfiler()
        p.start_stage("critical")
        with p.profile_hook("db_pool"):
            pass

        # health.startup_profile mid-boot: must not close the open stage.
        summary = p.get_summary()
        assert summary["total_hooks"] == 1
        assert summary["stages"][0]["hooks"] == 1
        assert summary["stages"][0]["slowest"] == "db_pool"

        # A hook recorded AFTER the summary call must still land.
        with p.profile_hook("second_hook"):
            pass
        p.finish_stage()

        [stage] = p._profile.stages
        assert [h.name for h in stage.hooks] == ["db_pool", "second_hook"]
        assert stage.hook_count == 2

    def test_summary_shape_unchanged(self):
        p = StartupProfiler()
        p.start_stage("critical")
        with p.profile_hook("db_pool"):
            pass
        p.finish_stage()
        summary = p.get_summary()
        assert set(summary) == SUMMARY_KEYS
        assert set(summary["stages"][0]) == STAGE_SUMMARY_KEYS


class TestModuleApi:
    """The module docstring advertises ``from ... import profile_hook``."""

    def test_module_level_profile_hook_importable(self):
        from infrastructure.startup_profiler import get_profiler

        p = reset_profiler()
        p.start_stage("critical")
        with profile_hook("db_pool") as hook:
            hook.add_metric("connections", 5)
        p.finish_stage()

        [stage] = get_profiler().get_profile().stages
        assert stage.hooks[0].name == "db_pool"
        assert stage.hooks[0].metrics == {"connections": 5}

    def test_bare_timeout_records_readable_error(self):
        p = StartupProfiler()
        p.start_stage("critical")
        with pytest.raises(TimeoutError):
            with p.profile_hook("hang"):
                raise TimeoutError()
        p.finish_stage()

        [stage] = p._profile.stages
        assert stage.hooks[0].error == "TimeoutError"  # not ""
        assert stage.hooks[0].success is False

    def test_profile_to_dict_keys_unchanged(self):
        p = StartupProfiler()
        p.start_stage("critical")
        with p.profile_hook("db_pool"):
            pass
        p.finish_stage()
        assert set(p.finish().to_dict()) == PROFILE_KEYS


class TestFinalizerWiresFinish:
    """⑩: _finalize_startup must compute the profile's top-level totals."""

    def test_finalize_sets_profile_totals(self):

        prof = reset_profiler()
        prof.start_stage("critical")
        with prof.profile_hook("db_pool"):
            pass
        # Stage intentionally left OPEN — finish() must close and total it.

        from infrastructure.startup import StartupOrchestrator

        orch = StartupOrchestrator.__new__(StartupOrchestrator)
        loader = MagicMock()
        loader.get_status.return_value = {"errors": {}}

        async def drive() -> None:
            with (
                patch("infrastructure.startup_terminal.get_terminal_viz") as viz_factory,
                patch("infrastructure.startup_webhooks.get_webhook_manager") as wh_factory,
            ):
                viz_factory.return_value = MagicMock()
                manager = MagicMock()
                manager.emit = AsyncMock()
                wh_factory.return_value = manager
                orch._finalize_startup(loader)
                await orch._startup_complete_task

        asyncio.run(drive())

        totals = prof.get_profile().to_dict()
        assert totals["total_hooks"] == 1
        assert totals["total_success"] == 1
        assert totals["total_failure"] == 0
        loader.finish_history.assert_called_once()
