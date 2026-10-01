"""L3: infra background threads must not outlive tests.

Gate run5 evidence (2026-10-01): 44x pytest-timeout(180s) in the benchmark
zone plus ~54 leaked threads (idle-manager, faf-worker, scheduler_loop)
versus ~5 healthy. Every leak site already has a stop API — what's missing
is (a) an *immediate* stop (flag + time.sleep can lag 30-300s; must be
event-based) and (b) something that calls the stops between tests.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

_leaked_workflow = None


def _alive(prefix: str) -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith(prefix) and t.is_alive()]


def _mock_manager(config):
    from domain.feedback._internal.workflow import FeedbackWorkflowManager

    deps = []
    for _ in range(4):
        m = MagicMock()
        m.get_stats.return_value = {}
        deps.append(m)
    return FeedbackWorkflowManager(
        config=config,
        feedback_db=deps[0],
        meta_manager=deps[1],
        lora_store=deps[2],
        lora_updater=deps[3],
    )


class TestWorkflowStop:
    def test_stop_all_workflows_kills_scheduler_within_2s(self):
        from domain.feedback._internal import workflow as wf_mod

        cfg = wf_mod.WorkflowConfig(
            background_training_enabled=False,
            health_check_interval_seconds=30,  # long sleep: flag+sleep cannot pass
        )
        mgr = _mock_manager(cfg)
        mgr.start()
        try:
            assert mgr._scheduler_thread is not None
            assert mgr._scheduler_thread.is_alive()
            wf_mod.stop_all_workflows()
            mgr._scheduler_thread.join(timeout=2.0)
            assert not mgr._scheduler_thread.is_alive()
        finally:
            mgr.stop()
            if mgr._scheduler_thread is not None:
                mgr._scheduler_thread.join(timeout=5.0)

    def test_registry_tracks_started_managers(self):
        from domain.feedback._internal import workflow as wf_mod

        cfg = wf_mod.WorkflowConfig(background_training_enabled=False)
        mgr = _mock_manager(cfg)
        assert mgr not in wf_mod.started_workflows()
        mgr.start()
        try:
            assert mgr in wf_mod.started_workflows()
        finally:
            mgr.stop()
            mgr._scheduler_thread.join(timeout=5.0)
        assert mgr not in wf_mod.started_workflows()


class TestPoolReset:
    def test_reset_pool_gives_fresh_open_pool(self):
        from domain.infrastructure._internal import fire_and_forget as faf

        p1 = faf.get_pool()
        p1.submit(lambda: None)
        faf.reset_pool()
        p2 = faf.get_pool()
        assert p2 is not p1
        assert not p2._closed
        faf.reset_pool()


class TestIdleReset:
    def test_reset_stops_idle_loop_promptly(self):
        from domain.infrastructure._internal.idle_manager import get_idle_manager

        mgr = get_idle_manager()
        mgr.register("l3-teardown-probe-model")
        assert any(t.name == "idle-manager" for t in threading.enumerate())
        mgr.reset()
        deadline = time.time() + 5.0
        while time.time() < deadline and _alive("idle-manager"):
            time.sleep(0.05)
        assert not _alive("idle-manager")


class TestConftestTeardown:
    """Order-dependent: test_a leaks on purpose, the autouse conftest
    teardown must clean it before test_b asserts."""

    def test_a_leak_everything(self):
        global _leaked_workflow
        from domain.feedback._internal import workflow as wf_mod
        from domain.infrastructure._internal import fire_and_forget as faf
        from domain.infrastructure._internal.idle_manager import get_idle_manager

        get_idle_manager().register("l3-leak-model")
        faf.get_pool().submit(lambda: None)
        cfg = wf_mod.WorkflowConfig(background_training_enabled=False)
        _leaked_workflow = _mock_manager(cfg)
        _leaked_workflow.start()

    def test_b_no_leaked_threads_after_teardown(self):
        assert _leaked_workflow is not None
        assert not _alive("idle-manager")
        assert not _alive("faf-worker")
        thread = _leaked_workflow._scheduler_thread
        assert thread is None or not thread.is_alive()
