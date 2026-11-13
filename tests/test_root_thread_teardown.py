"""The ROOT suite must stop leaked background threads between tests.

Card 20261102_025758 (CI on main never concludes): `test-root` finishes
all 1628 tests in ~2.5 min and then HANGS at interpreter exit — CI logs
show the `feedback-scheduler`/engine threads alive for hours after the
summary (hourly "I/O operation on closed file" from `scheduler_loop`,
`Engine[shared]: main loop stopped` during finalization) until GitHub's
timeout killed the job and it held the concurrency group for every later
push to main.

`tests/core-py/conftest.py` has had the autouse `_stop_infra_threads`
teardown since aabe3dd09 — but that conftest never loads for `tests/`,
so the root suite leaks the same threads the core-py suite learned to
stop. This pair proves the wiring: test_a deliberately leaks a started
workflow manager; test_b (which runs after test_a's teardown) asserts
the registry is empty. Without the fixture, test_b is RED.
"""

from __future__ import annotations

from unittest.mock import MagicMock


def _leaked_manager():
    """A started FeedbackWorkflowManager nobody stops — the leak under test."""
    from domain.feedback._internal.workflow import FeedbackWorkflowManager, WorkflowConfig

    cfg = WorkflowConfig(
        background_training_enabled=False,
        health_check_interval_seconds=30,
    )
    deps = [MagicMock() for _ in range(4)]
    for d in deps:
        d.get_stats.return_value = {}
    mgr = FeedbackWorkflowManager(
        config=cfg,
        feedback_db=deps[0],
        meta_manager=deps[1],
        lora_store=deps[2],
        lora_updater=deps[3],
    )
    mgr.start()
    return mgr


def test_a_leak_a_started_workflow():
    """Leak phase: start a scheduler and do NOT stop it."""
    from domain.feedback._internal.workflow import started_workflows

    mgr = _leaked_manager()
    assert mgr in started_workflows(), "manager must register its leak to be visible"


def test_b_root_conftest_stopped_the_leak_from_test_a():
    """Teardown phase: the autouse root fixture must have stopped it.

    Runs strictly after test_a — if tests/conftest.py lacks the infra
    teardown, the registry still holds test_a's manager and this is RED.
    """
    from domain.feedback._internal.workflow import started_workflows

    assert started_workflows() == [], (
        "root conftest leaked a started workflow to the next test — tests/"
        "conftest.py needs the same _stop_infra_threads teardown "
        "tests/core-py/conftest.py has (aabe3dd09); leaked scheduler threads "
        "hang interpreter exit and held the CI/CD concurrency group for hours "
        "(card 20261102_025758)"
    )
