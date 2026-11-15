"""Every job in the main-gating workflows must declare ``timeout-minutes``.

Card 20261102_025758 (CI on main never concludes): a hung job is not a
verdict, it is a hostage. On 2026-11-05 ``test-root`` finished all 1624
tests and then deadlocked at interpreter exit (plain ``atexit`` shutdown
ran after ``_python_exit`` had already joined the non-daemon pool
workers), sat for 6 hours until GitHub's default timeout killed it, and
held the workflow's concurrency group the whole time — every subsequent
main push queued with zero jobs and was cancelled the moment the next
push arrived. 6 hours of group lockup per hang because no job declared a
timeout.

The workflows are ours; the timeout is our policy, so the gate lives
here: a job without ``timeout-minutes`` reddens the suite instead of
quietly inheriting GitHub's 6-hour default.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# The two workflows that gate main: a hang in either one blocks every
# subsequent push to main (shared concurrency group per workflow).
GATING_WORKFLOWS = [
    REPO_ROOT / ".github" / "workflows" / "ci.yml",
    REPO_ROOT / ".github" / "workflows" / "ci_cd.yml",
]


def _jobs(workflow: Path) -> dict:
    data = yaml.safe_load(workflow.read_text())
    return data.get("jobs") or {}


@pytest.mark.parametrize("workflow", GATING_WORKFLOWS, ids=lambda p: p.name)
def test_every_job_declares_timeout_minutes(workflow: Path) -> None:
    """No job may inherit GitHub's 6-hour default timeout.

    A hung step must fail its own job in minutes, not hold the
    concurrency group for a full workday while every later push to main
    cancels with zero jobs.
    """
    jobs = _jobs(workflow)
    assert jobs, f"no jobs parsed from {workflow}"

    missing = sorted(name for name, job in jobs.items() if "timeout-minutes" not in job)
    assert not missing, (
        f"{workflow.name}: jobs missing timeout-minutes (6h default = group "
        f"held hostage on a hang, card 20261102_025758): {missing}"
    )


@pytest.mark.parametrize("workflow", GATING_WORKFLOWS, ids=lambda p: p.name)
def test_timeouts_are_positive_integers(workflow: Path) -> None:
    """A timeout of 0 or a string is a config bug, not a safety net."""
    for name, job in _jobs(workflow).items():
        value = job.get("timeout-minutes")
        if value is None:
            continue  # reported by test_every_job_declares_timeout_minutes
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"{workflow.name}: job {name!r} timeout-minutes must be an int, got {value!r}"
        )
        assert value > 0, f"{workflow.name}: job {name!r} timeout-minutes must be > 0"
