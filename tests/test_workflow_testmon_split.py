"""Main's full suite keeps the coverage gate and the testmon record in SEPARATE passes.

Card 20261104_085623 follow-up. pytest-cov and pytest-testmon both drive
coverage's ``CTracer``; under xdist their collectors stop out of LIFO order
at session end (``coverage/collector.py``: ``assert self._collectors[-1] is
self``) → INTERNALERROR exit 3 *after* the whole suite passed — main run
44542985298 (2026-11-19): 42596 passed, then
``Expected current collector to be <CTracer>, but it's <CTracer>``.

Each pass alone is CI-proven: ``--cov`` + ``-n auto`` without testmon is the
pre-card behavior; ``--testmon-noselect`` + ``-n auto`` without coverage is
every PR run of #353. This gate fails if a future edit merges them back into
one pytest command, or drops either pass entirely.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci_cd.yml"


def _test_core_py_run_steps() -> list[tuple[str, str]]:
    """(step name, run script) for every run-step of the test-core-py job."""
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["test-core-py"]
    return [
        (step.get("name", "?"), step["run"])
        for step in job["steps"]
        if "run" in step and "pytest" in step.get("run", "")
    ]


def test_coverage_and_testmon_never_share_one_pytest_command() -> None:
    """One pytest invocation may carry --cov XOR --testmon-*, never both."""
    steps = _test_core_py_run_steps()
    assert steps, "test-core-py has no pytest run step"
    for name, run in steps:
        has_cov = "--cov=" in run
        has_testmon = "--testmon" in run
        assert not (has_cov and has_testmon), (
            f"step {name!r} mixes the coverage gate and testmon recording in "
            "one pytest command — two CTracers stop out of LIFO order under "
            "xdist and crash the run after all tests passed (main run "
            "44542985298)"
        )


def test_both_passes_exist() -> None:
    """The split must not be 'solved' by deleting one of the passes."""
    steps = _test_core_py_run_steps()
    assert any("--cov=domain" in run for _, run in steps), (
        "the full-suite coverage gate (fail-under=50) is missing"
    )
    assert any("--testmon-noselect" in run for _, run in steps), (
        "the testmon recording pass is missing — PRs would restore a stale "
        "DB and over-select forever (never under-select, but the closure "
        "degrades to the full suite)"
    )
