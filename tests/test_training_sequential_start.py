"""``/training/start`` sequential branch — job record + summary projection.

Covers execution.py's manifests mode: request validation, pre-flight
branching (missing manifest → 400), the sequential job-record fields, and
the additive ``_job_summary`` projection the web poller reads. The heavy
stage runner is stubbed at the executor seam — ``run_sequential_stages``
itself is covered in ``tests/test_sequential_stages.py``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVER_DIR = _REPO_ROOT / "apps" / "api" / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


def _manifest(tmp_path: Path, name: str) -> str:
    p = tmp_path / f"{name}_manifest.json"
    p.write_text(f'{{"dataset_id": "{name}", "version": 1}}\n', encoding="utf-8")
    return str(p)


class _StubExecutor:
    def __init__(self) -> None:
        self.submitted: list = []

    def submit(self, fn, *args, **kwargs):
        self.submitted.append((fn, args))


@pytest.fixture()
def job_cleanup():
    from training.jobs import training_jobs

    created: list[str] = []
    yield created
    for jid in created:
        training_jobs.pop(jid, None)
    try:
        from domain.infrastructure.cancel_manager import get_cancel_manager

        mgr = get_cancel_manager()
        for jid in created:
            if mgr.get(jid) is not None:
                mgr.finish(jid, error="")
    except Exception:
        pass
    try:
        from training.controller import get_training_controller

        get_training_controller().reset()
    except Exception:
        pass


def _start(request):
    from training.execution import start_training

    return asyncio.run(start_training(request, auth_user=None))


def test_sequential_start_records_job_and_projects_summary(tmp_path, job_cleanup):
    from training.jobs import training_jobs
    from training.jobs_api import _job_summary
    from training.schemas import TrainingRequest

    m1 = _manifest(tmp_path, "alpha")
    m2 = _manifest(tmp_path, "beta")
    req = TrainingRequest(name="seq run", model="slonet-native", manifests=[m1, m2], device="cpu")

    stub = _StubExecutor()
    with patch("training.execution.get_training_executor", return_value=stub):
        resp = _start(req)

    jid = resp["job_id"]
    job_cleanup.append(jid)
    assert resp["status"] == "started"
    rec = training_jobs[jid]
    assert rec["mode"] == "sequential"
    assert rec["manifests"] == [m1, m2]
    assert rec["probe_standings"] == []
    assert rec["trained_on"] == []
    assert rec["data_path"] == m1
    assert rec["data_source"] == "sequential"
    assert rec["status"] == "running"
    # run_training queued on the executor seam, never executed in-process
    assert len(stub.submitted) == 1

    summary = _job_summary(rec)
    assert summary["mode"] == "sequential"
    assert summary["manifests"] == [m1, m2]
    assert summary["probe_standings"] == []
    assert summary["trained_on"] == []

    # Additive contract: legacy job without the fields loses no key it had.
    legacy = _job_summary({"id": "x", "status": "completed", "loss": 1.5})
    assert legacy["loss"] == 1.5
    assert "probe_standings" not in legacy
    assert "manifests" not in legacy


def test_sequential_start_missing_manifest_rejected(tmp_path, job_cleanup):
    from training.schemas import TrainingRequest

    req = TrainingRequest(
        name="bad",
        model="m",
        manifests=[str(tmp_path / "does-not-exist.json")],
    )
    with patch("training.execution.get_training_executor") as executor:
        with pytest.raises(Exception, match="Sequential manifest not found"):
            _start(req)
    executor.submit.assert_not_called()


def test_sequential_start_rejects_combined_sources(tmp_path):
    import pytest
    from pydantic import ValidationError
    from training.schemas import TrainingRequest

    with pytest.raises(ValidationError):
        TrainingRequest(
            name="bad",
            model="m",
            dataset="some-dataset",
            manifests=[_manifest(tmp_path, "gamma")],
        )
