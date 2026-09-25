"""Recovery eligibility policy — which persisted jobs may be restarted.

One source of truth for both consumers that must agree:

- ``POST /recovery/recover/{job_id}`` rejects an ineligible job *before* it
  creates a recovery job row or spawns a worker thread;
- ``JobStore.get_recoverable_jobs()`` hides the same jobs from the UI's
  "Recoverable Jobs" list.

If they disagreed, the UI would offer a Resume that is guaranteed to fail.
"""

from __future__ import annotations

from pathlib import Path

# Job ``type`` values whose training loop is NOT ``SloughGPTTrainer``.
# Restarting one through the recovery path would run the wrong trainer with the
# wrong hyperparameters — and distill jobs record no ``data_path`` at all, so
# they fail inside the worker with "Data file not found: ''".
NON_RECOVERABLE_TYPES = frozenset({"distill", "lora", "visual"})


def job_type(job: dict) -> str:
    """Return the flow marker for ``job`` (``''`` when the job predates one)."""
    marker = job.get("type")
    if marker:
        return str(marker)
    config = job.get("config")
    if isinstance(config, dict):
        return str(config.get("type") or "")
    return ""


def data_path_resolvable(data_path: str) -> bool:
    """Whether ``data_path`` names something ``SloughGPTTrainer`` can open.

    Mirrors ``train_pipeline``'s resolution exactly: an existing file or
    directory, a dataset directory under ``data/``, or ``data/<name>/input.txt``.
    An empty string resolves to ``Path(".")`` (the CWD) which *is* a directory —
    so callers must reject emptiness first, as :func:`recovery_incompatibility`
    does.
    """
    p = Path(data_path)
    if p.is_file() or p.is_dir():
        return True
    under_data = Path("data") / data_path
    if under_data.is_dir():
        return True
    return (under_data / "input.txt").is_file()


def recovery_incompatibility(job: dict) -> str | None:
    """Return why ``job`` cannot be recovered, or ``None`` when it can.

    Called before any recovery state is written. An impossible recovery must
    fail the *request* (immediately, with a reason the user can act on) rather
    than fail the *job* (after a phase flip to ``error`` and a worker stack
    trace).
    """
    marker = job_type(job)
    if marker in NON_RECOVERABLE_TYPES:
        return (
            f"'{marker}' jobs are not recoverable: recovery restarts a standard "
            "text fine-tune, which is not the trainer this job used."
        )

    data_path = str(job.get("data_path") or "").strip()
    if not data_path:
        return (
            "No dataset recorded for this job, so training cannot be restarted. "
            "Start a new training run instead."
        )
    if not data_path_resolvable(data_path):
        return f"Dataset not found: {data_path}"
    return None
