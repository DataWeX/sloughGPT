"""Boot-time auto-resume policy (decision card 20261102_001425, option 2).

The recovery machinery exists end-to-end; this module is only the
boot-time automation, and it is deliberately a *policy*, not a loop:
flag-gated (``enabled``), at most one concurrent resume, skip when no
loadable checkpoint. Every decision is emitted as a contract event from
``training.process_events`` so the automation is legible — what resumed,
what was skipped and why — and never a surprise GPU burn.

Pure planning: no I/O, no threads. The caller (startup hook) fetches the
durable rows, supplies the checkpoint probe, then executes ``to_resume``
through the same recovery path the endpoint uses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from training.process_events import make_event

logger = logging.getLogger("slo")

MAX_CONCURRENT_RESUMES = 1


class AutoResumeOutcome(Enum):
    DISABLED = "disabled"  # policy flag off — nothing planned, nothing emitted
    RESUMED = "resumed"  # at least one job handed to the recovery path
    SKIPPED = "skipped"  # enabled, but every eligible job failed a gate


@dataclass
class AutoResumePlan:
    outcome: AutoResumeOutcome
    to_resume: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


def plan_auto_resume(
    jobs: list[dict[str, Any]],
    *,
    enabled: bool,
    checkpoint_loadable: Callable[[dict[str, Any]], bool],
    max_concurrent: int = MAX_CONCURRENT_RESUMES,
) -> AutoResumePlan:
    """Decide which interrupted jobs auto-resume at boot.

    ``jobs`` must be in preference order (the durable snapshot sorts
    created_at desc — the freshest run is attempted first). Each job is
    evaluated once, in order:

    - flag off            -> nothing planned, no events (no phantom noise)
    - cap reached         -> stop evaluating; an un-evaluated job gets NO
                             event (claiming "skipped" for a job we never
                             looked at would be a lie)
    - checkpoint unloadable -> skipped WITH an event (legible decline)
    - otherwise           -> resumed WITH an auto_train_started event

    ``checkpoint_loadable`` may raise — a probe that explodes (corrupt
    file, missing dir) counts as unloadable, never crashes the boot path.
    """
    if not enabled:
        return AutoResumePlan(AutoResumeOutcome.DISABLED)

    to_resume: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for job in jobs:
        if len(to_resume) >= max_concurrent:
            break
        job_id = job.get("id") or ""
        if not job_id:
            continue
        try:
            loadable = bool(checkpoint_loadable(job))
        except Exception as e:
            logger.warning(
                "Auto-resume checkpoint probe failed for %s: %s",
                job_id,
                e,
                extra={"tag": "TRAIN"},
            )
            loadable = False
        if not loadable:
            events.append(
                make_event(
                    job_id,
                    "auto_resume_skipped",
                    job.get("status") or "interrupted",
                    surface="skipped",
                    reason="no loadable checkpoint",
                    name=job.get("name") or "",
                )
            )
            continue
        to_resume.append(job)
        events.append(
            make_event(
                job_id,
                "auto_train_started",
                "running",
                surface="running",
                name=job.get("name") or "",
            )
        )

    outcome = (
        AutoResumeOutcome.RESUMED if to_resume else AutoResumeOutcome.SKIPPED
    )
    return AutoResumePlan(outcome, to_resume, events)


def run_auto_resume(
    store: Any,
    *,
    enabled: bool,
    checkpoint_loadable: Callable[[dict[str, Any]], bool],
    recover: Callable[[str], dict[str, Any]],
    max_concurrent: int = MAX_CONCURRENT_RESUMES,
) -> AutoResumePlan:
    """Fetch, plan, and execute boot auto-resume.

    The execution seam the startup hook calls. ``recover`` is injected —
    the production wiring passes the SAME recovery path the endpoint uses
    (server-authority re-validation included), so a boot-time dispatch is
    never a privileged side door: a job that moved on since restore() is
    refused, the refusal is swallowed here (recorded as a skip event),
    and boot continues.
    """
    try:
        rows = store.get_recoverable_jobs()
    except Exception as e:
        logger.warning(
            "Auto-resume store fetch failed: %s", e, extra={"tag": "TRAIN"}
        )
        rows = []

    plan = plan_auto_resume(
        rows,
        enabled=enabled,
        checkpoint_loadable=checkpoint_loadable,
        max_concurrent=max_concurrent,
    )
    if plan.outcome is not AutoResumeOutcome.RESUMED:
        return plan

    executed: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for job in plan.to_resume:
        job_id = job.get("id") or ""
        try:
            recover(job_id)
            executed.append(job)
        except Exception as e:
            logger.warning(
                "Auto-resume refused for %s: %s", job_id, e, extra={"tag": "TRAIN"}
            )
            events.append(
                make_event(
                    job_id,
                    "auto_resume_skipped",
                    job.get("status") or "interrupted",
                    surface="skipped",
                    reason=f"recovery refused: {e}",
                    name=job.get("name") or "",
                )
            )
    # Merge: checkpoint-gate skips from planning stay; an executed resume
    # keeps its auto_train_started; a refused resume is re-emitted as a
    # skip so the dock shows the truth, not the intent.
    refused_ids = {e["process_id"] for e in events}
    merged = [
        e
        for e in plan.events
        if e["event"] == "auto_resume_skipped"
        or (
            e["process_id"] not in refused_ids
            and e["process_id"] in {j.get("id") for j in executed}
        )
    ] + events
    plan.to_resume = executed
    plan.events = merged
    plan.outcome = (
        AutoResumeOutcome.RESUMED if executed else AutoResumeOutcome.SKIPPED
    )
    return plan
