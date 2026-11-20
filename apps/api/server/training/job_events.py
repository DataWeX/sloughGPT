"""Subscribe-only stream of training job state changes (card 20261120_110356).

``/training/job-events/stream`` pushes one standard-envelope frame per job
mutation — ``started``, ``progress`` (each trainer step), and the terminal
status — published by the job store (``TrainingRuntime.sync``), so clients
aggregate live instead of polling ``GET /training/jobs/{id}`` every 3s. The
first frame is an ``init`` snapshot (``data.jobs``) of every tracked job at
connect time; heartbeats every 10s while idle.

This is the subscribe-only half the audit found missing: ``/training/stream``
stays start-and-stream (grandfathered — it enqueues the auto-train task), and
existing polls demote to fallback-only in the apps/web follow-up.

Envelope contract (pinned by ``test_training_job_events.py`` — web and mobile
consume this vocabulary):

* ``stream``: ``"training"``
* ``phase``: ``init`` | ``started`` | ``progress`` | ``completed`` |
  ``failed`` | ``cancelled`` | ``interrupted`` | ``abandoned`` | ``recovered``
* ``status``: ``working`` (init/started/progress) | ``complete`` (completed)
  | ``success`` (recovered) | ``error`` (other terminal)
* ``data``: the job view (``init`` → ``{"jobs": [...]}``)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from infrastructure.auth import require_auth_if_enabled
from infrastructure.contract import RouteSpec, create_router
from starlette.responses import StreamingResponse

from domain.agents import ToolSpec

from .runtime import get_training_runtime

logger = logging.getLogger("slo")

router = APIRouter()

STREAM_NAME = "training"
HEARTBEAT_INTERVAL_S = 10.0

# Closed phase vocabulary — the contract test pins this set. Anything the
# store publishes outside it would silently fall through to status "error".
PHASES = frozenset(
    {
        "init",
        "started",
        "progress",
        "completed",
        "failed",
        "cancelled",
        "interrupted",
        "abandoned",
        "recovered",
    }
)

_STATUS_BY_PHASE = {
    "init": "working",
    "started": "working",
    "progress": "working",
    "completed": "complete",
    "recovered": "success",
}
# failed / cancelled / interrupted / abandoned → "error" (default below).

_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _status_for(phase: str) -> str:
    """Map a phase onto the envelope's status vocabulary."""
    return _STATUS_BY_PHASE.get(phase, "error")


def _frame(phase: str, data: dict[str, Any], message: str = "") -> dict[str, Any]:
    """Build one standard envelope frame (shape pinned by the contract test)."""
    return {
        "stream": STREAM_NAME,
        "phase": phase,
        "status": _status_for(phase),
        "data": data,
        "meta": {},
        "message": message or phase,
    }


def _init_frame(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """First frame on connect: baseline snapshot of every tracked job."""
    return _frame("init", {"jobs": jobs}, message=f"{len(jobs)} job(s)")


def _envelope(event: dict[str, Any]) -> dict[str, Any]:
    """Wrap a store event (``{"phase", "job"}``) as an envelope frame."""
    phase = str(event.get("phase") or "progress")
    job = event.get("job") or {}
    job_id = job.get("id") or ""
    return _frame(phase, job, message=f"{phase}: {job_id}".rstrip(": "))


async def training_job_events(
    request: Request,
    auth_user: dict | None = Depends(require_auth_if_enabled),
):
    """Subscribe to training job state changes (SSE).

    Emits the standard envelope: an ``init`` snapshot first (``data.jobs``),
    then one frame per job mutation (``started`` / ``progress`` / terminal
    status) as the job store publishes them, with 10s heartbeats while idle.
    Raw wire (``envelope=False``) — the ``StreamingResponse`` passes through
    untouched.
    """
    runtime = get_training_runtime()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _push(event: dict[str, Any]) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except RuntimeError:
            pass  # loop already closed — the client is gone

    token, jobs = runtime.subscribe(_push)

    async def generate():
        try:
            yield "data: " + json.dumps(_init_frame(jobs), default=str) + "\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_INTERVAL_S)
                except TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield "data: " + json.dumps(_envelope(event), default=str) + "\n\n"
        finally:
            runtime.unsubscribe(token)

    return StreamingResponse(generate(), media_type="text/event-stream", headers=_HEADERS)


# ── Descriptor projection: one contract for the stream ──────────────────

SPEC_JOB_EVENTS = ToolSpec(
    name="training.job_events",
    description=inspect.cleandoc(training_job_events.__doc__ or ""),
    parameters=[],
    execute=training_job_events,
    auth_scope="authenticated",
    idempotent=True,
    version="1",
)

router.include_router(
    create_router(
        SPEC_JOB_EVENTS,
        RouteSpec(
            path="/training/job-events/stream",
            envelope=False,
            summary="Training Job Events Stream",
        ),
        training_job_events,
    )
)
