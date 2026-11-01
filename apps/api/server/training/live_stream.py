"""Live training stream — SSE feed-arrival + job activity over the wire.

``/training/live-stream`` pushes a snapshot every few seconds: per-source
record totals with deltas since the previous frame (what arrived), plus active
training jobs (what is running). It is the observability half of the feed
pipeline — ``/training/feed`` lets consumers *read* the corpus, this endpoint
lets operators *watch* it grow while training runs against it.

The client envelope matches the standard SSE shape (``stream``/``phase``/
``status``/``data``); ``data.changed`` is True when anything moved since the
previous frame so UIs can filter the event log to real arrivals.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from infrastructure.auth import require_auth_if_enabled
from starlette.responses import StreamingResponse

from .feeds import list_feed_sources
from .jobs import training_jobs

logger = logging.getLogger("slo")

router = APIRouter(tags=["training-feed"])

LIVE_STREAM_INTERVAL = 2.0

_INACTIVE_STATUSES = frozenset({"completed", "failed", "stopped"})


def _active_jobs() -> list[dict[str, Any]]:
    """Compact summary of jobs not in a terminal state (best-effort)."""
    try:
        items = list(training_jobs.values())
    except Exception as exc:  # store unavailable — stream stays up
        logger.warning("Live stream could not list jobs: %s", exc)
        return []
    active: list[dict[str, Any]] = []
    for job in items:
        status = str(job.get("status") or "")
        if status in _INACTIVE_STATUSES:
            continue
        active.append(
            {
                "id": str(job.get("id") or job.get("job_id") or ""),
                "name": str(job.get("name") or ""),
                "status": status,
                "dataset": str(job.get("dataset") or ""),
                "progress": job.get("progress") or 0,
            }
        )
    return active


def _jobs_sig(active: list[dict[str, Any]]) -> str:
    """Stable signature of the active-job list for change detection."""
    return "|".join(f"{j['id']}:{j['status']}:{j['progress']}" for j in active)


def build_live_snapshot(prev: dict[str, Any] | None = None) -> tuple[dict[str, Any], bool]:
    """Build one live snapshot and whether anything changed since ``prev``.

    Args:
        prev: the previous snapshot (None on the first frame).

    Returns:
        ``(snapshot, changed)`` — snapshot carries per-source totals with
        per-frame deltas, active jobs, and a timestamp; changed is True when a
        total grew or the active-job set shifted since ``prev``.
    """
    prev_totals: dict[str, int] = (prev or {}).get("totals") or {}
    prev_jobs_sig = _jobs_sig(((prev or {}).get("jobs") or {}).get("active") or []) if prev else ""

    sources: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    changed = False
    for row in list_feed_sources():
        name = row["name"]
        total = int(row.get("total") or 0)
        totals[name] = total
        delta = 0 if prev is None else total - int(prev_totals.get(name, 0))
        if delta:
            changed = True
        sources.append(
            {
                "name": name,
                "spec": row.get("spec") or f"feed:{name}",
                "total": total,
                "delta": delta,
                "last_updated": row.get("last_updated"),
            }
        )

    active = _active_jobs()
    if prev is not None and _jobs_sig(active) != prev_jobs_sig:
        changed = True

    snapshot = {
        "sources": sources,
        "totals": totals,
        "jobs": {"active": active, "count": len(active)},
        "at": datetime.now(UTC).isoformat(),
    }
    if prev is None:
        # First frame is the baseline but still worth rendering as an event.
        changed = True
    return snapshot, changed


def _envelope(snapshot: dict[str, Any], changed: bool) -> dict[str, Any]:
    return {
        "stream": "training-live",
        "phase": "LIVE",
        "status": "working",
        "data": {**snapshot, "changed": changed},
    }


@router.get("/training/live-stream")
async def training_live_stream(
    request: Request,
    auth_user: dict | None = Depends(require_auth_if_enabled),
):
    """SSE stream of feed arrivals and active training jobs.

    Emits one standard-envelope frame every ``LIVE_STREAM_INTERVAL`` seconds
    (heartbeat cadence, like the health stream). ``data.changed`` marks frames
    where a feed total grew or the active-job set shifted.

    Args:
        request: FastAPI Request — checked for disconnect each cycle.

    Returns:
        StreamingResponse with ``text/event-stream`` content type.
    """

    async def generate():
        prev: dict[str, Any] | None = None
        while True:
            if await request.is_disconnected():
                break
            try:
                snapshot, changed = await asyncio.to_thread(build_live_snapshot, prev)
                prev = snapshot
                frame = _envelope(snapshot, changed)
            except Exception as exc:
                logger.warning("Live stream snapshot failed: %s", exc)
                frame = {
                    "stream": "training-live",
                    "phase": "ERROR",
                    "status": "error",
                    "data": {"error": str(exc)},
                    "message": str(exc),
                }
            yield "data: " + json.dumps(frame, default=str) + "\n\n"
            await asyncio.sleep(LIVE_STREAM_INTERVAL)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
