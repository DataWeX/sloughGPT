"""Calendar HTTP projection — the retired Next ``/api/calendar/events`` handler, ported.

Parity contract (card ``51cd0e65``): same JSONL store the Next handler used
(``<project root>/.calendar/events.jsonl``, override ``APP_PLANNER_CALENDAR_DIR``),
same event shape and create defaults (``start_time`` 09:00, ``end_time`` 10:00,
``color`` primary, id ``event-<epoch-ms>-<rand>``), same ``?date=`` filter,
same response payloads (``{events}`` / ``{event}``) with creates answering 201.

Writes go through ``os.replace`` (atomic) instead of a plain truncate-write —
readers never observe a half-written journal. No other behavior is invented:
calendar events are plain documents, deliberately outside the board's
chain/OCC machinery.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app_planner import config as planner_config
from fastapi import APIRouter, Request
from infrastructure.contract import RouteSpec, create_router  # noqa: E402

from domain.agents import ToolParam, ToolSpec  # noqa: E402

_WS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_EVENT_KEYS = ("id", "title", "date")


def _events_file(request: Request) -> Path:
    """``<root>/.calendar[/ws]/events.jsonl`` — env override wins, then workspace."""
    env = os.environ.get("APP_PLANNER_CALENDAR_DIR")
    if env:
        root = Path(env)
    else:
        base = planner_config.find_project_root() or Path.cwd()
        ws = request.headers.get("x-workspace-id", "")
        root = base / ".calendar" / ws if ws and _WS_RE.match(ws) else base / ".calendar"
    return root / "events.jsonl"


def _read_events(path: Path, date: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # Next parity: malformed lines are skipped
        if not isinstance(obj, dict) or not all(obj.get(k) for k in _EVENT_KEYS):
            continue
        if date is not None and obj.get("date") != date:
            continue
        events.append(obj)
    return events


def _write_events(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _create_event(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone().isoformat()
    event: dict[str, Any] = {
        "id": f"event-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}",
        "title": data["title"],
        "description": data.get("description") or "",
        "date": data["date"],
        "start_time": data.get("start_time") or "09:00",
        "end_time": data.get("end_time") or "10:00",
        "color": data.get("color") or "primary",
        "created_at": now,
        "updated_at": now,
    }
    events = _read_events(path)
    events.append(event)
    _write_events(path, events)
    return event


# ── Handlers ────────────────────────────────────────────────────────────────


async def read_events(request: Request, date: str | None = None) -> dict[str, Any]:
    path = _events_file(request)

    def work() -> dict[str, Any]:
        return {"events": _read_events(path, date)}

    return await asyncio.to_thread(work)


async def create_event(
    request: Request,
    title: str,
    date: str,
    description: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    color: str | None = None,
) -> dict[str, Any]:
    path = _events_file(request)
    data: dict[str, Any] = {
        "title": title,
        "date": date,
        "description": description,
        "start_time": start_time,
        "end_time": end_time,
        "color": color,
    }

    def work() -> dict[str, Any]:
        return {"event": _create_event(path, data)}

    return await asyncio.to_thread(work)


# ── Descriptors + projection ────────────────────────────────────────────────

SPEC_LIST = ToolSpec(
    name="calendar.events.read",
    description="List calendar events, optionally filtered by ISO date.",
    version="1",
    parameters=[ToolParam("date", "string", "Only events on this date (YYYY-MM-DD).")],
    execute=read_events,
    auth_scope="authenticated",
    idempotent=True,
    result={"type": "object", "properties": {"events": {"type": "array", "items": {"type": "object"}}}},
)

SPEC_CREATE = ToolSpec(
    name="calendar.events.create",
    description="Create a calendar event.",
    version="1",
    parameters=[
        ToolParam("title", "string", "Event title.", required=True),
        ToolParam("date", "string", "Event date (YYYY-MM-DD).", required=True),
        ToolParam("description", "string", "Event description."),
        ToolParam("start_time", "string", "Start time (HH:MM)."),
        ToolParam("end_time", "string", "End time (HH:MM)."),
        ToolParam("color", "string", "Display color key."),
    ],
    execute=create_event,
    auth_scope="authenticated",
    idempotent=False,
    result={"type": "object", "properties": {"event": {"type": "object"}}},
)

router = APIRouter()
router.include_router(
    create_router(
        SPEC_LIST,
        RouteSpec(path="/events", method="GET", summary="List calendar events"),
        read_events,
        prefix="/api/calendar",
        tags=("calendar",),
    )
)
router.include_router(
    create_router(
        SPEC_CREATE,
        RouteSpec(path="/events", method="POST", summary="Create a calendar event", status_code=201),
        create_event,
        prefix="/api/calendar",
        tags=("calendar",),
    )
)
