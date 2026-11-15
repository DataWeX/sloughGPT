"""Agent routines — the background routine status stream (agent phase 2).

One descriptor (``SPEC``), one projection: verb, auth, validation and OpenAPI
metadata are all derived from it, and boot registration happens by importing
this module via the router manifest — see ``docs/TRANSPORT_PROJECTIONS.md``.

Deliberately its own module rather than a route inside the grandfathered
``routers/agents.py``: the declaration ratchet
(``handbuilt-routes-baseline.json``) forbids new hand-built route sites, and
nesting an ``APIRouter(prefix="/agents")`` inside agents' own prefixed router
would double the prefix on this FastAPI version (``/agents/agents/...``).
Mounted at ``prefix="/agents"`` from here, the projection lands on exactly
``GET /agents/routines/stream``.

Tier: internal/model-facing? No — this is a site read path of the app (it
projects the routine runner's event log to an operator/client surface); it
never enters the chat tool registry, ``domain/tools``, or anything the model
reasons over.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

from fastapi import Request
from fastapi.responses import StreamingResponse
from infrastructure.contract import RouteSpec, create_router

from domain.agents import ToolSpec, get_routine_runner
from domain.api import sse_complete, sse_error, sse_event

logger = logging.getLogger("slo.routers.agent_routines")

# Poll cadence for the routine status stream. Short enough that a client
# sees a run start promptly, long enough not to spin the event loop.
_ROUTINE_POLL_SECONDS = 0.5


async def routines_stream(
    request: Request,
    auth_user: dict | None = None,
) -> StreamingResponse:
    """Stream background routine status: snapshot, then live run events.

    The runner records every event its ``agent_run`` handler observes;
    this projects that log as SSE. Each frame carries a monotonic
    ``seq`` so a reconnecting client can resume without duplicates.
    """
    runner = get_routine_runner()
    phases = {
        "run_start": ("START", "working"),
        "tool_start": ("TOOL_CALL", "working"),
        "tool_result": ("TOOL_RESULT", "working"),
        "final": ("FINAL", "working"),
        "run_end": ("COMPLETE", "complete"),
        "error": ("ERROR", "error"),
    }

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            routines = [r.to_dict() for r in runner.store.list()]
            yield sse_event(
                stream="agent-routines",
                phase="SNAPSHOT",
                status="working",
                data={"routines": routines, "running": runner.running},
                message=f"{len(routines)} routine(s)",
            )
            cursor = 0
            while True:
                if await request.is_disconnected():
                    break
                for event in runner.events_since(cursor):
                    cursor = max(cursor, int(event.get("seq", 0)))
                    kind = str(event.get("event", "step"))
                    phase, status = phases.get(kind, (kind.upper(), "working"))
                    if kind == "error":
                        yield sse_error(
                            "agent-routines",
                            phase,
                            str(event.get("error", "unknown")),
                        )
                        continue
                    yield sse_event(
                        stream="agent-routines",
                        phase=phase,
                        status=status,
                        data=event,
                        id=str(event.get("seq", "")),
                    )
                await asyncio.sleep(_ROUTINE_POLL_SECONDS)
            yield sse_complete(stream="agent-routines", data={"running": runner.running})
        except Exception as e:
            logger.warning("Routine stream failed: %s", e, extra={"tag": "AGENT"})
            yield sse_error("agent-routines", "ERROR", str(e))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


SPEC = ToolSpec(
    name="agents.routines_stream",
    description=(
        "Stream background routine status as SSE: a routine snapshot first, "
        "then live run events, each carrying a monotonic seq for resume."
    ),
    parameters=[],
    execute=routines_stream,
    auth_scope="authenticated",
    idempotent=True,
    result={
        "type": "object",
        "properties": {"stream": {"type": "string"}},
    },
    version="1",
)

router = create_router(
    SPEC,
    RouteSpec(
        path="/routines/stream",
        method="GET",
        summary="Stream background routine status (SSE)",
        envelope=False,  # raw wire: StreamingResponse passthrough
    ),
    routines_stream,
    prefix="/agents",
    tags=("agents",),
)
