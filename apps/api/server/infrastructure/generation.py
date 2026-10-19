"""Generation engine port — the read interface between the chat surface and
the inference engine.

Dependency direction (the whole point of this module):

    routers/chat.py      ──imports──▶  infrastructure.generation (this port)
    routers/inference.py ──registers─▶ infrastructure.generation (this port)

Neither router imports the other. Chat *reads* the engine's output — tokens
plus the details they arrive with (model, usage, finish reason, timing) — as
plain data, then decides how to handle the response for the frontend. Inference
*implements* the engine and registers itself here at boot.

The engine's public seam today has two methods. ``respond`` is the primary,
non-streaming read: it returns a ``TurnResult`` carrying the full details. The
streaming side is currently exposed as ``stream_sse`` — the engine's SSE
projection, kept here so the chat router reaches the engine through this port
rather than importing a sibling router. A follow-on change replaces
``stream_sse`` with a raw-token ``stream`` generator and moves SSE framing into
the chat handler (the "stream is default, chat handles the frontend" target);
until then, framing still lives in the engine and this port is the seam that
makes that swap a local change.

This is an app-layer seam (it imports the wire contracts from ``schemas``), so
it lives in ``infrastructure`` — neutral territory both routers may depend on —
not in ``domain`` (which stays app-import-free).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from schemas.chat import ChatRequest


@dataclass(slots=True)
class TurnResult:
    """Everything the inference engine pushes out for one generation turn.

    This is the "details" chat needs to read off the engine — the values the
    non-streaming path historically computed and then discarded.
    """

    text: str
    session_id: str
    done: bool = True
    model: str | None = None
    usage: dict = field(default_factory=dict)
    elapsed_ms: float | None = None
    finish_reason: str | None = None
    tokens: list[str] = field(default_factory=list)


@runtime_checkable
class ChatEngine(Protocol):
    """The generation engine's public seam.

    ``respond`` returns the folded result plus details. ``stream_sse`` is the
    engine's SSE projection for the streaming route (transitional — a follow-on
    turns it into a raw-token generator the chat handler frames itself).
    """

    async def respond(self, req: ChatRequest) -> TurnResult:
        """Run one generation turn and return the folded result + details."""
        ...

    async def stream_sse(self, req: ChatRequest, request: Any, auth_user: Any) -> Any:
        """Run one generation turn as an SSE response (engine-owned framing)."""
        ...


_engine: ChatEngine | None = None


def register_chat_engine(engine: ChatEngine) -> None:
    """Register the inference-backed generation engine (called once at boot)."""
    global _engine
    _engine = engine


def get_chat_engine() -> ChatEngine:
    """Return the registered generation engine.

    Raises if nothing is registered — surfaces a wiring error at the call site
    rather than a silent ``None`` deep in the chat handler.
    """
    if _engine is None:
        raise RuntimeError(
            "No chat generation engine registered. "
            "routers.inference must register one at import/boot."
        )
    return _engine
