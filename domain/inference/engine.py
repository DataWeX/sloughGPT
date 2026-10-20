"""Inference engine port — the chat surface's read interface to the engine.

Dependency direction (the whole point of this module):

    routers/chat.py      ──imports──▶  domain.inference (this port)
    routers/inference.py ──registers─▶ domain.inference (this port)

Neither router imports the other. This is the inference *capability*, so it
lives in the inference domain alongside its adapter (routers/inference.py is
the domain's model-facing implementation). It stays app-import-free: no
pydantic, no FastAPI — the wire contracts live in ``schemas`` and the
transport (SSE framing) lives at the edge; both translate at the boundary,
never into the core.

The contract is stream-only. ``stream`` is *the* inference method — it yields
raw tokens, then a final ``TurnResult`` carrying the details. There is no
separate non-streaming backend and no ``stream: bool`` mode: streaming is not a
choice, it is how the model emits tokens. A non-streaming surface is simply a
fold of this stream performed at the edge (collect the tokens, return the last
``TurnResult``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Awaitable, Protocol, runtime_checkable


@dataclass(slots=True)
class TurnRequest:
    """Inference inputs, domain-native (no wire coupling).

    The edge (routers) translates its pydantic ``ChatRequest`` into this.
    """

    messages: list[dict]
    model: str = "qwen2.5-0.5b-instruct"
    temperature: float = 0.7
    max_tokens: int = 128
    top_p: float = 0.85
    top_k: int = 40
    repetition_penalty: float = 1.15
    session_id: str = "default"
    user_id: str = "default"
    system_prompt: str | None = None
    knowledge: list[str] | None = None
    images: list[str] | None = None
    agent_id: str | None = None
    use_context_core: bool = True
    use_rag: bool = True
    model_fields_set: set[str] = field(default_factory=set)


@dataclass(slots=True)
class TurnResult:
    """Everything the engine pushes out for one turn — the "details" the chat
    surface reads off the output (values the old handler computed and
    discarded)."""

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
    """The inference engine's public seam.

    ``stream`` is the single inference method: it yields raw tokens, then a
    final ``TurnResult``. Non-streaming surfaces fold this stream at the edge.
    """

    def stream(self, req: TurnRequest) -> AsyncIterator[TurnResult | str]: ...


def fold(stream: AsyncIterator[TurnResult | str]) -> Awaitable[TurnResult]:
    """Collect a token stream into its final ``TurnResult``.

    This is *the* non-streaming path — not a second backend, just a fold of
    ``stream``. Edges that need a complete response (e.g. the mobile BFF's
    offline sync) call this; nothing here runs the model independently.
    """

    async def _collect() -> TurnResult:
        result: TurnResult | None = None
        async for item in stream:
            if isinstance(item, TurnResult):
                result = item
        if result is None:
            raise RuntimeError("engine stream ended without a final TurnResult")
        return result

    return _collect()


_engine: ChatEngine | None = None


def register_chat_engine(engine: ChatEngine) -> None:
    """Register the inference-backed engine (once, at boot)."""
    global _engine
    _engine = engine


def get_chat_engine() -> ChatEngine:
    """Return the registered engine (raises if unwired)."""
    if _engine is None:
        raise RuntimeError(
            "No chat engine registered. routers.inference must register one at boot."
        )
    return _engine
