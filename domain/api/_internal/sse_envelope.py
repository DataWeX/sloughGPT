"""
Standard SSE Envelope — SloughGPT Streaming API

Every SSE endpoint emits events with this shape:

    {
        "stream": "auto-train",   # identifies which stream (auto-train | chat | regenerate)
        "phase": "TRAIN",          # current logical phase
        "status": "working",       # working | success | error | complete
        "data": {},                # structured payload (varies by stream/phase)
        "meta": {},                # optional diagnostic: step, elapsed_ms, epoch, etc.
        "message": "loss=1.234"    # human-readable, always present
    }

Phases (TrainingSequence):
    IDLE → GENERATE_DATA → DISTILL → TRAIN → EVALUATE → DEPLOY → COMPLETE | FAILED | EARLY_STOP

Phases (Chat sequence — CHAT_SEQUENCE):
    IDLE → PROCESSING → STREAMING → FINALIZED | ERROR | CANCELLED

    PROCESSING is the core: work running in the inference event loop (engine
    logic — context assembly, tool calls, the provider request) and its
    sub-events. STREAMING is token delivery only — tokens are not processing.
    FINALIZED is terminal and is emitted LAST, after the post-gen work
    (memory persist, session store, reflection) has finished. PROCESSING may
    reappear as per-frame labels after STREAMING (finalizing); the sequence is
    a vocabulary of per-frame labels, not a strictly linear wire order.
    CANCELLED and ERROR are terminals reachable from any non-terminal — cancel
    works from any phase.

Chat statuses (CHAT_STATUSES):
    working    — engine logic tasks in flight (context, tool calls, request)
    generating — reasoning / decode — this is what "thinking" used to be called
    streaming  — a token frame (phase=STREAMING, data={"token": ...})
    finalizing — post-gen wrap-up (memory persist, session save, reflection)
    finalized  — terminal: stream done, nothing follows
    cancelled  — terminal: stopped by the turn's cancel flag
    error      — terminal: something went wrong

Status (grandfathered streams — auto-train, generate, ...; CHAT_STATUSES
excludes these except where noted):
    working  — intermediate event, keep consuming
    success  — phase/stage succeeded (maps to previous 'done=True')
    error    — something went wrong
    complete — entire stream done, no more events follow

Token streaming shortcut:
    When phase=STREAMING and data={}, the client appends data.token to the output.
    On the chat sequence every token frame carries status=streaming (never
    working); the chat terminal is a separate FINALIZED/finalized event.
    On grandfathered streams an event with status=complete still signals end
    of stream.

Usage in routers:
    from domain.api._internal.sse_envelope import sse_event, SSEEnvelope

    yield sse_event(
        stream="auto-train",
        phase="TRAIN",
        status="working",
        data={"step": 42, "loss": 1.234, "progress": 45},
        meta={"epoch": 1, "total_epochs": 10, "elapsed_ms": 1500},
        message="loss=1.2340",
    )

    yield sse_event(stream="chat", phase="PROCESSING", status="generating",
                    data={}, message="Generating")

    yield sse_event(stream="chat", phase="STREAMING", status="streaming",
                    data={"token": "Hello"}, message="")

    yield sse_event(stream="chat", phase="FINALIZED", status="finalized",
                    data={}, meta={"elapsed_ms": 890}, message="")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

TRAINING_SEQUENCE = [
    "IDLE",
    "GENERATE_DATA",
    "DISTILL",
    "TRAIN",
    "EVALUATE",
    "DEPLOY",
    "COMPLETE",
    "FAILED",
    "EARLY_STOP",
]

CHAT_SEQUENCE = [
    "IDLE",
    "PROCESSING",
    "STREAMING",
    "FINALIZED",
    "ERROR",
    "CANCELLED",
]

CHAT_STATUSES = [
    "working",
    "generating",
    "streaming",
    "finalizing",
    "finalized",
    "cancelled",
    "error",
]


class StreamPhase(Enum):
    IDLE = "IDLE"
    GENERATE_DATA = "GENERATE_DATA"
    DISTILL = "DISTILL"
    TRAIN = "TRAIN"
    EVALUATE = "EVALUATE"
    DEPLOY = "DEPLOY"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    EARLY_STOP = "EARLY_STOP"
    STREAMING = "STREAMING"
    ERROR = "ERROR"
    PROCESSING = "PROCESSING"
    FINALIZED = "FINALIZED"
    CANCELLED = "CANCELLED"


class StreamStatus(Enum):
    WORKING = "working"
    SUCCESS = "success"
    ERROR = "error"
    COMPLETE = "complete"
    GENERATING = "generating"
    STREAMING = "streaming"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    CANCELLED = "cancelled"


@dataclass
class SSEEnvelope:
    stream: str
    phase: str
    status: StreamStatus | str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        status = self.status.value if isinstance(self.status, StreamStatus) else self.status
        result = {
            "stream": self.stream,
            "phase": self.phase,
            "status": status,
            "message": self.message,
            "data": self.data,
            "meta": self.meta,
        }
        if self.id is not None:
            result["id"] = self.id
        return result


def sse_event(
    stream: str,
    phase: str,
    status: StreamStatus | str,
    data: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    message: str = "",
    id: str | None = None,
) -> str:
    """
    Build a standard SSE data line.

    Args:
        stream:   "auto-train" | "chat" | "regenerate" | "eval" | ...
        phase:    current phase name (chat sequence: see CHAT_SEQUENCE)
        status:   frame status — "working" | "success" | "error" | "complete"
                  for grandfathered streams, or a CHAT_STATUSES value on the
                  chat sequence
        data:     structured payload (token, loss, step, etc.)
        meta:     diagnostic info (step, epoch, elapsed_ms, etc.)
        message:  human-readable one-liner
        id:       optional event ID for Last-Event-ID reconnection

    Returns:
        SSE data line string: "data: <json>\n\n"
    """
    import json

    env = SSEEnvelope(
        stream=stream,
        phase=phase,
        status=status,
        message=message,
        data=data or {},
        meta=meta or {},
        id=id,
    )
    return "data: " + json.dumps(env.to_dict(), default=_json_safe) + "\n\n"


def _json_safe(val: Any) -> Any:
    """Coerce non-serializable numpy/Python types to JSON-safe equivalents."""
    if hasattr(val, "tolist"):
        return val.tolist()
    if hasattr(val, "item"):
        return val.item()
    return val


def sse_error(
    stream: str,
    phase: str,
    error: str,
    meta: dict[str, Any] | None = None,
    code: str | None = None,
    http_status: int | None = None,
) -> str:
    """Convenience: emit an error event.

    Args:
        stream:     stream name (e.g. "chat")
        phase:      phase name (e.g. "TIMEOUT", "IDLE")
        error:      human-readable error message
        meta:       optional diagnostic metadata
        code:       structured error code (e.g. "MODEL_TIMEOUT", "E_VAL_REQUEST")
        http_status: suggested HTTP status code for the client (e.g. 503, 400)
    """
    data: dict[str, Any] = {"error": error}
    if code is not None:
        data["code"] = code
    if http_status is not None:
        data["http_status"] = http_status
    return sse_event(
        stream=stream,
        phase=phase,
        status="error",
        data=data,
        meta=meta or {},
        message=f"Error: {error}",
    )


def sse_complete(
    stream: str,
    phase: str = "COMPLETE",
    data: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    message: str = "Done",
) -> str:
    """Convenience: emit a completion event."""
    return sse_event(
        stream=stream,
        phase=phase,
        status="complete",
        data=data or {},
        meta=meta or {},
        message=message,
    )


def sse_token(
    stream: str,
    token: str,
    done: bool = False,
    meta: dict[str, Any] | None = None,
    elapsed_ms: float | None = None,
    status: str | None = None,
) -> str:
    """
    Token streaming shortcut. phase=STREAMING.

    Args:
        stream:     "chat" | "regenerate" | ...
        token:      the token text (empty string signals done)
        done:       True if this is the final event (grandfathered streams)
        meta:       optional diagnostics
        elapsed_ms: total elapsed time in ms (included on done event)
        status:     explicit frame status. The chat sequence passes
                    "streaming" so the legacy default never labels a chat
                    frame; omitted, the grandfathered default applies
                    (working while streaming, complete when done). Chat's
                    terminal is a separate FINALIZED/finalized event, not a
                    done token.
    """
    phase = "STREAMING"
    if status is None:
        status = "complete" if done else "working"
    m = meta or {}
    if done and elapsed_ms is not None:
        m["elapsed_ms"] = round(elapsed_ms, 1)

    return sse_event(
        stream=stream,
        phase=phase,
        status=status,
        data={"token": token},
        meta=m,
        message="",
    )
