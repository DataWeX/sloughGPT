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

Phases (Chat / token streams):
    IDLE → STREAMING → COMPLETE | ERROR

Status:
    working  — intermediate event, keep consuming
    success  — phase/stage succeeded (maps to previous 'done=True')
    error    — something went wrong
    complete — entire stream done, no more events follow

Token streaming shortcut:
    When phase=STREAMING and data={}, the client appends data.token to the output.
    An event with status=complete signals end of stream.

Usage in routers:
    from domains.api.sse_envelope import sse_event, SSEEnvelope

    yield sse_event(
        stream="auto-train",
        phase="TRAIN",
        status="working",
        data={"step": 42, "loss": 1.234, "progress": 45},
        meta={"epoch": 1, "total_epochs": 10, "elapsed_ms": 1500},
        message="loss=1.2340",
    )

    yield sse_event(stream="chat", phase="STREAMING", status="working",
                    data={"token": "Hello"}, message="")

    yield sse_event(stream="chat", phase="STREAMING", status="complete",
                    data={}, meta={"elapsed_ms": 890}, message="done")
"""

from typing import Any, Optional, Dict, Union, Literal
from dataclasses import dataclass, field
from enum import Enum


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
    "STREAMING",
    "COMPLETE",
    "ERROR",
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


class StreamStatus(Enum):
    WORKING = "working"
    SUCCESS = "success"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class SSEEnvelope:
    stream: str
    phase: str
    status: Union[StreamStatus, str]
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        status = self.status.value if isinstance(self.status, StreamStatus) else self.status
        return {
            "stream": self.stream,
            "phase": self.phase,
            "status": status,
            "message": self.message,
            "data": self.data,
            "meta": self.meta,
        }


def sse_event(
    stream: str,
    phase: str,
    status: Union[StreamStatus, str],
    data: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    message: str = "",
) -> str:
    """
    Build a standard SSE data line.

    Args:
        stream:   "auto-train" | "chat" | "regenerate" | "eval" | ...
        phase:    current phase name
        status:   "working" | "success" | "error" | "complete"
        data:     structured payload (token, loss, step, etc.)
        meta:     diagnostic info (step, epoch, elapsed_ms, etc.)
        message:  human-readable one-liner

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
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Convenience: emit an error event."""
    return sse_event(
        stream=stream,
        phase=phase,
        status="error",
        data={"error": error},
        meta=meta or {},
        message=f"Error: {error}",
    )


def sse_complete(
    stream: str,
    phase: str = "COMPLETE",
    data: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
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
    meta: Optional[Dict[str, Any]] = None,
    elapsed_ms: Optional[float] = None,
) -> str:
    """
    Token streaming shortcut. phase=STREAMING, status=complete when done.

    Args:
        stream:    "chat" | "regenerate"
        token:     the token text (empty string signals done)
        done:      True if this is the final event
        meta:      optional diagnostics
        elapsed_ms: total elapsed time in ms (included on done event)
    """
    phase = "STREAMING"
    if done:
        status = "complete"
        m = meta or {}
        if elapsed_ms is not None:
            m["elapsed_ms"] = round(elapsed_ms, 1)
    else:
        status = "working"
        m = meta or {}

    return sse_event(
        stream=stream,
        phase=phase,
        status=status,
        data={"token": token},
        meta=m,
        message="",
    )
