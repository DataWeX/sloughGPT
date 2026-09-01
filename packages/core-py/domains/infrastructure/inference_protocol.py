"""IPC protocol for inference engine separation.

Defines message types and wire format for communication between the API
server process and the inference engine process.

Wire format: length-prefixed JSON messages over TCP.

    uint32  big-endian payload length
    bytes   JSON payload

Message types:

    Client → Engine:
        health       — {"type": "health"}
        generate     — {"type": "generate", "id": ..., "prompt": ..., "params": {...}}
        stream_start — {"type": "stream_start", "id": ..., "prompt": ..., "params": {...}}
        stream_stop  — {"type": "stream_stop", "id": ...}
        reload       — {"type": "reload", "model_id": ..., "slnc_path": ...}

    Engine → Client:
        health_ok    — {"type": "health_ok", "model_id": ..., "loaded": bool}
        result       — {"type": "result", "id": ..., "text": ..., "meta": {...}}
        token        — {"type": "token", "id": ..., "token": ...}
        stream_done  — {"type": "stream_done", "id": ..., "meta": {...}}
        reload_ok    — {"type": "reload_ok", "model_id": ..., "elapsed": ...}
        error        — {"type": "error", "id": ..., "message": ...}
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Dict


# ── Wire helpers ──────────────────────────────────────────────────────

HEADER_FMT = "!I"  # uint32 big-endian
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def encode_message(msg: dict) -> bytes:
    """Encode a message dict to length-prefixed JSON bytes."""
    payload = json.dumps(msg, separators=(",", ":")).encode()
    return struct.pack(HEADER_FMT, len(payload)) + payload


def decode_header(data: bytes) -> int:
    """Decode payload length from header bytes."""
    return struct.unpack(HEADER_FMT, data)[0]


# ── Message types ─────────────────────────────────────────────────────

@dataclass
class HealthRequest:
    type: str = "health"


@dataclass
class HealthResponse:
    type: str = "health_ok"
    model_id: str = ""
    loaded: bool = False
    model_type: str = ""
    quantized: bool = False


@dataclass
class GenerateRequest:
    id: str = ""
    prompt: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    type: str = "generate"


@dataclass
class GenerateResult:
    id: str = ""
    text: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    type: str = "result"


@dataclass
class StreamStartRequest:
    id: str = ""
    prompt: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    type: str = "stream_start"


@dataclass
class StreamStopRequest:
    id: str = ""
    type: str = "stream_stop"


@dataclass
class StreamToken:
    id: str = ""
    token: str = ""
    type: str = "token"


@dataclass
class StreamDone:
    id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    type: str = "stream_done"


@dataclass
class ErrorResponse:
    id: str = ""
    message: str = ""
    type: str = "error"
