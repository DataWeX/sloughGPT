"""Event model — what happened during an Arken session."""

from __future__ import annotations

import base64
import binascii
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _encode_bytes(data: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe copy of an event's data: bytes values become base64 strings.

    Without this, ``json.dump(..., default=str)`` mangles pixel bytes into
    ``"b'\\x89PN..'"`` — legal JSON, unrecoverable pixels. Base64 keeps the
    interchange real JSON AND lossless (the same rule ``FlowStep.to_dict``
    applies for per-step screenshots).
    """
    if not any(isinstance(v, (bytes, bytearray)) for v in data.values()):
        return data
    return {
        k: base64.b64encode(v).decode("ascii") if isinstance(v, (bytes, bytearray)) else v
        for k, v in data.items()
    }


def _restore_screenshot(data: dict[str, Any], flagged: bool) -> dict[str, Any]:
    """Rehydrate a base64 ``screenshot`` back to bytes (flag-gated).

    The ``has_screenshot`` flag gates the decode so a plain string that
    merely *looks* like base64 never gets mangled, and files written before
    pixels persisted (flag, no key) load exactly as they did. Invalid
    base64 keeps the string as-is — degrade, never raise.
    """
    if not flagged or "screenshot" not in data:
        return data
    shot = data.get("screenshot")
    if not isinstance(shot, str):
        return data
    try:
        raw = base64.b64decode(shot, validate=True)
    except (binascii.Error, ValueError):
        return data
    out = dict(data)
    out["screenshot"] = raw
    return out


def strip_screenshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop screenshot pixels from a to_dict payload, keeping the flag.

    Delivery targets that must never carry megabytes (the stdlib log bridge)
    strip here; durable stores (journal, ``save()``) keep the pixels.
    """
    data = payload.get("data")
    if isinstance(data, dict) and "screenshot" in data:
        payload = dict(payload)
        payload["data"] = {k: v for k, v in data.items() if k != "screenshot"}
        payload["has_screenshot"] = True
    return payload


class EventType(Enum):
    """Kinds of events Arken records."""

    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    NAVIGATE = "navigate"
    ELEMENT_FOUND = "element_found"
    ELEMENT_NOT_FOUND = "element_not_found"
    CLICK = "click"
    FILL = "fill"
    KEY_PRESS = "key_press"
    SELECT = "select"
    SCREENSHOT = "screenshot"
    TASK_STARTED = "task_started"
    TASK_PASSED = "task_passed"
    TASK_FAILED = "task_failed"
    CUSTOM = "custom"


@dataclass
class Event:
    """A single recorded happening."""

    type: EventType
    name: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str = ""
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON interchange form — lossless, including screenshot pixels.

        Byte values (screenshots) are base64 strings here, so ``save()`` and
        the journal persist the pixels instead of only the presence flag;
        ``has_screenshot`` is kept as an additive marker for readers written
        against the old flag-only contract.
        """
        d: dict[str, Any] = {
            "seq": self.seq,
            "type": self.type.value,
            "name": self.name,
            "success": self.success,
            "timestamp": self.timestamp,
        }
        if self.data:
            d["data"] = _encode_bytes(self.data)
            if "screenshot" in self.data:
                d["has_screenshot"] = True
        if self.error:
            d["error"] = self.error
        if self.duration_ms:
            d["duration_ms"] = round(self.duration_ms, 2)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Event:
        """Rebuild an event from its dict form (the to_dict inverse).

        Screenshots round-trip as base64 strings gated by the
        ``has_screenshot`` flag; the flag itself never becomes data — files
        written before pixels persisted keep loading flag-only.
        """
        data = dict(d.get("data") or {})
        return cls(
            type=EventType(d["type"]),
            name=d.get("name", ""),
            data=_restore_screenshot(data, bool(d.get("has_screenshot"))),
            success=bool(d.get("success", True)),
            error=d.get("error", ""),
            duration_ms=float(d.get("duration_ms", 0.0)),
            timestamp=float(d.get("timestamp", 0.0)),
            seq=int(d.get("seq", 0)),
        )
