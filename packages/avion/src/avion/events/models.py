"""Event model — what happened during an Arken session."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
        d: dict[str, Any] = {
            "seq": self.seq,
            "type": self.type.value,
            "name": self.name,
            "success": self.success,
            "timestamp": self.timestamp,
        }
        if self.data:
            d["data"] = {k: v for k, v in self.data.items() if k != "screenshot"}
            if "screenshot" in self.data:
                d["has_screenshot"] = True
        if self.error:
            d["error"] = self.error
        if self.duration_ms:
            d["duration_ms"] = round(self.duration_ms, 2)
        return d
