"""
Logging filter that captures tagged events for the dashboard.

Installs on the root logger. When a log record has a ``tag`` attribute
matching one of the watched categories, a punchy one-liner is extracted
and recorded into the EventBuffer for the CLI monitor and /dashboard/stream.

Watched tags:
    TRAIN, MODEL, INFRA, ERROR, CHAT, SOUL, START, IDLE

The filter formats concise summaries from common log patterns:
    "Loaded gpt2 (124M params)"
    "Train step 310/500 — loss 2.341"
    "Self-train started (pid 641618)"
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from domains.infrastructure.event_buffer import get_event_buffer


# Tags we capture into the dashboard event feed
_WATCHED_TAGS = frozenset({
    "TRAIN", "MODEL", "INFRA", "ERROR", "CHAT", "SOUL", "START", "IDLE",
})

# Patterns that produce punchy one-liners from log messages
_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Training step progress: "step 310/500" or "step 310 epoch 2" with loss
    (re.compile(r"step\s+(\d+)/(\d+).*?loss[:\s]+([\d.]+)", re.I),
     "TRAIN", "Train step {1}/{2} — loss {3}"),
    # Training step without loss
    (re.compile(r"step\s+(\d+)/(\d+)", re.I),
     "TRAIN", "Train step {1}/{2}"),
    # Model loaded: "Loaded gpt2" or "model loaded: gpt2"
    (re.compile(r"(?:loaded|model loaded)[:\s]+(\S+).*?(\d+[MmKk]?\s*param)", re.I),
     "MODEL", "Loaded {1} ({2})"),
    # Model loaded simple
    (re.compile(r"(?:loaded|model loaded)[:\s]+(\S+)", re.I),
     "MODEL", "Loaded {1}"),
    # Checkpoint saved
    (re.compile(r"checkpoint saved[:\s]+(\S+)", re.I),
     "TRAIN", "Checkpoint saved: {1}"),
    # Self-train started
    (re.compile(r"self-training started.*?pid=(\d+)", re.I),
     "TRAIN", "Self-train started (pid {1})"),
    # Self-train stopped
    (re.compile(r"self-training stopped", re.I),
     "TRAIN", "Self-train stopped"),
    # Auto-train complete
    (re.compile(r"auto.?train(?:ing)? complete", re.I),
     "TRAIN", "Auto-train complete"),
    # Server ready
    (re.compile(r"server ready|uvicorn running|startup complete", re.I),
     "SYSTEM", "Server ready"),
    # Memory pressure (before generic error — "out of memory" is specific)
    (re.compile(r"memory pressure|oom|out of memory", re.I),
     "ERROR", "Memory pressure detected"),
    # Error / exception (after memory — catches "failed:", "error:", etc.)
    (re.compile(r"(error|exception|failed)[:\s]*(.{10,60})", re.I),
     "ERROR", "{1}: {2}"),
    # Idle unload
    (re.compile(r"unloading.*?idle|idle.*?unload", re.I),
     "MODEL", "Model unloaded (idle)"),
    # Inference first token
    (re.compile(r"first.?token.*?(\d+)\s*ms", re.I),
     "INFERENCE", "First token latency: {1}ms"),
]


def _format_punchy(record: logging.LogRecord) -> Optional[tuple[str, str]]:
    """Extract a punchy (category, message) from a log record.

    Tries regex patterns first, then falls back to a truncated message
    for records with a matching tag.

    Returns:
        (category, message) tuple, or None if no match.
    """
    msg = record.getMessage()
    tag = getattr(record, "tag", "") or ""

    for pattern, default_cat, template in _PATTERNS:
        m = pattern.search(msg)
        if m:
            # Build message from template using match groups
            try:
                parts = template.split(" — ")
                result_parts = []
                for part in parts:
                    # Replace {1}, {2}, etc. with match groups
                    for i, group in enumerate(m.groups(), 1):
                        part = part.replace(f"{{{i}}}", group)
                    result_parts.append(part)
                return default_cat, " — ".join(result_parts)
            except Exception:
                pass

    # Fallback: use tag as category, truncate message
    if tag and tag in _WATCHED_TAGS:
        category = tag
        if len(msg) > 80:
            msg = msg[:77] + "..."
        return category, msg

    return None


class DashboardFilter(logging.Filter):
    """Logging filter that captures events for the CLI dashboard.

    Install on the root logger in setup_logging(). Non-blocking:
    event buffer writes are lock-protected and fast.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Always returns True (never drops records)."""
        try:
            tag = getattr(record, "tag", "") or ""
            if tag in _WATCHED_TAGS or record.levelno >= logging.ERROR:
                result = _format_punchy(record)
                if result:
                    category, message = result
                    get_event_buffer().record(category, message)
        except Exception:
            pass
        return True
