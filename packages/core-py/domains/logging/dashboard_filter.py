"""
Logging filter that captures tagged events for the dashboard.

Installs on the root logger. When a log record has an ``op`` attribute
or ``tag`` attribute matching one of the watched categories, a punchy
one-liner is extracted and recorded into the EventBuffer for the CLI
monitor and /dashboard/stream.

Watched ops (slo.log v1):
    train.*, model.*, infer.*, http.request, rag.*, download.*,
    workflow.*, sys.startup, infra.*

Watched tags (legacy fallback):
    TRAIN, MODEL, INFRA, ERROR, CHAT, SOUL, START, IDLE, DOWNLOAD, SLOW

The filter formats concise summaries from common log patterns:
    "Loaded gpt2 (124M params)"
    "Train step 310/500 - loss 2.341"
    "Self-train started (pid 641618)"
    "Downloaded 45% - 120MB/267MB"
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from domains.infrastructure.event_buffer import get_event_buffer


_WATCHED_TAGS = frozenset({
    "TRAIN", "MODEL", "INFRA", "ERROR", "CHAT", "SOUL", "START", "IDLE",
    "DOWNLOAD", "SLOW", "INFERENCE", "WORKFLOW",
})

# slo.log v1: op prefix -> dashboard category
_WATCHED_OPS = {
    "train":     "TRAIN",
    "model":     "MODEL",
    "infer":     "INFERENCE",
    "http":      "INFRA",
    "rag":       "COG",
    "download":  "DOWNLOAD",
    "workflow":  "WORKFLOW",
    "sys":       "SYSTEM",
    "infra":     "INFRA",
    "web":       "CHAT",
}

_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # -- Training ----------------------------------------------------------
    # Step with loss: "step 310/500 - loss 2.341"
    (re.compile(r"step\s+(\d+)/(\d+).*?loss[:\s]+([\d.]+)", re.I),
     "TRAIN", "Train step {1}/{2} - loss {3}"),
    # Step without loss
    (re.compile(r"step\s+(\d+)/(\d+)", re.I),
     "TRAIN", "Train step {1}/{2}"),
    # Epoch progress: "epoch 2/10"
    (re.compile(r"epoch\s+(\d+)/(\d+)", re.I),
     "TRAIN", "Epoch {1}/{2}"),
    # Training complete
    (re.compile(r"train(?:ing)?\s+(?:complete|finished|done)", re.I),
     "TRAIN", "Training complete"),
    # Training started
    (re.compile(r"train(?:ing)?\s+started", re.I),
     "TRAIN", "Training started"),
    # Training failed
    (re.compile(r"train(?:ing)?\s+(?:failed|error)", re.I),
     "TRAIN", "Training failed"),
    # Checkpoint saved: "checkpoint saved: ep3.soul"
    (re.compile(r"checkpoint\s+saved[:\s]+(\S+)", re.I),
     "TRAIN", "Checkpoint saved: {1}"),
    # Distillation
    (re.compile(r"distill(?:ation)?\s+(?:complete|finished)", re.I),
     "TRAIN", "Distillation complete"),
    # Eval result
    (re.compile(r"eval.*?loss[:\s]+([\d.]+)", re.I),
     "TRAIN", "Eval loss: {1}"),
    # Auto-train complete
    (re.compile(r"auto.?train(?:ing)?\s+complete", re.I),
     "TRAIN", "Auto-train complete"),
    # Auto-train started
    (re.compile(r"auto.?train(?:ing)?\s+started", re.I),
     "TRAIN", "Auto-train started"),
    # Self-train started
    (re.compile(r"self.?train(?:ing)?\s+started.*?pid[=\s]+(\d+)", re.I),
     "TRAIN", "Self-train started (pid {1})"),
    # Self-train stopped
    (re.compile(r"self.?train(?:ing)?\s+stopped", re.I),
     "TRAIN", "Self-train stopped"),

    # -- Model -------------------------------------------------------------
    # Model loaded with params: "loaded gpt2 (124M params)"
    (re.compile(r"(?:loaded|model loaded|loading model)[:\s]+(\S+).*?(\d+[MmKk]?\s*param)", re.I),
     "MODEL", "Loaded {1} ({2})"),
    # Model loaded simple
    (re.compile(r"(?:loaded|model loaded)[:\s]+(\S+)", re.I),
     "MODEL", "Loaded {1}"),
    # Model unloaded
    (re.compile(r"(?:unloaded|unloading)[:\s]+(\S+)", re.I),
     "MODEL", "Unloaded {1}"),
    # Idle unload
    (re.compile(r"unloading.*?idle|idle.*?unload", re.I),
     "MODEL", "Model unloaded (idle)"),
    # Model swap
    (re.compile(r"model\s+sw(?:ap|itched)\s+(?:to|from)\s+(\S+)", re.I),
     "MODEL", "Model swapped to {1}"),
    # Soul loaded
    (re.compile(r"soul\s+loaded[:\s]+(\S+)", re.I),
     "MODEL", "Soul loaded: {1}"),
    # Soul switched
    (re.compile(r"soul\s+switched\s+to[:\s]+(\S+)", re.I),
     "MODEL", "Soul switched: {1}"),

    # -- Inference ---------------------------------------------------------
    # First token latency
    (re.compile(r"first.?token.*?(\d+)\s*ms", re.I),
     "INFERENCE", "First token: {1}ms"),
    # Generate complete
    (re.compile(r"generate.*?(\d+)\s*tokens?\s*in\s*([\d.]+)\s*s", re.I),
     "INFERENCE", "Generated {1} tokens in {2}s"),
    # Stream stall
    (re.compile(r"stream\s+stall", re.I),
     "INFERENCE", "Stream stall detected"),
    # Client disconnect
    (re.compile(r"client\s+disconnect", re.I),
     "INFERENCE", "Client disconnected"),

    # -- System ------------------------------------------------------------
    # Server ready
    (re.compile(r"server\s+ready|uvicorn\s+running|startup\s+complete", re.I),
     "SYSTEM", "Server ready"),
    # Idle manager
    (re.compile(r"idle\s+manager\s+active", re.I),
     "SYSTEM", "Idle manager active"),
    # Cancel
    (re.compile(r"cancel(?:led|ing)?", re.I),
     "SYSTEM", "Operation cancelled"),

    # -- Download ----------------------------------------------------------
    # Download progress: "downloaded 45%" or "45% - 120MB/267MB"
    (re.compile(r"download.*?(\d+)%.*?(\d+\.?\d*)\s*[MmGg].*?/.*?(\d+\.?\d*)\s*[MmGg]", re.I),
     "DOWNLOAD", "Download {1}% - {2}/{3}MB"),
    (re.compile(r"download.*?(\d+)%", re.I),
     "DOWNLOAD", "Download {1}%"),
    # Download complete
    (re.compile(r"download\s+complete", re.I),
     "DOWNLOAD", "Download complete"),
    # Download failed
    (re.compile(r"download\s+(?:failed|error)", re.I),
     "DOWNLOAD", "Download failed"),

    # -- Workflow ----------------------------------------------------------
    # Feedback workflow
    (re.compile(r"feedback\s+workflow\s+(?:started|stopped|complete)", re.I),
     "WORKFLOW", "Feedback workflow {1}"),
    # Webhook
    (re.compile(r"webhook\s+(?:sent|failed|notification)", re.I),
     "WORKFLOW", "Webhook {1}"),

    # -- Errors (last - catch-all) -----------------------------------------
    # Memory pressure
    (re.compile(r"memory\s+pressure|oom|out\s+of\s+memory", re.I),
     "ERROR", "Memory pressure"),
    # Generic error with context
    (re.compile(r"(error|exception|failed)[:\s]+(.{8,60})", re.I),
     "ERROR", "{1}: {2}"),
    # Slow request
    (re.compile(r"slow\s+request|SLOW.*?(\d+\.?\d*)s", re.I),
     "SLOW", "Slow request ({1}s)"),
]


def _summarize_from_op(record: logging.LogRecord, op: str) -> Optional[tuple[str, str]]:
    """Build a punchy summary from slo.log v1 structured fields.

    Returns (category, message) or None if no good summary can be built.
    """
    domain = op.split(".")[0] if "." in op else op
    category = _WATCHED_OPS.get(domain)
    if not category:
        return None

    # Try to build a meaningful one-liner from domain payload
    msg = record.getMessage()

    if domain == "train":
        step = getattr(record, "step", None)
        total = getattr(record, "total_steps", None)
        loss = getattr(record, "loss", None)
        if step and total:
            base = f"Train step {step}/{total}"
            if loss is not None:
                base += f" - loss {loss}"
            return category, base

    if domain == "model":
        model_id = getattr(record, "id", None) or getattr(record, "model_id", None)
        if model_id:
            return category, f"Model {model_id} - {op}"

    if domain == "infer":
        tokens = getattr(record, "tokens", None)
        model_id = getattr(record, "model_id", None)
        if tokens and model_id:
            return category, f"Generated {tokens} tokens ({model_id})"

    if domain == "download":
        resource = getattr(record, "resource", None)
        elapsed = getattr(record, "elapsed_s", None)
        if resource:
            base = f"Downloaded {resource}"
            if elapsed:
                base += f" ({elapsed:.1f}s)"
            return category, base

    if domain == "http":
        method = getattr(record, "method", None)
        path = getattr(record, "path", None)
        status = getattr(record, "status", None)
        if method and path:
            return category, f"{method} {path} {status or ''}"

    if domain == "rag":
        results = getattr(record, "results", None)
        if results is not None:
            return category, f"RAG query - {results} results"

    if domain == "sys":
        phase = getattr(record, "phase", None)
        if phase:
            return category, f"System {phase}"

    # Fallback: truncate message
    if len(msg) > 80:
        msg = msg[:77] + "..."
    return category, msg


def _format_punchy(record: logging.LogRecord) -> Optional[tuple[str, str]]:
    """Extract a punchy (category, message) from a log record."""
    # Check slo.log v1 op first
    op = getattr(record, "op", None)
    if op:
        result = _summarize_from_op(record, op)
        if result:
            return result

    # Fallback: legacy tag + regex patterns
    msg = record.getMessage()
    tag = getattr(record, "tag", "") or ""

    for pattern, default_cat, template in _PATTERNS:
        m = pattern.search(msg)
        if m:
            try:
                parts = template.split(" - ")
                result_parts = []
                for part in parts:
                    for i, group in enumerate(m.groups(), 1):
                        part = part.replace(f"{{{i}}}", group or "")
                    result_parts.append(part)
                return default_cat, " - ".join(result_parts)
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
            # slo.log v1: check op field
            op = getattr(record, "op", None)
            if op:
                domain = op.split(".")[0] if "." in op else op
                if domain in _WATCHED_OPS or record.levelno >= logging.ERROR:
                    result = _format_punchy(record)
                    if result:
                        category, message = result
                        get_event_buffer().record(category, message)
                        return True

            # Legacy: check tag field
            tag = getattr(record, "tag", "") or ""
            if tag in _WATCHED_TAGS or record.levelno >= logging.ERROR:
                result = _format_punchy(record)
                if result:
                    category, message = result
                    get_event_buffer().record(category, message)
        except Exception:
            pass
        return True
