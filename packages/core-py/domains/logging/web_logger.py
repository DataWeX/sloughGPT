"""
WebLogger — structured JSON output for the browser frontend.

Inherits Logger and emits records as structured JSON objects that can be
consumed by browser ``console.*`` methods, sent to a logging API, or
stored in ``localStorage`` / ``IndexedDB``.

Usage::

    from domains.logging import WebLogger, LogLevel

    log = WebLogger("slo.web.chat")
    log.info("message sent", session_id="abc123")
    log.error("stream failed", exception="AbortError: timeout")

    # In browser context, log.emit() calls console.warn / console.error / console.log
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from .base import Logger, LogLevel, LogRecord


# ── Browser console mapping ────────────────────────────────────────────

_CONSOLE_METHOD = {
    LogLevel.DEBUG:    "debug",
    LogLevel.INFO:     "log",
    LogLevel.WARNING:  "warn",
    LogLevel.ERROR:    "error",
    LogLevel.CRITICAL: "error",
}


# ── Event → tag mapping (shared by WebLogger.track_event) ──────────────

_EVENT_TAG_MAP: Dict[str, str] = {
    "model_":      "MODEL",
    "training_":   "TRAIN",
    "session_":    "CHAT",
    "stream_":     "CHAT",
    "chat_":       "CHAT",
    "webhook_":    "WORKFLOW",
    "download_":   "DOWNLOAD",
    "auth_":       "AUTH",
    "vm_":         "INFRA",
    "shell_":      "INFRA",
    "soul_":       "SOUL",
}

_DEFAULT_EVENT_TAG = "UI"


class WebLogger(Logger):
    """Structured logger for browser/web environments.

    Emits records as JSON dicts.  In a browser context, delegates to
    ``console.debug / console.log / console.warn / console.error``.
    In Node.js / SSR, writes to ``stderr`` as JSON lines.

    Also handles UI event tracking via ``track_event()``, which
    auto-infers a ``LogTag`` from the event name prefix and delegates
    to ``TaggedLogger`` internally.

    Parameters:
        name:      Logger name (e.g. ``"slo.web.chat"``).
        level:     Minimum severity to emit.
        console:   Browser/console object to write to (default: ``console``).
        writable:  Writable stream for SSR (default: ``sys.stderr``).
        context:   Default context attached to every record.
    """

    __slots__ = ('_browser_console', '_writable')

    def __init__(
        self,
        name: str = "slo.web",
        level: LogLevel = LogLevel.INFO,
        context: Optional[Dict[str, Any]] = None,
        console: Any = None,
        writable: Any = None,
    ) -> None:
        super().__init__(name=name, level=level, context=context)
        self._browser_console = console
        self._writable = writable

    # ── Serialization ───────────────────────────────────────────────────

    def _record_to_dict(self, record: LogRecord) -> Dict[str, Any]:
        """Convert a LogRecord to a plain dict for JSON serialization."""
        d: Dict[str, Any] = {
            "level": record.level.value,
            "logger": record.logger,
            "message": record.message,
            "timestamp": record.timestamp,
        }
        if record.context:
            d["context"] = record.context
        if record.exception:
            d["exception"] = record.exception
        return d

    def _to_json(self, record: LogRecord) -> str:
        return json.dumps(self._record_to_dict(record), default=str)

    # ── Emit ────────────────────────────────────────────────────────────

    def emit(self, record: LogRecord) -> None:
        """Emit the record as structured output.

        - Browser: calls ``console.debug/log/warn/error`` with a
          formatted string and the structured dict.
        - Node/SSR: writes a JSON line to stderr.
        """
        method = _CONSOLE_METHOD.get(record.level, "log")
        formatted = self._format_brief(record)
        data = self._record_to_dict(record)

        with self._lock:
            # Browser console
            if self._browser_console is not None:
                fn = getattr(self._browser_console, method, None)
                if fn:
                    fn(f"[{record.logger}] {record.message}", data)
                    return

            # Node.js / SSR fallback — write JSON to stderr
            if self._writable is not None:
                try:
                    self._writable.write(self._to_json(record) + "\n")
                    self._writable.flush()
                except (OSError, ValueError):
                    pass

    @staticmethod
    def _format_brief(record: LogRecord) -> str:
        """One-line ``[logger] message`` format for console output."""
        parts = [f"[{record.logger}]"]
        if record.context:
            ctx = " ".join(f"{k}={v}" for k, v in record.context.items())
            parts.append(f"({ctx})")
        parts.append(record.message)
        if record.exception:
            parts.append(f"— {record.exception}")
        return " ".join(parts)

    # ── Web-specific helpers ────────────────────────────────────────────

    def to_json(self, record: LogRecord) -> str:
        """Serialize a record to a JSON string (for transport / storage)."""
        return self._to_json(record)

    def from_json(self, raw: str) -> LogRecord:
        """Deserialize a JSON string back into a LogRecord."""
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            return LogRecord(
                level=LogLevel.WARNING,
                message=raw,
                logger="slo",
                timestamp=time.time(),
                context={},
                exception=None,
            )
        return LogRecord(
            level=LogLevel(d.get("level", LogLevel.WARNING.value)),
            message=d.get("message", ""),
            logger=d.get("logger", "slo"),
            timestamp=d.get("timestamp", time.time()),
            context=d.get("context", {}),
            exception=d.get("exception"),
        )

    # ── Event tracking ──────────────────────────────────────────────────

    @staticmethod
    def _infer_tag(event: str) -> str:
        for prefix, tag in _EVENT_TAG_MAP.items():
            if event.startswith(prefix):
                return tag
        return _DEFAULT_EVENT_TAG

    def track_event(self, event: str, data: Optional[Dict[str, Any]] = None,
                    tag: Optional[str] = None) -> None:
        """Log a UI event with auto-inferred tag.

        The tag is resolved in order:
          1. Explicit ``tag`` argument
          2. Prefix match from ``_EVENT_TAG_MAP``
          3. Default ``"UI"``

        Delegates to ``TaggedLogger`` so the tag is stamped consistently
        with every other ``log.tag("X").info(...)`` call.
        """
        resolved_tag = tag or self._infer_tag(event)
        summary_parts = [event]
        if data:
            summary_parts.extend(f"{k}={v}" for k, v in data.items())
        ctx = dict(data) if data else {}
        ctx["tag"] = resolved_tag
        tagged = self.tag(resolved_tag)
        tagged.emit(tagged._make_record(
            level=LogLevel.INFO,
            message=" ".join(summary_parts),
            context=ctx,
        ))
