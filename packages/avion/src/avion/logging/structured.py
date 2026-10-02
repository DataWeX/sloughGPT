"""Structured logger — context-enriched log entries, plain-text + JSON."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from avion.events.models import Event, EventType

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import graph acyclic
    from avion.events.logger import EventLogger

# Entry keys that are metadata, not payload — excluded from the journal
# data dict (they live in the Event's own fields or the stdlib record).
_METADATA_KEYS = frozenset({"time", "logger", "kind", "message"})


class StructuredLogger:
    """Logger that keeps machine-readable entries alongside stdlib logging.

    Two views over the same call:

    * the **stdlib record** — the message *plus* ``**fields`` under one
      ``avion_fields`` key (stdibb raises ``KeyError`` when ``extra``
      collides with a LogRecord attribute, so fields are never splatted
      in bare, and never dropped either);
    * the **journal** — when attached to an ``EventLogger``, the entry is
      journaled as a ``CUSTOM`` event through the same journal-first path.

    Attach a ``StdlibSink`` only when routing *not* through this class —
    this class already emits the stdlib line, so wiring both would double
    every message.

    Usage::

        log = StructuredLogger("chat_check")
        log.set_context(run="1")
        log.navigation("/chat", True)
        log.element("clicked", "text='Start'")
        log.save("arken_output/logs.json")
    """

    def __init__(self, name: str = "arken", *, logger: EventLogger | None = None):
        self.name = name
        self._context: dict[str, str] = {}
        self._entries: list[dict[str, Any]] = []
        self._py_logger = logging.getLogger(f"arken.{name}")
        self._logger = logger

    @property
    def logger(self) -> EventLogger | None:
        return self._logger

    def set_context(self, **kwargs: str) -> None:
        """Attach key-values included in every later entry."""
        self._context.update(kwargs)

    def _log(self, kind: str, message: str, level: int = logging.INFO, **fields: Any) -> None:
        entry = {
            "time": time.time(),
            "logger": self.name,
            "kind": kind,
            "message": message,
            **self._context,
            **fields,
        }
        self._entries.append(entry)
        # Fields reach the LogRecord as one safe key: preserved, not dropped,
        # and never named anything stdlib reserves.
        self._py_logger.log(level, message, extra={"avion_fields": dict(entry)})
        if self._logger is not None:
            payload = {k: v for k, v in entry.items() if k not in _METADATA_KEYS}
            explicit = payload.pop("success", None)
            self._logger.log(
                Event(
                    type=EventType.CUSTOM,
                    name=kind,
                    data={"message": message, **payload},
                    success=bool(explicit) if explicit is not None else level < logging.WARNING,
                    error=message if level >= logging.ERROR else "",
                )
            )

    def info(self, message: str, **fields: Any) -> None:
        self._log("info", message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._log("warning", message, level=logging.WARNING, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._log("error", message, level=logging.ERROR, **fields)

    def navigation(self, path: str, success: bool) -> None:
        self._log(
            "navigation",
            f"{'OK' if success else 'FAIL'} goto {path}",
            level=logging.INFO if success else logging.WARNING,
            path=path,
            success=success,
        )

    def element(self, action: str, description: str) -> None:
        self._log("element", f"{action}: {description}", action=action, target=description)

    def task_start(self, name: str) -> None:
        self._log("task_start", f"starting task {name}", task=name)

    def task_end(self, name: str, success: bool) -> None:
        self._log(
            "task_end",
            f"task {name} {'passed' if success else 'failed'}",
            level=logging.INFO if success else logging.WARNING,
            task=name,
            success=success,
        )

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self._entries if e["kind"] == kind]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._entries, f, indent=2, default=str)

    def clear(self) -> None:
        self._entries.clear()
