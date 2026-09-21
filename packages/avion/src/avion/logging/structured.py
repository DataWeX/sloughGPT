"""Structured logger — context-enriched log entries, plain-text + JSON."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any


class StructuredLogger:
    """Logger that keeps machine-readable entries alongside stdlib logging.

    Usage::

        log = StructuredLogger("chat_check")
        log.set_context(run="1")
        log.navigation("/chat", True)
        log.element("clicked", "text='Start'")
        log.save("arken_output/logs.json")
    """

    def __init__(self, name: str = "arken"):
        self.name = name
        self._context: dict[str, str] = {}
        self._entries: list[dict[str, Any]] = []
        self._py_logger = logging.getLogger(f"arken.{name}")

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
        self._py_logger.log(level, message)

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
