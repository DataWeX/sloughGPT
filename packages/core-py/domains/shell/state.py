"""
ShellState — MogDB-backed persistence for shell history and aliases.

Stores to a MogDB ``shell_state`` collection (single document keyed by
``_id: "state"``) instead of a raw JSON file.  Maximum 500 history
entries.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("slo.shell.state")

_MAX_HISTORY = 500

# Module-level MogDB handle (lazily initialised).
_db = None
_collection = None


def _get_collection(db_path: Optional[str] = None):
    """Return the ``shell_state`` collection, creating it on first call."""
    global _db, _collection
    if _collection is not None:
        return _collection
    from mogdb import MogDB
    if db_path is None:
        from domains.shared import find_repo_root
        repo = find_repo_root(Path(__file__).resolve())
        db_path = str(repo / "data" / "shell_state_mogdb")
    _db = MogDB(db_path)
    _collection = _db.collection("shell_state")
    return _collection


def set_shell_state_db(db_path: str) -> None:
    """Replace the module-level collection with one at *db_path* (for tests)."""
    global _db, _collection
    from mogdb import MogDB
    _db = MogDB(db_path)
    _collection = _db.collection("shell_state")


def reset_shell_state_db() -> None:
    """Clear the module-level collection reference."""
    global _db, _collection
    _db = None
    _collection = None


class ShellState:
    """Persistent shell state backed by MogDB."""

    def __init__(self, db_path: Optional[str] = None):
        self.history: list[str] = []
        self.aliases: dict[str, str] = {}
        self.env: dict[str, str] = {}
        self.last_session: str = ""
        self.first_run: bool = True
        self._col = _get_collection(db_path)
        self._load()

    # ── persistence ────────────────────────────────────────────────

    def _load(self) -> None:
        doc = self._col.find_one({"_id": "state"})
        if doc is None:
            return
        try:
            self.history = doc.get("history", [])
            self.aliases = doc.get("aliases", {})
            self.env = doc.get("env", {})
            self.last_session = doc.get("last_session", "")
            self.first_run = doc.get("first_run", True)
            logger.debug(
                "Loaded shell state (%d entries, %d aliases, %d env vars)",
                len(self.history), len(self.aliases), len(self.env),
            )
        except Exception as e:
            logger.warning("Failed to load shell state: %s", e, extra={"tag": "INFRA"})

    def save(self) -> None:
        data = {
            "history": self.history[-_MAX_HISTORY:],
            "aliases": self.aliases,
            "env": self.env,
            "last_session": datetime.now(timezone.utc).isoformat(),
            "first_run": self.first_run,
        }
        try:
            existing = self._col.find_one({"_id": "state"})
            if existing is not None:
                self._col.update_one({"_id": "state"}, {"$set": data})
            else:
                self._col.insert_one({"_id": "state", **data})
        except Exception as e:
            logger.warning("Failed to save shell state: %s", e, extra={"tag": "INFRA"})

    # ── mutators ───────────────────────────────────────────────────

    def add_history(self, line: str) -> None:
        if line and (not self.history or self.history[-1] != line):
            self.history.append(line)

    def set_alias(self, name: str, command: str) -> None:
        self.aliases[name] = command

    def unset_alias(self, name: str) -> bool:
        return self.aliases.pop(name, None) is not None

    def set_env(self, name: str, value: str) -> None:
        self.env[name] = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_count": len(self.history),
            "aliases": dict(self.aliases),
            "env_vars": len(self.env),
            "last_session": self.last_session,
            "first_run": self.first_run,
        }
