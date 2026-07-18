"""
ShellState — file-backed persistence for shell history and aliases.

Stores to ~/.config/sloughgpt/shell_state.json with automatic
load/save cycle. Maximum 500 history entries.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("slo.shell.state")

_CONFIG_DIR = Path.home() / ".config" / "sloughgpt"
_STATE_FILE = _CONFIG_DIR / "shell_state.json"
_MAX_HISTORY = 500


class ShellState:
    """Persistent shell state backed by a JSON file."""

    def __init__(self):
        self.history: list[str] = []
        self.aliases: dict[str, str] = {}
        self.env: dict[str, str] = {}
        self.last_session: str = ""
        self.first_run: bool = True
        self._load()

    def _load(self) -> None:
        if _STATE_FILE.is_file():
            try:
                data = json.loads(_STATE_FILE.read_text())
                self.history = data.get("history", [])
                self.aliases = data.get("aliases", {})
                self.env = data.get("env", {})
                self.last_session = data.get("last_session", "")
                self.first_run = data.get("first_run", True)
                logger.debug("Loaded shell state (%d entries, %d aliases, %d env vars)",
                             len(self.history), len(self.aliases), len(self.env))
            except Exception as e:
                logger.warning("Failed to load shell state: %s", e, extra={"tag": "INFRA"})

    def save(self) -> None:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            _STATE_FILE.write_text(json.dumps({
                "history": self.history[-_MAX_HISTORY:],
                "aliases": self.aliases,
                "env": self.env,
                "last_session": datetime.now(timezone.utc).isoformat(),
                "first_run": self.first_run,
            }, indent=2))
        except Exception as e:
            logger.warning("Failed to save shell state: %s", e, extra={"tag": "INFRA"})

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
