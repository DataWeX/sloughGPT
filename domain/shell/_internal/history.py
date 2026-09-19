"""HistoryStore — single owner for shell command history.

Replaces the previous three-way split:

- ``ShellREPL._history`` (plain list copy of state)
- ``ShellState.history`` (MogDB persistence, 500 cap)
- ``TuiRepl._cmd_history`` (snapshot taken at TUI startup)

The store owns one ``entries`` list.  An optional ``ShellState``
reference is kept in sync on :meth:`append` (dedup of consecutive
duplicates happens in ``ShellState.add_history``, the in-memory list
keeps every entry so ``history`` numbering is stable).

Readline's ``~/.config/sloughgpt/.shell_history`` file stays separate
(owned by ``ShellREPL._setup_readline``) — it is a UI affordance, not
the source of truth.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import ShellState

logger = logging.getLogger("slo.shell.history")


class HistoryStore:
    """List-backed command history with optional state sync."""

    def __init__(
        self,
        initial: list[str] | None = None,
        state: ShellState | None = None,
        max_entries: int = 500,
    ) -> None:
        self.entries: list[str] = list(initial) if initial else []
        self._state = state
        self._max_entries = max_entries

    def append(self, line: str) -> bool:
        """Append *line*; sync to state when attached.

        Returns True when the entry was recorded, False for blank lines.
        """
        if not line:
            return False
        self.entries.append(line)
        if self._state is not None:
            try:
                self._state.add_history(line)
            except Exception as e:
                logger.debug("history state sync failed: %s", e)
        return True

    def list(self, n: int | None = None) -> list[str]:
        """Return all entries, or the last *n*."""
        if n is None or n >= len(self.entries):
            return list(self.entries)
        return self.entries[-n:]

    def get(self, num: int) -> str | None:
        """1-based lookup (``history`` numbering); None when out of range."""
        if num < 1 or num > len(self.entries):
            return None
        return self.entries[num - 1]

    def search(self, query: str, limit: int = 50) -> list[str]:
        """Most-recent-first substring search."""
        if not query:
            return list(reversed(self.entries[-limit:]))
        q = query.lower()
        return [h for h in reversed(self.entries) if q in h.lower()][:limit]

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def __iter__(self):
        return iter(self.entries)
