"""Global key handlers for TUI — command palette, quit, navigation."""

from __future__ import annotations

import readchar


CTRL_P = "\x10"
CTRL_C = "\x03"
CTRL_D = "\x04"
ESC = "\x1b"


def is_global_key(key: str) -> bool:
    """Check if a key is a global action (command palette, quit)."""
    return key in (CTRL_P, CTRL_C, "q", "Q")


def handle_global_key(key: str) -> str:
    """Map a global key to an action string. Returns '' if not global."""
    if key == CTRL_P:
        return "__palette__"
    if key in (CTRL_C, "q", "Q"):
        return "quit"
    return ""
