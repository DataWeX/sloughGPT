"""Key binding system for TUI screens.

``Binding`` is a data class for individual key mappings.
``BindingManager`` holds global bindings and provides ``format_footer()`` and ``simple_input()``.

Screens create their own ``BindingManager`` (or use ``BindingManager.default()``)
and pass screen-specific bindings as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from apps.tui.keys import CTRL_P, CTRL_C, ESC


@dataclass
class Binding:
    """A key binding mapping one or more keys to an action."""

    keys: List[str]
    description: str
    action: str

    KEY_LABELS = {
        CTRL_P: "Ctrl+P",
        CTRL_C: "Ctrl+C",
        ESC: "Esc",
        "\r": "Enter",
        "\x7f": "BS",
        "/": "/",
        "r": "r",
        "t": "t",
        "m": "m",
    }

    @property
    def key_label(self) -> str:
        labels = []
        for k in self.keys:
            labels.append(self.KEY_LABELS.get(k, k))
        return "/".join(labels)


class BindingManager:
    """Manages global bindings and provides input/footer helpers.

    Usage::

        bm = BindingManager.default()
        footer = bm.format_footer(screen_bindings)
        action = bm.simple_input(screen_bindings)
    """

    def __init__(self):
        self.global_bindings: List[Binding] = [
            Binding([CTRL_P], "palette", "__palette__"),
            Binding([CTRL_C, "q", "Q"], "quit", "quit"),
            Binding([ESC], "home", "home"),
        ]
        self.footer_global: List[Binding] = [
            Binding([ESC], "home", "home"),
            Binding([CTRL_P], "palette", "__palette__"),
        ]

    @classmethod
    def default(cls) -> BindingManager:
        """Shortcut: returns a fresh BindingManager with standard global bindings."""
        return cls()

    def format_footer(self, screen_bindings: List[Binding], *, include_global: bool = True) -> str:
        """Format bindings as a footer hint string."""
        all_bindings = list(screen_bindings)
        if include_global:
            all_bindings.extend(self.footer_global)
        return "  ".join(f"[{b.key_label}] {b.description}" for b in all_bindings)

    def simple_input(self, screen_bindings: List[Binding]) -> str:
        """Handle input for simple navigation screens.

        Reads a key, checks global bindings first, then screen bindings.
        Returns the action string.
        """
        import readchar

        while True:
            key = readchar.readkey()

            for b in self.global_bindings:
                if key in b.keys:
                    return b.action

            for b in screen_bindings:
                if key in b.keys:
                    return b.action


# Pre-created default for convenience
_default_manager = BindingManager.default()


def format_footer(screen_bindings: List[Binding], *, include_global: bool = True) -> str:
    """Module-level shortcut for ``BindingManager.default().format_footer()``."""
    return _default_manager.format_footer(screen_bindings, include_global=include_global)


def simple_input(screen_bindings: List[Binding]) -> str:
    """Module-level shortcut for ``BindingManager.default().simple_input()``."""
    return _default_manager.simple_input(screen_bindings)
