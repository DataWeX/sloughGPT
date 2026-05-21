"""Screen base class for all TUI screens.

All screens inherit from ``Screen`` and implement ``render()``.
Simple navigation screens use the default ``handle_input()``.
Complex screens (chat, generate) override ``handle_input()`` for text entry.

Each screen gets its own ``BindingManager`` for global key dispatch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from rich.text import Text

from apps.tui.bindings import Binding, BindingManager
from apps.tui.session import TuiSession
from apps.tui.components import CONSOLE, Color, header, divider


class Screen(ABC):
    """Base class for all TUI screens.

    Subclasses set ``name`` and ``bindings``, then implement ``render()``.
    State lives in instance attributes (e.g. ``self.page``, ``self.search``).
    """

    name: str = ""
    bindings: List[Binding] = []

    def __init__(self) -> None:
        self.binding_manager = BindingManager.default()

    @abstractmethod
    def render(self, session: TuiSession) -> str:
        """Render screen, return next screen name or empty to re-render."""
        ...

    def handle_input(self) -> str:
        """Default input loop for navigation-only screens."""
        return self.binding_manager.simple_input(self.bindings)

    def format_footer(self) -> str:
        """Format footer from bindings via BindingManager."""
        return self.binding_manager.format_footer(self.bindings)

    def render_header(self, title: str, subtitle: str = "") -> None:
        """Render standard header with clear."""
        CONSOLE.clear()
        header(title, subtitle)

    def render_footer(self) -> None:
        """Render standard footer with keybindings hint."""
        CONSOLE.print()
        divider()
        CONSOLE.print(Text(self.format_footer(), style=Color.MUTED))
