"""Command palette — quick screen switching (Ctrl+P)."""

from __future__ import annotations

from rich.text import Text

from apps.tui.components import CONSOLE, Color
from apps.tui.screen import Screen
from apps.tui.session import TuiSession


ITEMS = [
    ("home",     "Home",         "Home dashboard"),
    ("c",        "Chat",         "Chat with the model"),
    ("g",        "Generate",     "One-shot text generation"),
    ("m",        "Models",       "Browse local models"),
    ("e",        "Eval",         "Eval results dashboard"),
    ("d",        "Datasets",     "Browse datasets"),
    ("s",        "Status",       "System & API status"),
    ("t",        "Train",        "Start training"),
    ("h",        "Help",         "Keyboard shortcuts & about"),
    ("q",        "Quit",         "Exit TUI"),
]


class PaletteScreen(Screen):
    """Overlay palette — launched from Ctrl+P, returns a screen name to jump to."""

    name = "palette"
    bindings = []

    def render(self, session: TuiSession) -> str:
        CONSOLE.print()
        CONSOLE.print(Text("  Command Palette", style=f"bold {Color.PRIMARY}"))
        CONSOLE.print(Text("  Jump to screen:", style=Color.MUTED))
        CONSOLE.print()

        for key, label, desc in ITEMS:
            CONSOLE.print(f"  [{Color.HIGHLIGHT}]{key}[/]  {label:<12} {desc}")

        CONSOLE.print()
        CONSOLE.print(Text("  [Esc] cancel  [key] select", style=Color.MUTED))
        return self._handle_input()

    def _handle_input(self) -> str:
        import readchar
        while True:
            k = readchar.readkey()
            if k == readchar.key.ESC or k == "\x1b":
                return ""
            for pk, _, _ in ITEMS:
                if k.lower() == pk:
                    return pk
