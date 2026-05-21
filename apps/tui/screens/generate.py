"""Generate screen — one-shot text generation."""

from __future__ import annotations

from rich.text import Text

from apps.tui.components import CONSOLE, Color
from apps.tui.session import TuiSession
from apps.tui.screen import Screen
from apps.tui.bindings import Binding


class GenerateScreen(Screen):
    name = "generate"
    bindings = [
        Binding(["t"], "cycle temp", "temp"),
        Binding(["m"], "cycle max", "max"),
    ]

    def __init__(self):
        super().__init__()
        self.prompt: str = ""
        self.result: str = ""
        self.temperature: float = 0.8
        self.max_tokens: int = 128
        self.generating: bool = False

    def render(self, session: TuiSession) -> str:
        self.render_header("Generate", "One-shot text generation")
        CONSOLE.print()
        CONSOLE.print(f"  [{Color.MUTED}]t[/] temp: {self.temperature:.1f}      [{Color.MUTED}]m[/] max: {self.max_tokens}")
        CONSOLE.print()

        CONSOLE.print(Text("  Prompt:", style=f"bold {Color.PRIMARY}"))
        terminal_w = CONSOLE.width - 4
        text = self.prompt or "(type your prompt)"
        if not self.prompt:
            CONSOLE.print(Text(f"  {text}", style=Color.MUTED))
        else:
            remaining = text
            while remaining:
                CONSOLE.print(Text(f"  {remaining[:terminal_w]}", style=Color.WHITE))
                remaining = remaining[terminal_w:]

        if self.result:
            CONSOLE.print()
            CONSOLE.print(Text("  Generated:", style=f"bold {Color.SUCCESS}"))
            word_count = len(self.result.split())
            CONSOLE.print(Text(f"  ({word_count} words)", style=Color.MUTED))
            remaining = self.result
            while remaining:
                CONSOLE.print(Text(f"  {remaining[:terminal_w]}", style=Color.WHITE))
                remaining = remaining[terminal_w:]

        if self.generating:
            CONSOLE.print()
            CONSOLE.print(Text("  [generating...]", style=Color.MUTED))
            return ""

        CONSOLE.print()
        CONSOLE.print(Text(self.binding_manager.format_footer(self.bindings), style=Color.MUTED))
        return self._handle_input(session)

    def _handle_input(self, session: TuiSession) -> str:
        import readchar

        while True:
            key = readchar.readkey()

            for b in self.binding_manager.global_bindings:
                if key in b.keys:
                    return b.action

            if key == readchar.key.ENTER:
                if not self.prompt.strip() or self.generating:
                    return "generate"

                self.result = ""
                self.generating = True

                collected = ""
                for token in session.api_client.generate_text(
                    session.api_base_url,
                    self.prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ):
                    collected += token

                self.result = collected
                self.generating = False
                return "generate"

            elif key == "t":
                temps = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
                idx = temps.index(self.temperature) if self.temperature in temps else 4
                self.temperature = temps[(idx + 1) % len(temps)]
                return "generate"

            elif key == "m":
                sizes = [16, 32, 64, 128, 256, 512, 1024]
                idx = sizes.index(self.max_tokens) if self.max_tokens in sizes else 3
                self.max_tokens = sizes[(idx + 1) % len(sizes)]
                return "generate"

            elif key == readchar.key.BACKSPACE:
                self.prompt = self.prompt[:-1]
                return "generate"

            elif len(key) == 1 and key.isprintable():
                self.prompt += key
                return "generate"

            return "generate"
