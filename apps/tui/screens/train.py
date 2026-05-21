"""Training screen — start and monitor training jobs."""

from __future__ import annotations

from rich.text import Text

from apps.tui.components import CONSOLE, Color
from apps.tui.session import TuiSession
from apps.tui.screen import Screen


class TrainScreen(Screen):
    name = "train"
    bindings = []

    def render(self, session: TuiSession) -> str:
        self.render_header("Training", "Train a model")
        CONSOLE.print()
        CONSOLE.print(Text("  Training Pipeline", style=f"bold {Color.PRIMARY}"))
        CONSOLE.print()

        steps = [
            ("1.", "Select dataset", "shakespeare, wikitext, or custom"),
            ("2.", "Configure hyperparameters", "layers, heads, embedding size, LR"),
            ("3.", "Start training", "python3 cli.py train --dataset shakespeare"),
            ("4.", "Monitor progress", "loss curves, step counters"),
        ]
        for num, title, desc in steps:
            CONSOLE.print(f"    [{Color.MUTED}]{num}[/]  [{Color.WHITE}]{title}[/]")
            CONSOLE.print(f"        {desc}")
            CONSOLE.print()

        CONSOLE.print(Text("  Quick start:", style=f"bold {Color.SUCCESS}"))
        CONSOLE.print("    python3 cli.py train --dataset shakespeare --max-steps 80")

        self.render_footer()
        return self.handle_input()
