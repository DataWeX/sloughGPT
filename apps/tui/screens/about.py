"""Help / About screen — keyboard shortcuts, project info, docs links."""

from __future__ import annotations

from rich.text import Text
from rich.table import Table
from rich.box import SIMPLE

from apps.tui.components import CONSOLE, Color
from apps.tui.session import TuiSession
from apps.tui.screen import Screen


class AboutScreen(Screen):
    name = "about"
    bindings = []

    def render(self, session: TuiSession) -> str:
        self.render_header("Help & About", "Keyboard shortcuts  ·  project info")
        CONSOLE.print()

        CONSOLE.print(Text("  SloughGPT TUI", style=f"bold {Color.PRIMARY}"))
        CONSOLE.print(Text("  Interactive terminal UI for local LLM training and inference", style=Color.MUTED))
        CONSOLE.print()
        CONSOLE.print(f"    [{Color.MUTED}]▪[/]  API:  {session.api_base_url}")
        CONSOLE.print(f"    [{Color.MUTED}]▪[/]  Repo: {session.repo_root}")
        CONSOLE.print()

        CONSOLE.print(Text("  Screens", style=f"bold {Color.PRIMARY}"))
        t = Table(show_header=False, box=SIMPLE, border_style=Color.BORDER, padding=(0, 2), pad_edge=False)
        t.add_column("Key", style=Color.MUTED, width=18)
        t.add_column("Screen", style=Color.WHITE, width=14)
        t.add_column("Description", style=Color.MUTED)
        for key, label, desc in [
            ("home", "Home", "Dashboard with quick actions"),
            ("c", "Chat", "Streaming conversation"),
            ("g", "Generate", "One-shot text generation"),
            ("m", "Models", "Browse local models"),
            ("e", "Eval", "Eval results dashboard"),
            ("d", "Datasets", "Browse datasets"),
            ("s", "Status", "System & API diagnostics"),
            ("t", "Train", "Training quick-start guide"),
            ("h", "Help", "This screen"),
        ]:
            t.add_row(f"  [{Color.HIGHLIGHT}]{key}[/]", label, desc)
        CONSOLE.print(t)
        CONSOLE.print()

        CONSOLE.print(Text("  Global Keys (from any screen)", style=f"bold {Color.PRIMARY}"))
        t2 = Table(show_header=False, box=SIMPLE, border_style=Color.BORDER, padding=(0, 2), pad_edge=False)
        t2.add_column("Key", style=Color.MUTED, width=18)
        t2.add_column("Action", style=Color.WHITE, width=20)
        t2.add_column("Description", style=Color.MUTED)
        for key, action, desc in [
            ("Ctrl+P", "Command palette", "Jump to any screen by key"),
            ("Ctrl+C / q", "Quit", "Exit the TUI"),
            ("Esc", "Home", "Return to home screen"),
        ]:
            t2.add_row(f"  [{Color.HIGHLIGHT}]{key}[/]", action, desc)
        CONSOLE.print(t2)
        CONSOLE.print()

        CONSOLE.print(Text("  Per-Screen Keys", style=f"bold {Color.PRIMARY}"))
        t3 = Table(show_header=False, box=SIMPLE, border_style=Color.BORDER, padding=(0, 2), pad_edge=False)
        t3.add_column("Screen", style=Color.MUTED, width=18)
        t3.add_column("Key", style=Color.WHITE, width=16)
        t3.add_column("Action", style=Color.MUTED)
        for scr, key, action in [
            ("Chat", "Enter", "Send message"),
            ("Chat", "/", "Clear conversation"),
            ("Generate", "t", "Cycle temperature"),
            ("Generate", "m", "Cycle max tokens"),
            ("Generate", "Enter", "Generate text"),
            ("Models", "r", "Refresh"),
            ("Models", "/", "Search by name"),
            ("Models", "n/p", "Page navigation"),
            ("Eval/Dataset/Status", "r", "Refresh"),
            ("Dataset", "s", "Cycle sort order"),
        ]:
            t3.add_row(f"  [{Color.MUTED}]{scr}[/]", f"  [{Color.HIGHLIGHT}]{key}[/]", action)
        CONSOLE.print(t3)

        self.render_footer()
        return self.handle_input()
