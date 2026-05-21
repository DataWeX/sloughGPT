"""Models screen — browse local models and souls with pagination and search."""

from __future__ import annotations

from pathlib import Path
from typing import List

from rich.text import Text
from rich.table import Table
from rich.box import SIMPLE

from apps.tui.components import CONSOLE, Color
from apps.tui.session import TuiSession
from apps.tui.screen import Screen
from apps.tui.bindings import Binding


PAGE_SIZE = 15


class ModelsScreen(Screen):
    name = "models"
    bindings = [
        Binding(["r"], "refresh", "models"),
        Binding(["n"], "next page", "__page_down"),
        Binding(["p"], "prev page", "__page_up"),
    ]

    def __init__(self):
        super().__init__()
        self.page: int = 0
        self.search: str = ""

    @staticmethod
    def _scan_models(repo_root: Path) -> List[dict]:
        models_dir = repo_root / "models"
        if not models_dir.is_dir():
            return []
        entries = []
        for f in sorted(models_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.suffix in (".slo", ".pt", ".pth", ".safetensors"):
                size = f.stat().st_size
                size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
                entries.append({"name": f.name, "size": size_str, "suffix": f.suffix})
        return entries

    def render(self, session: TuiSession) -> str:
        self.render_header("Models", str(session.repo_root / "models"))

        all_entries = self._scan_models(session.repo_root)

        if self.search:
            all_entries = [e for e in all_entries if self.search.lower() in e["name"].lower()]

        total = len(all_entries)
        pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        if self.page >= pages:
            self.page = pages - 1

        start = self.page * PAGE_SIZE
        entries = all_entries[start:start + PAGE_SIZE]

        if not entries:
            msg = f'  No models matching "{self.search}".' if self.search else "  No models found in models/ directory."
            CONSOLE.print(Text(msg, style=Color.MUTED))
        else:
            table = Table(show_header=True, box=SIMPLE, border_style=Color.BORDER, padding=(0, 1))
            table.add_column("Name", style=Color.WHITE)
            table.add_column("Size", style=Color.MUTED, width=10)
            table.add_column("Type", style=Color.MUTED, width=12)
            type_labels = {".slo": "Slo", ".pt": "PyTorch", ".pth": "PyTorch", ".safetensors": "SafeTensors"}
            for e in entries:
                table.add_row(e["name"], e["size"], type_labels.get(e["suffix"], e["suffix"]))
            CONSOLE.print(table)
            if total > PAGE_SIZE:
                CONSOLE.print(Text(f"  Page {self.page + 1}/{pages}  ({total} total)  [n] next  [p] prev", style=Color.MUTED))

        CONSOLE.print()
        if self.search:
            CONSOLE.print(Text(f"  Search: {self.search}  [/] clear  type to filter", style=Color.MUTED))
        else:
            CONSOLE.print(Text("  [/] search by name", style=Color.MUTED))

        self.render_footer()
        return self._handle_input(pages)

    def _handle_input(self, pages: int) -> str:
        import readchar

        while True:
            key = readchar.readkey()

            for b in self.binding_manager.global_bindings:
                if key in b.keys:
                    self.search = ""
                    if b.action == "home":
                        self.page = 0
                    return b.action

            for b in self.bindings:
                if key in b.keys:
                    if b.action == "__page_down" and self.page + 1 < pages:
                        self.page += 1
                    elif b.action == "__page_up" and self.page > 0:
                        self.page -= 1
                    return "models"

            if key == "/":
                return "models"

            if key == readchar.key.BACKSPACE:
                self.search = self.search[:-1]
                return "models"

            if len(key) == 1 and key.isprintable():
                self.search += key
                return "models"

            return "models"
