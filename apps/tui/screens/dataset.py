"""Dataset Browser — browse datasets and conversation files with pagination and sort."""

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


PAGE_SIZE = 12
SORT_OPTIONS = ["mtime", "name", "size"]


class DatasetScreen(Screen):
    name = "dataset"
    bindings = [
        Binding(["r"], "refresh", "dataset"),
        Binding(["n"], "next page", "__page_down"),
        Binding(["p"], "prev page", "__page_up"),
        Binding(["s"], "cycle sort", "__sort"),
    ]

    def __init__(self):
        super().__init__()
        self.page: int = 0
        self.sort_key: str = "mtime"
        self.sort_reverse: bool = True

    @staticmethod
    def _scan_datasets(repo_root: Path) -> List[dict]:
        entries = []

        ds_dir = repo_root / "datasets"
        if ds_dir.is_dir():
            for f in sorted(ds_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if f.name.startswith("."):
                    continue
                size = f.stat().st_size
                entries.append({
                    "name": f.name, "source": "datasets/",
                    "size": size,
                    "size_str": f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB",
                    "type": "dir" if f.is_dir() else "file", "mtime": f.stat().st_mtime,
                })

        conv_dir = repo_root / "data" / "conversations"
        if conv_dir.is_dir():
            for f in sorted(conv_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if f.name.startswith("."):
                    continue
                size = f.stat().st_size
                entries.append({
                    "name": f.name, "source": "conversations",
                    "size": size, "size_str": f"{size / 1024:.1f} KB",
                    "type": "json", "mtime": f.stat().st_mtime,
                })

        data_dir = repo_root / "data"
        if data_dir.is_dir():
            for f in sorted(data_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
                size = f.stat().st_size
                entries.append({
                    "name": f.name, "source": "data/",
                    "size": size, "size_str": f"{size / 1024:.1f} KB",
                    "type": "jsonl", "mtime": f.stat().st_mtime,
                })

        return entries

    @staticmethod
    def _sort_entries(entries: List[dict], key: str, reverse: bool) -> List[dict]:
        if key == "name":
            entries.sort(key=lambda e: e["name"].lower(), reverse=reverse)
        elif key == "size":
            entries.sort(key=lambda e: e["size"], reverse=reverse)
        else:
            entries.sort(key=lambda e: e["mtime"], reverse=reverse)
        return entries

    def render(self, session: TuiSession) -> str:
        self.render_header("Dataset Browser", "datasets/  ·  conversations  ·  data/")

        all_entries = self._scan_datasets(session.repo_root)
        all_entries = self._sort_entries(all_entries, self.sort_key, self.sort_reverse)

        total = len(all_entries)
        pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        if self.page >= pages:
            self.page = pages - 1

        start = self.page * PAGE_SIZE
        entries = all_entries[start:start + PAGE_SIZE]

        if not entries:
            CONSOLE.print(Text("  No datasets or conversation files found.", style=Color.MUTED))
        else:
            table = Table(show_header=True, box=SIMPLE, border_style=Color.BORDER, padding=(0, 1))
            table.add_column("Name", style=Color.WHITE)
            table.add_column("Location", style=Color.MUTED, width=14)
            table.add_column("Size", style=Color.MUTED, width=10)
            table.add_column("Type", style=Color.MUTED, width=6)
            for e in entries:
                loc_style = Color.PRIMARY if e["source"] == "datasets/" else Color.MUTED
                table.add_row(e["name"][:40], f"[{loc_style}]{e['source']}[/]", e["size_str"], e["type"])
            CONSOLE.print(table)

            sort_dir = "↓" if self.sort_reverse else "↑"
            parts = [f"  Page {self.page + 1}/{pages}  ({total} total)"]
            if total > PAGE_SIZE:
                parts.append("[n] next  [p] prev")
            parts.append(f"sort: {self.sort_key} {sort_dir}")
            CONSOLE.print(Text("  ".join(parts), style=Color.MUTED))

        self.render_footer()
        return self._handle_input(pages)

    def _handle_input(self, pages: int) -> str:
        import readchar

        while True:
            key = readchar.readkey()

            for b in self.binding_manager.global_bindings:
                if key in b.keys:
                    return b.action

            val = next((b.action for b in self.bindings if key in b.keys), None)

            if val == "__page_down" and self.page + 1 < pages:
                self.page += 1
            elif val == "__page_up" and self.page > 0:
                self.page -= 1
            elif val == "__sort":
                idx = SORT_OPTIONS.index(self.sort_key)
                idx = (idx + 1) % len(SORT_OPTIONS)
                self.sort_key = SORT_OPTIONS[idx]
                self.sort_reverse = True
            elif key == readchar.key.ESC:
                return "home"
            else:
                return "dataset"

            return "dataset"
