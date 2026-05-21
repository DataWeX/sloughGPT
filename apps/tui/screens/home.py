"""Home dashboard screen for SloughGPT TUI."""

from __future__ import annotations

from typing import Optional

from rich.text import Text

from apps.tui.components import CONSOLE, Color, header, divider
from apps.tui.adapters.http_api import ApiJsonResult
from apps.tui.adapters.local_status import scan_local_repo, LocalStatusSnapshot
from apps.tui.session import TuiSession
from apps.tui.screen import Screen


class HomeScreen(Screen):
    name = "home"
    bindings = []

    def render(self, session: TuiSession) -> str:
        CONSOLE.clear()
        header("SloughGPT", "Terminal UI")

        CONSOLE.print(Text("  New here?", style=f"bold {Color.SECONDARY}"))
        CONSOLE.print(f"    [{Color.MUTED}]c[/]  Start a conversation  ·  [{Color.MUTED}]g[/]  Generate text  ·  [{Color.MUTED}]Ctrl+P[/]  Command palette")
        CONSOLE.print(f"    [{Color.MUTED}]python3 cli.py start[/]  Getting started guide")
        divider()

        health: Optional[ApiJsonResult] = None
        try:
            health = session.api_client.fetch_health(timeout=3.0)
        except Exception:
            pass
        api_icon = f"[{Color.SUCCESS}]●[/]" if (health and health.status_code == 200) else f"[{Color.ERROR}]○[/]"
        api_label = "healthy" if (health and health.status_code == 200) else f"status={health.status_code if health else 0}"
        CONSOLE.print(f"  {api_icon}  API  {session.api_base_url}  {api_label}")

        snap: Optional[LocalStatusSnapshot] = None
        try:
            snap = scan_local_repo(session.repo_root)
        except Exception:
            pass
        if snap:
            CONSOLE.print(f"  [{Color.MUTED}]▪[/]  {snap.repo_root}")
            CONSOLE.print(f"     {snap.model_file_count} models  ·  {snap.dataset_entry_count} datasets")
        else:
            CONSOLE.print(f"  [{Color.MUTED}]▪[/]  {session.repo_root}")

        if session.last_checkpoint:
            CONSOLE.print(f"  [{Color.MUTED}]▪[/]  checkpoint: {session.last_checkpoint.name}")
        if session.last_job_id:
            CONSOLE.print(f"  [{Color.MUTED}]▪[/]  job: {session.last_job_id}")

        CONSOLE.print()
        divider()
        CONSOLE.print(Text("  Quick Actions", style=f"bold {Color.PRIMARY}"))
        CONSOLE.print()

        actions = [
            ("c", "Chat"), ("g", "Generate"), ("m", "Models"),
            ("e", "Eval"), ("d", "Datasets"), ("s", "Status"),
            ("t", "Train"), ("h", "Help"), ("q", "Quit"),
        ]
        for i in range(0, len(actions), 3):
            row = actions[i:i + 3]
            CONSOLE.print("   " + "  ".join(f"[{Color.HIGHLIGHT}]{k}[/]  {l:<10}" for k, l in row))

        CONSOLE.print()
        return self._handle_input()

    def _handle_input(self) -> str:
        import readchar
        while True:
            key = readchar.readkey()
            for b in self.binding_manager.global_bindings:
                if key in b.keys:
                    return b.action
            k = key.lower()
            if k in ("c", "chat"):
                return "chat"
            elif k in ("g", "generate"):
                return "generate"
            elif k in ("m", "models"):
                return "models"
            elif k in ("e", "eval"):
                return "eval"
            elif k in ("d", "dataset"):
                return "dataset"
            elif k in ("s", "status"):
                return "status"
            elif k in ("t", "train"):
                return "train"
            elif k in ("h", "help", "about"):
                return "about"
            elif k in ("q", readchar.key.CTRL_C):
                return "quit"
