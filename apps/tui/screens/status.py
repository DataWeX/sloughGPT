"""Status screen — system information and API diagnostics."""

from __future__ import annotations

from typing import Optional

from rich.text import Text

from apps.tui.components import CONSOLE, Color, divider
from apps.tui.adapters.http_api import ApiJsonResult
from apps.tui.adapters.local_status import scan_local_repo
from apps.tui.session import TuiSession
from apps.tui.screen import Screen
from apps.tui.bindings import Binding


class StatusScreen(Screen):
    name = "status"
    bindings = [Binding(["r"], "refresh", "status")]

    def render(self, session: TuiSession) -> str:
        self.render_header("System Status", session.api_base_url)
        divider()
        CONSOLE.print()

        CONSOLE.print(Text("  API Health", style=f"bold {Color.PRIMARY}"))
        health: Optional[ApiJsonResult] = None
        try:
            health = session.api_client.fetch_health(timeout=3.0)
        except Exception:
            pass
        if health and health.status_code == 200:
            status = (health.payload or {}).get("status", "unknown")
            CONSOLE.print(f"    [{Color.SUCCESS}]●[/]  {status}")
        else:
            code = health.status_code if health else 0
            CONSOLE.print(f"    [{Color.ERROR}]○[/]  unreachable  (status={code})")
        CONSOLE.print()

        CONSOLE.print(Text("  Local Repository", style=f"bold {Color.PRIMARY}"))
        CONSOLE.print(f"    {session.repo_root}")
        snap = scan_local_repo(session.repo_root)
        CONSOLE.print(f"    [{Color.MUTED}]▪[/]  {snap.model_file_count} models  ·  {snap.dataset_entry_count} datasets")
        CONSOLE.print()

        CONSOLE.print(Text("  Session", style=f"bold {Color.PRIMARY}"))
        CONSOLE.print(f"    API:  {session.api_base_url}")
        if session.last_checkpoint:
            CONSOLE.print(f"    Last checkpoint:  {session.last_checkpoint.name}")
        if session.last_job_id:
            CONSOLE.print(f"    Last job:  {session.last_job_id}")

        self.render_footer()
        return self.handle_input()
