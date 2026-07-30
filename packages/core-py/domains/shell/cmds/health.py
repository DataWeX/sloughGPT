"""health — show API server status."""

from __future__ import annotations

from typing import Any as _Any

from ..console import Console
from ..commands import ShellCommands

help = "Show API server status"


def run(argv: list[str], out: Console, api: ShellCommands,
        env: dict[str, str]) -> int:
    with out.spinner("Checking health") as s:
        h = api.health()
    s.ok("Health check complete")
    status = h.get("status", "unknown")
    if status == "unknown":
        out.status("error", "API server is not responding")
        out.note("Use 'api start' to launch it.")
        return 1
    out.status("ok" if status == "healthy" else "warn",
               f"Status: {status}")
    out.kvlist([
        ("Model", h.get("model_type", "\u2014")),
        ("Soul", h.get("soul_name", "\u2014")),
    ])
    return 0
