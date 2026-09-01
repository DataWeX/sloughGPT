"""health — show API server status."""

from __future__ import annotations

from ..console import Console
from ..commands import ShellCommands

help = "Show API server status"


def run(argv: list[str], out: Console, api: ShellCommands,
        env: dict[str, str]) -> int:
    try:
        with out.spinner("Checking health") as s:
            h = api.health()
    except Exception as e:
        out.status("error", f"Health check failed: {type(e).__name__}: {e}")
        out.note("Use 'api start' to launch the API server.")
        return 1
    if not isinstance(h, dict):
        out.status("error", "API server returned invalid response")
        return 1
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
