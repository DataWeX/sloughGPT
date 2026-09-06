"""status — show quick system status."""

from __future__ import annotations

from ..console import Console
from ..commands import ShellCommands

help = "Show quick system status"


def run(argv: list[str], out: Console, api: ShellCommands,
        env: dict[str, str]) -> int:
    # Parse flags
    json_output = "--json" in argv or "-j" in argv

    try:
        with out.spinner("Checking status"):
            h = api.health()
    except Exception as e:
        out.status("error", f"Status check failed: {type(e).__name__}: {e}")
        return 1

    if not isinstance(h, dict):
        out.status("error", "API server returned invalid response")
        return 1

    status = h.get("status", "unknown")
    if status == "unknown":
        out.status("error", "API server is not responding")
        return 1

    if json_output:
        import json
        print(json.dumps({
            "status": status,
            "model": h.get("model_type", ""),
            "soul": h.get("soul_name", ""),
            "uptime": h.get("uptime", 0),
            "model_loaded": h.get("model_loaded", False),
        }))
        return 0

    # One-liner: ● online | gpt2 | default | up 2h 30m
    icon = "●" if status == "healthy" else "○"
    model = h.get("model_type", "—")
    soul = h.get("soul_name", "—")
    uptime = _fmt_uptime(h.get("uptime", 0))
    loaded = h.get("model_loaded", False)

    status_str = f"{icon} {status}"
    model_str = f"model={model}" if loaded else f"model={model} (loading)"
    soul_str = f"soul={soul}"

    out.write(f"{status_str} | {model_str} | {soul_str} | up {uptime}")
    return 0


def _fmt_uptime(secs: int) -> str:
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        m, s = divmod(secs, 60)
        return f"{m}m {s}s"
    h, remainder = divmod(secs, 3600)
    m, _ = divmod(remainder, 60)
    return f"{h}h {m:02d}m"
