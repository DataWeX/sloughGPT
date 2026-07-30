"""souls / switch / whoami — manage AI personalities (souls)."""

from __future__ import annotations

from ..console import Console
from ..commands import ShellCommands

help = "List, switch, or show current soul"
names = ["souls", "switch", "whoami"]


def run(argv: list[str], out: Console, api: ShellCommands,
        env: dict[str, str]) -> int:
    cmd = argv[0] if argv else "souls"

    if cmd == "souls":
        with out.spinner("Fetching souls") as s:
            souls = api.souls()
        s.ok("Souls loaded")
        if not souls:
            out.print("  No souls available")
            return 0
        rows = []
        for s in souls:
            name = s.get("name", s.get("id", "?"))
            desc = s.get("description", "")[:50]
            traits = s.get("traits", [])
            trait_str = ", ".join(str(t)[:15] for t in traits[:3])
            rows.append([name, desc, trait_str])
        out.table(rows, ["Soul", "Description", "Traits"])
        return 0

    if cmd == "switch":
        soul_name = " ".join(argv[1:]) if len(argv) > 1 else ""
        if not soul_name:
            out.print("  Usage: switch <soul_name>")
            return 1
        with out.spinner("Switching soul") as s:
            result = api.switch_soul(soul_name)
        s.ok(f"Switched to {soul_name}")
        out.json(result)
        return 0

    if cmd == "whoami":
        with out.spinner("Fetching current soul") as s:
            soul = api.current_soul()
        s.ok("Current soul loaded")
        out.print(f"  Current soul: {soul.get('name', 'unknown')}")
        if soul.get("description"):
            out.print(f"  Description: {soul['description']}")
        return 0

    return 0
