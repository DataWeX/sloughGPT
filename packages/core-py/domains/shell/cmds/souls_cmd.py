"""souls / switch / whoami — manage AI personalities (souls)."""

from __future__ import annotations

from ..console import Console
from ..commands import ShellCommands

help = "List, switch, or show current soul"
names = ["souls", "switch", "whoami"]


def run(argv: list[str], out: Console, api: ShellCommands,
        env: dict[str, str]) -> int:
    cmd = argv[0] if argv else "souls"

    try:
        if cmd == "souls":
            with out.spinner("Fetching souls") as s:
                souls = api.souls()
            if isinstance(souls, dict) and "error" in souls:
                s.fail("Failed to fetch souls")
                out.print(f"  Error: {souls['error']}")
                return 1
            s.ok("Souls loaded")
            if not souls:
                out.print("  No souls available")
                return 0
            rows = []
            for soul in souls:
                name = soul.get("name", soul.get("id", "?"))
                desc = soul.get("description", "")[:50]
                traits = soul.get("traits", [])
                trait_str = ", ".join(str(t)[:15] for t in traits[:3])
                rows.append([name, desc, trait_str])
            out.table(rows, ["Soul", "Description", "Traits"])
            return 0

        if cmd == "switch":
            soul_name = " ".join(argv[1:]) if len(argv) > 1 else ""
            if not soul_name:
                souls = []
                with out.spinner("Fetching souls") as s:
                    souls = api.souls()
                if isinstance(souls, dict) and "error" in souls:
                    s.fail("Failed to fetch souls")
                    out.print(f"  Error: {souls['error']}")
                    return 1
                s.ok("Souls loaded")
                if not souls:
                    out.print("  No souls available")
                    return 1
                names = [s.get("name", s.get("id", "?")) for s in souls]
                soul_name = out.select("Switch to soul:", names)
            if not soul_name:
                out.print("  Cancelled")
                return 0
            with out.spinner("Switching soul") as s:
                result = api.switch_soul(soul_name)
            if isinstance(result, dict) and "error" in result:
                s.fail("Switch failed")
                out.print(f"  Error: {result['error']}")
                return 1
            s.ok(f"Switched to {soul_name}")
            return 0

        if cmd == "whoami":
            with out.spinner("Fetching current soul") as s:
                soul = api.current_soul()
            if isinstance(soul, dict) and "error" in soul:
                s.fail("Failed to fetch current soul")
                out.print(f"  Error: {soul['error']}")
                return 1
            s.ok("Current soul loaded")
            out.print(f"  Current soul: {soul.get('name', 'unknown')}")
            if soul.get("description"):
                out.print(f"  Description: {soul['description']}")
            return 0

    except Exception as e:
        from domains.shell.error import format_error
        out.print(format_error(e, "souls", color=False))
        return 1

    return 0
