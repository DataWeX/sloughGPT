#!/usr/bin/env python3
"""Pre-commit hook: stamp latest commit hash into anchored_summary.md."""
import subprocess
import pathlib

summary = pathlib.Path("anchored_summary.md")
if not summary.exists():
    raise SystemExit(0)

result = subprocess.run(
    ["git", "log", "--oneline", "-1"],
    capture_output=True, text=True,
)
commit_line = result.stdout.strip()

d = summary.read_text()
parts = d.split("## Current Task")
if len(parts) < 2:
    raise SystemExit(0)

# Get everything after the first line of Current Task section
rest_lines = parts[1].split("\n", 2)
after = rest_lines[2] if len(rest_lines) > 2 else ""

summary.write_text(
    parts[0]
    + "## Current Task\n"
    + f"No active task. Last commit: {commit_line}\n\n"
    + after.lstrip("\n")
)

# Re-stage so the commit doesn't fail with "file modified"
subprocess.run(["git", "add", str(summary)], capture_output=True)
