#!/usr/bin/env python3
"""
terminal-tui.py — Standalone TUI demo.

Run in a real terminal:
    python3 scripts/terminal-tui.py

Shows:
  - Single-line spinner (fixed height, no scroll)
  - Multi-line fixed-height panel
  - Full dev dashboard with tabs and logs
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "cli", "src"))

from core.tui import (
    LiveDisplay, DevDashboard, TabConfig,
    FG, _fg, _RESET, _BOLD, _DIM,
    render_panel, render_gradient_header, render_metrics_row,
    render_tab_bar, render_footer,
)

SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def demo_single_line():
    """One line. Spinner rotates in place. No vertical movement."""
    with LiveDisplay() as d:
        for i in range(60):
            ch = SPINNERS[i % len(SPINNERS)]
            d.update(f"  {ch}  Loading 10 tasks...")
            time.sleep(0.08)


def demo_multi_line_fixed():
    """Fixed-height block. Lines update in place."""
    with LiveDisplay() as d:
        for frame in range(60):
            lines = []
            lines.append(f"  {_BOLD}Loading 10 tasks{_RESET}")
            for i in range(10):
                ch = SPINNERS[(frame + i) % len(SPINNERS)]
                lines.append(f"    {ch}  Task {i+1}/10")
            lines.append("")
            lines.append(f"  {_DIM}frame {frame+1}{_RESET}")
            d.update("\n".join(lines))
            time.sleep(0.08)


def demo_dashboard():
    """Full dev-server-style dashboard."""
    tabs = [
        TabConfig("api", "API", port=8000),
        TabConfig("web", "Web", port=3000),
    ]

    api_lines = [
        "08:00:01 INF [START] slo Logging: level=INFO",
        "08:00:01 INF [START] slo Config: Qwen2.5-0.5B @ 0.0.0.0:8000",
        "08:00:01 INF [INFRA] middleware RateLimitMiddleware registered",
        "08:00:02 INF [INFRA] middleware AuthMiddleware registered",
        "08:00:02 INF [START] slo Server ready on :8000",
        "08:00:05 WRN [INFRA] config Unknown config key: api_port",
        "08:00:10 INF [HEALTH] Server healthy (uptime 9s)",
        "08:00:15 INF [INFRA] watchdog Auto-save triggered",
    ]

    db = DevDashboard("SloughGPT Dev Server", tabs=tabs)
    db._api_lines = api_lines
    db._web_lines = [
        "> next dev",
        "  ▲ Next.js 15.5.24",
        "  - Local: http://localhost:3000",
        "  ✓ Ready in 2.9s",
    ]
    db._states = {"api": "ready", "web": "ready"}

    with LiveDisplay() as d:
        for frame in range(180):
            if frame % 20 == 0 and len(api_lines) < 200:
                api_lines.append(f"08:{len(api_lines):02d}:00 INF [HEALTH] heartbeat")
            db._frame = frame
            d.update(db._render())
            time.sleep(0.05)


def main():
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("TUI Terminal Demo")
        print("=" * 40)
        print("  1. Single-line spinner (one line, rotates in place)")
        print("  2. Multi-line fixed block (10 spinners, fixed height)")
        print("  3. Full dashboard (tabs, logs, metrics)")
        print()
        choice = input("Choose [1/2/3]: ").strip()

    if choice == "1":
        demo_single_line()
    elif choice == "2":
        demo_multi_line_fixed()
    elif choice == "3":
        demo_dashboard()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
