#!/usr/bin/env python3
"""
Run the UX_FLOWS user journeys (docs/UX_FLOWS.md) against a live stack.

Legacy entrypoint: this now delegates to ``tests/ux_journeys``. It used to
carry a second implementation — its own ``Flow`` dataclass (no ``spec`` field),
its own 13 flow definitions and its own runner — so it drifted from the
spec-linked registry and escaped the copy gate that keeps declared flows
matching the shipped UI. There is one registry now; this script only forwards
argv.

Engine: avion (packages/avion, formerly voyager/arken) on Firefox — chromium
never commits navigations on this hardware.

Usage (from the repo root):
    scripts/python scripts/run_ux_flows.py                # all journeys
    scripts/python scripts/run_ux_flows.py --flow 2-write
    scripts/python scripts/run_ux_flows.py --list
    scripts/python scripts/run_ux_flows.py --strict-errors

Env:
    SLO_WEB_URL         default http://localhost:5173
    SLO_API_URL         default http://localhost:8000
    SLO_JOURNEY_BROWSER firefox (default) | chromium
"""

from __future__ import annotations

import os
import sys

# `tests` is a package under the repo root; resolve it there rather than
# letting a stray user-site `tests` shadow the repo's (AGENTS.md).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.ux_journeys.runner import main  # noqa: E402  (path bootstrap above)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
