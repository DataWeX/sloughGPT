"""UX_FLOWS journey harness — one spec-linked registry for docs/UX_FLOWS.md.

Public API:
    Flow, FLOWS, get_flow, list_flows, main

Entry points (from the repo root):
    python -m tests.ux_journeys        # run journeys (live stack)
    python -m tests.ux_journeys --list # registry + spec links

This is test machinery: it drives a live stack and writes a JSON report.
Product surfaces read the report file and spawn this across a process
boundary — they never import it.
"""

from __future__ import annotations

from .flows import FLOWS, Flow, get_flow, list_flows
from .runner import main

__all__ = ["Flow", "FLOWS", "get_flow", "list_flows", "main"]
