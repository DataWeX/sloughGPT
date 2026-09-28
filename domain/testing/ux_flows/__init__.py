"""UX_FLOWS journey library — one spec-linked registry for docs/UX_FLOWS.md.

Public API:
    Flow, FLOWS, get_flow, list_flows, main

Entry points:
    .venv/bin/python -m domain.testing.ux_flows        # run journeys (live stack)
    .venv/bin/python -m domain.testing.ux_flows --list # registry + spec links
"""

from __future__ import annotations

from domain.testing.ux_flows.flows import FLOWS, Flow, get_flow, list_flows
from domain.testing.ux_flows.runner import main

__all__ = ["Flow", "FLOWS", "get_flow", "list_flows", "main"]
