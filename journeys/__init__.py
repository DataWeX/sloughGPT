"""UX_FLOWS journey library — one spec-linked registry for docs/UX_FLOWS.md.

Public API:
    Flow, FLOWS, get_flow, list_flows, main

Entry points:
    .venv/bin/python -m journeys        # run journeys (live stack)
    .venv/bin/python -m journeys --list # registry + spec links
"""

from __future__ import annotations

from journeys.flows import (
    ALL_FLOWS,
    FLOWS,
    MOBILE_FLOWS,
    Flow,
    get_any_flow,
    get_flow,
    list_all_flows,
    list_flows,
)
from journeys.runner import main

__all__ = [
    "Flow",
    "FLOWS",
    "MOBILE_FLOWS",
    "ALL_FLOWS",
    "get_flow",
    "get_any_flow",
    "list_flows",
    "list_all_flows",
    "main",
]
