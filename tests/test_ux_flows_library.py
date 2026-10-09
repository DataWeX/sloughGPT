"""Registry integrity + spec-linking for domain.journeys.

Runs without a live stack — validates the declarative registry against
docs/UX_FLOWS.md so a renamed spec heading or dropped flow fails the suite.
The live sweep is `python -m domain.journeys` (marked slow, only when
SLO_UX_LIVE=1).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pytest

from domain.journeys import FLOWS, get_flow, list_flows

_REPO = Path(__file__).resolve().parent.parent
_UX_FLOWS_MD = _REPO / "docs" / "UX_FLOWS.md"


def test_thirteen_flows_home_plus_twelve_features():
    assert len(FLOWS) == 13
    assert len({f.id for f in FLOWS}) == 13, "flow ids must be unique"


def test_get_flow_lookup():
    assert get_flow("2-write").label == "Writing Assistant"
    with pytest.raises(KeyError):
        get_flow("does-not-exist")


def test_every_flow_links_to_an_existing_spec_heading():
    text = _UX_FLOWS_MD.read_text()
    for flow in FLOWS:
        assert flow.spec in text, f"{flow.id}: spec heading missing from UX_FLOWS.md: {flow.spec!r}"


def test_flows_build_without_errors():
    for flow in FLOWS:
        steps = flow.build()
        assert steps, f"{flow.id} built zero steps"


def test_list_flows_preserves_registry_order():
    assert [f.id for f in list_flows()] == [f.id for f in FLOWS]


def _api_ok() -> bool:
    try:
        with urllib.request.urlopen(
            os.environ.get("SLO_API_URL", "http://localhost:8000") + "/health", timeout=3
        ):
            return True
    except Exception:
        return False


@pytest.mark.slow
def test_live_home_flow_smoke():
    """Single-flow live smoke — the full sweep is `python -m domain.journeys`."""
    if os.environ.get("SLO_UX_LIVE") != "1":
        pytest.skip("set SLO_UX_LIVE=1 to run live journeys")
    if not _api_ok():
        pytest.skip("API not reachable")

    import asyncio

    from domain.journeys.runner import run

    rc = asyncio.run(run([get_flow("0-home")], headed=False, strict_errors=False))
    assert rc == 0, "home journey failed"
