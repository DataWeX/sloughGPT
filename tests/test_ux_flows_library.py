"""Registry integrity + spec-linking for journeys.

Runs without a live stack — validates the declarative registry against
docs/UX_FLOWS.md so a renamed spec heading or dropped flow fails the suite.
The live sweep is `python -m journeys` (marked slow, only when
SLO_UX_LIVE=1).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pytest

from journeys import ALL_FLOWS, FLOWS, MOBILE_FLOWS, get_flow, list_all_flows, list_flows

_REPO = Path(__file__).resolve().parent.parent
_UX_FLOWS_MD = _REPO / "docs" / "UX_FLOWS.md"

# Ratchet: web flows grandfathered without mapped tests. May only shrink.
_UNMAPPED_WEB_BASELINE = 12


def test_thirteen_flows_home_plus_twelve_features():
    assert len(FLOWS) == 13
    assert len({f.id for f in FLOWS}) == 13, "flow ids must be unique"


def test_get_flow_lookup():
    assert get_flow("2-write").label == "Writing Assistant"
    with pytest.raises(KeyError):
        get_flow("does-not-exist")


def test_every_flow_links_to_an_existing_spec_heading():
    text = _UX_FLOWS_MD.read_text()
    for flow in ALL_FLOWS:
        assert flow.spec in text, f"{flow.id}: spec heading missing from UX_FLOWS.md: {flow.spec!r}"
        for feat in flow.features:
            assert feat in text, f"{flow.id}: feature heading missing from UX_FLOWS.md: {feat!r}"


def test_all_flow_ids_unique():
    assert len({f.id for f in ALL_FLOWS}) == len(ALL_FLOWS)


def test_mapped_impl_and_test_paths_exist():
    """Traceability: every listed implementation and test file must exist."""
    for flow in ALL_FLOWS:
        for path in flow.impl:
            assert (_REPO / path).is_file(), f"{flow.id}: impl path missing: {path}"
        for path in flow.tests:
            assert (_REPO / path).is_file(), f"{flow.id}: test path missing: {path}"


def test_mobile_flows_are_fully_mapped():
    """New (mobile) journeys must ship complete: features, impl, tests, owner."""
    assert MOBILE_FLOWS, "expected at least one mobile journey"
    for flow in MOBILE_FLOWS:
        assert flow.surface == "mobile"
        assert flow.features and flow.impl and flow.tests, f"{flow.id}: incomplete mapping"
        assert flow.owner, f"{flow.id}: no owner"


def test_web_unmapped_count_only_shrinks():
    unmapped = [f.id for f in FLOWS if not f.tests]
    assert len(unmapped) <= _UNMAPPED_WEB_BASELINE, (
        f"unmapped web flows grew past baseline: {unmapped}"
    )


def test_list_all_flows_preserves_registry_order():
    assert [f.id for f in list_all_flows()] == [f.id for f in ALL_FLOWS]


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
    """Single-flow live smoke — the full sweep is `python -m journeys`."""
    if os.environ.get("SLO_UX_LIVE") != "1":
        pytest.skip("set SLO_UX_LIVE=1 to run live journeys")
    if not _api_ok():
        pytest.skip("API not reachable")

    import asyncio

    from journeys.runner import run

    rc = asyncio.run(run([get_flow("0-home")], headed=False, strict_errors=False))
    assert rc == 0, "home journey failed"
