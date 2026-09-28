"""
Live UX journeys — one pytest test per Flow in scripts/run_ux_flows.py.

Module-scoped fixture walks all 13 docs/UX_FLOWS.md journeys in one browser
session and writes a per-flow report; each test asserts its own flow's result.
Deselected from default/CI runs (marker `browser`); run explicitly:

    .venv/bin/python -m pytest -m browser packages/core-py/tests/test_ux_flows_browser.py

Requires a live stack (uvicorn :8000 + vite :5175) and Firefox (avion);
skips when the stack is down.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import pathlib
import sys

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.live]

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _load_runner():
    path = ROOT / "scripts" / "run_ux_flows.py"
    spec = importlib.util.spec_from_file_location("run_ux_flows", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_ux_flows"] = mod
    spec.loader.exec_module(mod)
    return mod


RUNNER = _load_runner()


@pytest.fixture(scope="module")
def flow_report(tmp_path_factory):
    if not (RUNNER._http_ok(f"{RUNNER.API}/health") and RUNNER._http_ok(f"{RUNNER.WEB}/")):
        pytest.skip(f"stack not running (need {RUNNER.API} + {RUNNER.WEB})")
    report_path = str(tmp_path_factory.mktemp("ux") / "ux-flows-report.json")
    saved = RUNNER.REPORT
    RUNNER.REPORT = report_path
    try:
        asyncio.run(RUNNER.run(list(RUNNER.FLOWS), headed=False, strict_errors=False))
        with open(report_path) as f:
            return json.load(f)
    finally:
        RUNNER.REPORT = saved


@pytest.mark.parametrize("flow", RUNNER.FLOWS, ids=lambda f: f.id)
def test_flow(flow, flow_report):
    recs = {r.get("task"): r for r in flow_report["flows"]}
    assert flow.id in recs, f"flow never ran: {flow.id}"
    rec = recs[flow.id]
    assert rec["status"] == "passed", f"{flow.id} ({flow.spec}): {rec.get('error')}"
