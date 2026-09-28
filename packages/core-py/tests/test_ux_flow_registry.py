"""
Journey registry ↔ spec drift guard.

Asserts the single journey registry (scripts/run_ux_flows.py FLOWS) stays
linked to docs/UX_FLOWS.md: every Flow carries a verbatim spec heading from
the doc, and every numbered feature heading in the doc is claimed by exactly
one Flow. Fast, no browser, no live stack — runs in the default suite.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _load_runner():
    path = ROOT / "scripts" / "run_ux_flows.py"
    spec = importlib.util.spec_from_file_location("run_ux_flows", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_ux_flows"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


@pytest.fixture(scope="module")
def doc_text() -> str:
    return (ROOT / "docs" / "UX_FLOWS.md").read_text()


def test_registry_size_and_unique_ids(runner):
    ids = [f.id for f in runner.FLOWS]
    assert len(ids) == 13, f"expected 13 journeys, got {len(ids)}"
    assert len(set(ids)) == len(ids), f"duplicate flow ids: {ids}"


def test_every_flow_has_spec_anchor(runner, doc_text):
    for f in runner.FLOWS:
        assert f.spec, f"{f.id} has no spec anchor"
        assert f.spec in doc_text, f"{f.id} spec not found verbatim in docs/UX_FLOWS.md: {f.spec!r}"


def test_every_doc_feature_claimed(runner):
    doc_text = (ROOT / "docs" / "UX_FLOWS.md").read_text()
    doc_nums = set(re.findall(r"^### (\d+)\. ", doc_text, re.M))
    flow_nums = {f.id.split("-", 1)[0] for f in runner.FLOWS} - {"0"}
    assert doc_nums == flow_nums, (
        f"docs/UX_FLOWS.md feature numbers {sorted(doc_nums, key=int)} "
        f"!= registry flow numbers {sorted(flow_nums, key=int)}"
    )


def test_flow_labels_nonempty(runner):
    for f in runner.FLOWS:
        assert f.label.strip(), f"{f.id} has empty label"
        assert f.url.strip(), f"{f.id} has empty url"
