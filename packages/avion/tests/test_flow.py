"""Flow-map tests — the model half of "map out flows".

Pure data, no UI: builds a FlowGraph from recorded events (both Event objects
and the dicts EventRecorder.save emits) and checks the graph shape, the picker
picks, navigation segmentation, and the two serialisations.
"""

from __future__ import annotations

import json

from avion.events.models import Event, EventType
from avion.events.recorder import EventRecorder
from avion.flow import FlowStep, build_flow, load_flow


def _ev(seq, etype, name="", success=True, data=None):
    return Event(seq=seq, type=etype, name=name, success=success, data=data or {})


# ── build_flow from Event objects ───────────────────────────────────────────


def test_build_flow_skips_session_lifecycle_events():
    events = [
        _ev(0, EventType.SESSION_STARTED, "session_start"),
        _ev(1, EventType.NAVIGATE, "go /home"),
        _ev(2, EventType.SELECT, "pick <button>"),
        _ev(3, EventType.SESSION_ENDED, "session_end"),
    ]
    g = build_flow(events, session_name="s")
    assert [s.type for s in g.steps] == ["navigate", "select"]


def test_build_flow_from_dicts_matches_events():
    """The same flow, built from Event objects or their dict form, is identical."""
    events = [_ev(1, EventType.NAVIGATE, "go"), _ev(2, EventType.CLICK, "click")]
    dicts = [e.to_dict() for e in events]
    from_events = build_flow(events)
    from_dicts = build_flow(dicts)
    assert [s.type for s in from_events.steps] == [s.type for s in from_dicts.steps]
    assert [s.seq for s in from_events.steps] == [s.seq for s in from_dicts.steps]


def test_build_flow_from_recorder_roundtrip(tmp_path):
    """EventRecorder.save -> load_flow reproduces the flow."""
    rec = EventRecorder(session_name="rt")
    rec.record(EventType.NAVIGATE, name="go /a")
    rec.record(EventType.SELECT, name="pick", data={"tag": "button"})
    path = tmp_path / "flow.json"
    rec.save(str(path))

    g = load_flow(str(path))
    assert g.session_name == "rt"
    assert len(g.steps) == 2
    assert g.picks[0].data["tag"] == "button"


# ── graph queries ────────────────────────────────────────────────────────────


def test_picks_filters_only_select_steps():
    g = build_flow(
        [
            _ev(1, EventType.NAVIGATE, "go"),
            _ev(2, EventType.SELECT, "pick a"),
            _ev(3, EventType.SELECT, "pick b"),
            _ev(4, EventType.CLICK, "click"),
        ]
    )
    assert len(g.picks) == 2
    assert all(s.type == "select" for s in g.picks)


def test_failed_collects_failures():
    g = build_flow(
        [
            _ev(1, EventType.CLICK, "ok"),
            _ev(2, EventType.CLICK, "bad", success=False),
        ]
    )
    assert len(g.failed) == 1
    assert g.failed[0].name == "bad"


def test_segments_split_on_navigation():
    g = build_flow(
        [
            _ev(1, EventType.NAVIGATE, "go /a"),
            _ev(2, EventType.SELECT, "pick"),
            _ev(3, EventType.NAVIGATE, "go /b"),
            _ev(4, EventType.CLICK, "click"),
        ]
    )
    segs = g.segments()
    assert len(segs) == 2
    assert [s.type for s in segs[0]] == ["navigate", "select"]
    assert [s.type for s in segs[1]] == ["navigate", "click"]


def test_step_label_falls_back_to_type():
    s = FlowStep(seq=1, type="click", name="")
    assert s.label == "click"


# ── serialisation ────────────────────────────────────────────────────────────


def test_to_json_is_valid_and_carries_counts():
    g = build_flow(
        [
            _ev(1, EventType.SELECT, "pick"),
            _ev(2, EventType.CLICK, "bad", success=False),
        ]
    )
    d = json.loads(g.to_json())
    assert d["step_count"] == 2
    assert d["pick_count"] == 1
    assert d["failed_count"] == 1


def test_to_mermaid_emits_nodes_edges_and_failure_style():
    g = build_flow(
        [
            _ev(1, EventType.NAVIGATE, "go"),
            _ev(2, EventType.SELECT, "pick"),
            _ev(3, EventType.CLICK, "boom", success=False),
        ]
    )
    m = g.to_mermaid()
    assert m.startswith("flowchart TD")
    assert "s1 --> s2" in m and "s2 --> s3" in m
    # picks get the diamond shape; failures get the red style.
    assert "s2{{" in m
    assert "style s3" in m


def test_picked_step_exposes_flow_data():
    """A pick step's data carries the selector resolution the viewer shows."""
    g = build_flow(
        [
            _ev(
                1,
                EventType.SELECT,
                "pick",
                data={
                    "tag": "button",
                    "best": "testId='save'",
                    "candidates": [{"strategy": "test_id", "value": "save"}],
                },
            ),
        ]
    )
    pick = g.picks[0]
    assert pick.is_pick
    assert pick.data["best"] == "testId='save'"
