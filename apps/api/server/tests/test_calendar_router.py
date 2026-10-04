"""Calendar router — exact parity with the retired Next /api/calendar handlers.

The Next handler (``apps/web/app/api/calendar``) read/wrote
``<project root>/.calendar/events.jsonl`` with fixed create defaults and a
``?date=`` filter; the port keeps all of it (overridable via
``APP_PLANNER_CALENDAR_DIR``), answers 201 on create, and skips malformed
lines instead of failing the whole read.

Descriptor names exercised here (contract gate):
    calendar.events.read, calendar.events.create.
"""

from __future__ import annotations

import pytest
from infrastructure.contract import REGISTRY
from test_support import _data, get_test_client

DESCRIPTORS = (
    "calendar.events.read",
    "calendar.events.create",
)


@pytest.fixture(autouse=True)
def calendar_tmp(tmp_path, monkeypatch):
    events_dir = tmp_path / "calendar"
    monkeypatch.setenv("APP_PLANNER_CALENDAR_DIR", str(events_dir))
    yield events_dir


def test_events_read_starts_empty_under_the_shared_envelope():
    resp = get_test_client().get("/api/calendar/events")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert _data(resp) == {"events": []}


def test_event_create_answers_201_with_the_next_defaults():
    resp = get_test_client().post(
        "/api/calendar/events", json={"title": "Standup", "date": "2026-10-06"}
    )
    assert resp.status_code == 201
    event = _data(resp)["event"]
    assert event["title"] == "Standup"
    assert event["date"] == "2026-10-06"
    assert event["description"] == ""
    assert event["start_time"] == "09:00"  # Next createEvent default
    assert event["end_time"] == "10:00"
    assert event["color"] == "primary"
    assert event["id"].startswith("event-")
    assert event["created_at"] and event["updated_at"]


def test_event_create_accepts_explicit_times_and_color():
    resp = get_test_client().post(
        "/api/calendar/events",
        json={
            "title": "Review",
            "date": "2026-10-07",
            "description": "PR review",
            "start_time": "14:00",
            "end_time": "15:00",
            "color": "secondary",
        },
    )
    event = _data(resp)["event"]
    assert event["start_time"] == "14:00"
    assert event["end_time"] == "15:00"
    assert event["color"] == "secondary"
    assert event["description"] == "PR review"


def test_event_create_missing_date_is_a_contract_violation():
    """Next answered 400 {error}; the projection answers 422 (same rejection)."""
    resp = get_test_client().post("/api/calendar/events", json={"title": "No date"})
    assert resp.status_code == 422


def test_date_filter_returns_exact_day_matches(calendar_tmp):
    client = get_test_client()
    client.post("/api/calendar/events", json={"title": "A", "date": "2026-10-06"})
    client.post("/api/calendar/events", json={"title": "B", "date": "2026-10-07"})
    client.post("/api/calendar/events", json={"title": "C", "date": "2026-10-07"})

    j6 = _data(client.get("/api/calendar/events", params={"date": "2026-10-06"}))
    assert [e["title"] for e in j6["events"]] == ["A"]

    j7 = _data(client.get("/api/calendar/events", params={"date": "2026-10-07"}))
    assert [e["title"] for e in j7["events"]] == ["B", "C"]

    none = _data(client.get("/api/calendar/events", params={"date": "2030-01-01"}))
    assert none["events"] == []


def test_persisted_events_round_trip_with_malformed_lines_skipped(calendar_tmp):
    client = get_test_client()
    client.post("/api/calendar/events", json={"title": "Kept", "date": "2026-10-06"})

    events_file = calendar_tmp / "events.jsonl"
    raw = events_file.read_text(encoding="utf-8")
    events_file.write_text(raw + "not json at all\n" + '{"id":"x"}\n', encoding="utf-8")

    events = _data(client.get("/api/calendar/events"))["events"]
    assert [e["title"] for e in events] == ["Kept"]  # Next parity: skip, don't fail


def test_every_descriptor_is_projected_and_served():
    names = {e["name"] for e in REGISTRY.routes()}
    for name in DESCRIPTORS:
        assert name in names, name
    served = {(e["method"], e["path"]) for e in REGISTRY.routes()}
    assert ("GET", "/api/calendar/events") in served
    assert ("POST", "/api/calendar/events") in served
