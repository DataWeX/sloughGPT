"""Events tests — record, filter, export, listeners."""

import json
import os
import tempfile

from avion.events import Event, EventRecorder, EventType


class TestRecord:
    def test_record_assigns_seq(self):
        rec = EventRecorder()
        e1 = rec.record(EventType.NAVIGATE, name="/chat")
        e2 = rec.record(EventType.CLICK, name="start")
        assert (e1.seq, e2.seq) == (1, 2)
        assert len(rec) == 2

    def test_record_defaults_success(self):
        rec = EventRecorder()
        e = rec.record(EventType.CUSTOM, name="x")
        assert e.success is True
        assert isinstance(e, Event)

    def test_record_failure_carries_error(self):
        rec = EventRecorder()
        rec.record(EventType.ELEMENT_NOT_FOUND, name="missing", success=False, error="nope")
        assert rec.failed()[0].error == "nope"


class TestFilter:
    def test_of_type(self):
        rec = EventRecorder()
        rec.record(EventType.CLICK, name="a")
        rec.record(EventType.FILL, name="b")
        rec.record(EventType.CLICK, name="c")
        assert len(rec.of_type(EventType.CLICK)) == 2

    def test_failed_only(self):
        rec = EventRecorder()
        rec.record(EventType.CLICK, name="a")
        rec.record(EventType.CLICK, name="b", success=False)
        assert [e.name for e in rec.failed()] == ["b"]

    def test_stats(self):
        rec = EventRecorder(session_name="s")
        rec.record(EventType.CLICK, name="a")
        rec.record(EventType.CLICK, name="b", success=False)
        s = rec.stats()
        assert s == {
            "session": "s",
            "total": 2,
            "failed": 1,
            "by_type": {"click": 2},
        }


class TestListenersExport:
    def test_listener_fires_and_survives_errors(self):
        rec = EventRecorder()
        seen = []
        rec.on_event(seen.append)
        rec.on_event(lambda e: 1 / 0)
        rec.record(EventType.CLICK, name="a")
        assert [e.name for e in seen] == ["a"]

    def test_screenshot_kept_out_of_json(self):
        rec = EventRecorder()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "events.json")
            rec.record(EventType.SCREENSHOT, name="s", screenshot=b"raw-bytes")
            rec.save(p)
            data = json.load(open(p))
            assert data["events"][0]["has_screenshot"] is True
            assert "raw-bytes" not in json.dumps(data)

    def test_clear(self):
        rec = EventRecorder()
        rec.record(EventType.CLICK)
        rec.clear()
        assert len(rec) == 0
