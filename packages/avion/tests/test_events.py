"""Events tests — record, filter, export, listeners."""

import base64
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

    def test_screenshot_pixels_persist_as_base64_never_raw(self):
        """save() persists the pixels (base64) — the flag-only hole is closed;
        raw byte literals still never appear in the JSON."""
        rec = EventRecorder()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "events.json")
            rec.record(EventType.SCREENSHOT, name="s", screenshot=b"raw-bytes")
            rec.save(p)
            data = json.load(open(p))
            assert data["events"][0]["has_screenshot"] is True
            saved = data["events"][0]["data"]["screenshot"]
            assert saved == base64.b64encode(b"raw-bytes").decode("ascii")
            assert "raw-bytes" not in json.dumps(data)
            # the durable file round-trips back to pixels
            back = Event.from_dict(data["events"][0])
            assert back.data["screenshot"] == b"raw-bytes"

    def test_clear(self):
        rec = EventRecorder()
        rec.record(EventType.CLICK)
        rec.clear()
        assert len(rec) == 0


class TestJournalScreenshotRoundTrip:
    def test_journal_restores_pixels_and_stores_base64(self):
        """The journal (source of truth) persists pixels as base64 JSON and
        hands them back as bytes on reload — same contract as save()."""
        from avion.events.journal import EventJournal
        from avion.events.logger import EventLogger

        with tempfile.TemporaryDirectory() as d:
            jpath = os.path.join(d, "events.jsonl")
            with EventLogger(EventJournal(jpath)) as log:
                log.log(Event(EventType.SCREENSHOT, name="s", data={"screenshot": b"\x89PNG\x00"}))
            raw_line = open(jpath).read()
            assert "b'\\x89" not in raw_line  # never the bytes repr
            with EventLogger(EventJournal(jpath)) as again:
                back = again.events()[0]
            assert back.data["screenshot"] == b"\x89PNG\x00"
