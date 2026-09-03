"""Tests for EventBuffer — thread-safe ring buffer for dashboard events."""
from __future__ import annotations

from domains.infrastructure.event_buffer import DashboardEvent, EventBuffer, get_event_buffer


class TestDashboardEvent:
    def test_to_dict(self):
        e = DashboardEvent(ts=1.0, category="TRAIN", message="step 1")
        d = e.to_dict()
        assert d["ts"] == 1.0
        assert d["category"] == "TRAIN"
        assert d["message"] == "step 1"

    def test_frozen(self):
        e = DashboardEvent(ts=1.0, category="X", message="Y")
        try:
            e.category = "Z"
            assert False, "should be frozen"
        except AttributeError:
            pass


class TestEventBuffer:
    def test_record_and_recent(self):
        buf = EventBuffer(maxlen=10)
        buf.record("TRAIN", "step 1")
        buf.record("TRAIN", "step 2")
        recent = buf.recent(5)
        assert len(recent) == 2
        assert recent[0]["message"] == "step 2"  # newest first
        assert recent[1]["message"] == "step 1"

    def test_maxlen(self):
        buf = EventBuffer(maxlen=3)
        for i in range(5):
            buf.record("SYS", f"event {i}")
        recent = buf.recent(10)
        assert len(recent) == 3
        assert recent[0]["message"] == "event 4"

    def test_clear(self):
        buf = EventBuffer()
        buf.record("SYS", "event")
        buf.clear()
        assert buf.recent() == []

    def test_recent_empty(self):
        buf = EventBuffer()
        assert buf.recent() == []

    def test_recent_n(self):
        buf = EventBuffer(maxlen=50)
        for i in range(10):
            buf.record("SYS", f"e{i}")
        assert len(buf.recent(3)) == 3

    def test_thread_safety(self):
        import threading
        buf = EventBuffer(maxlen=100)
        def worker(n):
            for i in range(10):
                buf.record("SYS", f"t{n}-e{i}")
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Should have 100 events (buffer full)
        assert len(buf.recent(200)) == 100


class TestGetEventBuffer:
    def test_singleton(self):
        a = get_event_buffer()
        b = get_event_buffer()
        assert a is b
