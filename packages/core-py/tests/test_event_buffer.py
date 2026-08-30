"""Tests for EventBuffer — thread-safe ring buffer for dashboard events."""

import threading
import time
from domains.infrastructure.event_buffer import EventBuffer, DashboardEvent, get_event_buffer


class TestDashboardEvent:
    def test_creation(self):
        e = DashboardEvent(ts=1.0, category="MODEL", message="loaded")
        assert e.ts == 1.0
        assert e.category == "MODEL"
        assert e.message == "loaded"

    def test_to_dict(self):
        e = DashboardEvent(ts=2.0, category="TRAIN", message="step 1")
        d = e.to_dict()
        assert d == {"ts": 2.0, "category": "TRAIN", "message": "step 1"}

    def test_frozen(self):
        e = DashboardEvent(ts=1.0, category="A", message="b")
        try:
            e.ts = 5.0
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestEventBuffer:
    def test_record_and_recent(self):
        buf = EventBuffer(maxlen=10)
        buf.record("MODEL", "loaded v1")
        buf.record("TRAIN", "step 1")
        events = buf.recent()
        assert len(events) == 2
        assert events[0]["message"] == "step 1"
        assert events[1]["message"] == "loaded v1"

    def test_maxlen_enforced(self):
        buf = EventBuffer(maxlen=3)
        for i in range(5):
            buf.record("SYSTEM", f"event {i}")
        events = buf.recent(10)
        assert len(events) == 3
        assert events[0]["message"] == "event 4"

    def test_custom_timestamp(self):
        buf = EventBuffer()
        buf.record("ERROR", "fail", ts=100.0)
        events = buf.recent()
        assert events[0]["ts"] == 100.0

    def test_clear(self):
        buf = EventBuffer()
        buf.record("MODEL", "a")
        buf.record("TRAIN", "b")
        buf.clear()
        assert buf.recent() == []

    def test_recent_n(self):
        buf = EventBuffer(maxlen=10)
        for i in range(10):
            buf.record("SYSTEM", f"e{i}")
        assert len(buf.recent(3)) == 3
        assert len(buf.recent(1)) == 1

    def test_recent_empty(self):
        buf = EventBuffer()
        assert buf.recent() == []

    def test_thread_safety(self):
        buf = EventBuffer(maxlen=200)
        errors = []

        def writer(n):
            try:
                for i in range(50):
                    buf.record("SYSTEM", f"t{n}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        events = buf.recent(200)
        assert len(events) == 200


class TestSingleton:
    def test_get_event_buffer_returns_same_instance(self):
        a = get_event_buffer()
        b = get_event_buffer()
        assert a is b
