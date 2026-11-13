"""EventLogger tests — journal-first ordering, sink isolation, degrade.

Also covers the recorder-as-view semantics and the fact that journal
writes alone never reach stdlib (that is ``StdlibSink``'s job, optional).
"""

from __future__ import annotations

import logging

from avion.events.bus import EventBus
from avion.events.journal import EventJournal
from avion.events.logger import EventLogger
from avion.events.models import Event, EventType
from avion.events.recorder import EventRecorder
from avion.events.sinks import BusSink, CallbackSink, StdlibSink
from avion.logging.structured import StructuredLogger


class Boom:
    """A sink that always fails."""

    @property
    def name(self) -> str:
        return "boom"

    def emit(self, event: Event) -> None:
        raise RuntimeError("sink down")


class Collect:
    """A well-behaved sink that remembers what it saw."""

    def __init__(self, name: str = "collect"):
        self._name = name
        self.seen: list[Event] = []

    @property
    def name(self) -> str:
        return self._name

    def emit(self, event: Event) -> None:
        self.seen.append(event)


class BrokenJournal:
    """Duck-typed journal whose append always fails (disk full)."""

    path = "broken"

    def append(self, event: Event) -> Event:
        raise OSError("disk full")

    def events(self) -> list[Event]:
        return []

    def close(self) -> None:
        pass


class TestJournalFirst:
    def test_event_survives_all_sinks_failing(self, tmp_path):
        journal = EventJournal(tmp_path / "events.jsonl")
        collect = Collect()
        logger = EventLogger(journal, [Boom(), collect])

        logger.log(Event(EventType.CLICK, name="x"))  # must not raise

        assert [e.name for e in journal.events()] == ["x"]
        assert [e.name for e in collect.seen] == ["x"]
        assert logger.sink_failures("boom") == 1

    def test_sibling_sinks_are_not_affected_by_a_failure(self):
        first, second = Collect("first"), Collect("second")
        logger = EventLogger(None, [Boom(), first, second])
        logger.log(Event(EventType.CLICK, name="x"))
        assert len(first.seen) == 1
        assert len(second.seen) == 1

    def test_failure_is_counted_not_swallowed_silently(self):
        logger = EventLogger(None, [Boom()])
        for _ in range(2):
            logger.log(Event(EventType.CLICK))
        assert logger.sink_failures("boom") == 2
        assert logger.sink_status()["boom"] == {"failures": 2, "disabled": False}

    def test_journal_failure_is_counted_and_sinks_still_run(self):
        collect = Collect()
        logger = EventLogger(BrokenJournal(), [collect])
        logger.log(Event(EventType.CLICK, name="x"))
        assert logger.journal_failures == 1
        assert len(collect.seen) == 1

    def test_stats_reports_both_health_sides(self, tmp_path):
        logger = EventLogger(EventJournal(tmp_path / "e.jsonl"), [Boom()])
        logger.log(Event(EventType.CLICK))
        stats = logger.stats()
        assert stats["journal"].endswith("e.jsonl")
        assert stats["journal_failures"] == 0
        assert stats["sinks"]["boom"]["failures"] == 1


class TestDegradation:
    def test_degrades_after_consecutive_failures(self):
        logger = EventLogger(None, [Boom()], degrade_after=3)
        for _ in range(5):
            logger.log(Event(EventType.CLICK))
        # Disabled at 3: the two later events never reached it.
        assert logger.sink_failures("boom") == 3
        assert logger.sink_status()["boom"]["disabled"] is True

    def test_reset_revives_a_degraded_sink(self):
        logger = EventLogger(None, [Boom()], degrade_after=2)
        for _ in range(3):
            logger.log(Event(EventType.CLICK))
        assert logger.sink_status()["boom"]["disabled"] is True
        assert logger.reset_sink("boom") is True
        logger.log(Event(EventType.CLICK))
        assert logger.sink_failures("boom") == 3

    def test_reset_unknown_sink_is_false(self):
        assert EventLogger().reset_sink("nope") is False

    def test_success_clears_the_consecutive_count(self):
        flaky = Collect()  # healthy sink: counter must never build up
        logger = EventLogger(None, [flaky], degrade_after=2)
        for _ in range(10):
            logger.log(Event(EventType.CLICK))
        assert logger.sink_status()["collect"] == {"failures": 0, "disabled": False}

    def test_reregistering_revives(self):
        logger = EventLogger(None, [Boom()], degrade_after=1)
        logger.log(Event(EventType.CLICK))
        assert logger.sink_status()["boom"]["disabled"] is True
        logger.register_sink(Boom())
        assert logger.sink_status()["boom"]["disabled"] is False

    def test_unregister_removes_sink(self):
        logger = EventLogger(None, [Collect()])
        assert logger.unregister_sink("collect") is True
        assert logger.unregister_sink("collect") is False
        assert logger.sinks == []


class TestSinks:
    def test_callback_sink(self):
        seen: list[str] = []
        logger = EventLogger(None, [CallbackSink(lambda e: seen.append(e.name))])
        logger.log(Event(EventType.CLICK, name="target"))
        assert seen == ["target"]

    def test_bus_sink_publishes_to_subscribers(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("click", lambda e: received.append(e))
        logger = EventLogger(None, [BusSink(bus)])
        logger.log(Event(EventType.CLICK, name="via-bus"))
        assert [e.name for e in received] == ["via-bus"]

    def test_bus_sink_fails_loudly_when_bus_has_no_publish(self):
        logger = EventLogger(None, [BusSink(object())])
        logger.log(Event(EventType.CLICK))  # counted, not raised
        assert logger.sink_failures("bus") == 1

    def test_stdlib_sink_bridges_with_payload(self):
        capture = _Capture()
        py_logger = logging.getLogger("avion.test_bridge")
        py_logger.addHandler(capture)
        try:
            StdlibSink("avion.test_bridge").emit(
                Event(EventType.CLICK, name="x", success=False, error="boom")
            )
        finally:
            py_logger.removeHandler(capture)
        record = capture.records[0]
        assert record.levelno == logging.ERROR
        assert record.avion_event["name"] == "x"
        assert record.avion_event["error"] == "boom"

    def test_stdlib_sink_success_maps_to_configured_level(self):
        capture = _Capture()
        py_logger = logging.getLogger("avion.test_bridge_ok")
        py_logger.setLevel(logging.DEBUG)
        py_logger.addHandler(capture)
        try:
            StdlibSink("avion.test_bridge_ok").emit(Event(EventType.CLICK, name="x"))
        finally:
            py_logger.removeHandler(capture)
            py_logger.setLevel(logging.NOTSET)
        assert capture.records[0].levelno == logging.INFO

    def test_stdlib_sink_never_carries_raw_screenshot_bytes(self):
        capture = _Capture()
        py_logger = logging.getLogger("avion.test_bridge_shot")
        py_logger.setLevel(logging.DEBUG)
        py_logger.addHandler(capture)
        try:
            StdlibSink("avion.test_bridge_shot").emit(
                Event(EventType.SCREENSHOT, name="shot", data={"screenshot": b"\x89PNG" * 1000})
            )
        finally:
            py_logger.removeHandler(capture)
            py_logger.setLevel(logging.NOTSET)
        payload = capture.records[0].avion_event
        assert payload["has_screenshot"] is True
        assert "screenshot" not in payload.get("data", {})
        assert b"\x89PNG" not in repr(payload).encode()


class TestNoAccidentalStdlib:
    def test_journal_writes_alone_never_reach_stdlib(self):
        capture = _Capture()
        py_logger = logging.getLogger("arken.isolation_probe")
        py_logger.addHandler(capture)
        try:
            logger = EventLogger(None, [Collect()])  # no StdlibSink
            logger.log(Event(EventType.CLICK, name="x"))
        finally:
            py_logger.removeHandler(capture)
        assert capture.records == []


class TestRecorderIsAView:
    def test_record_goes_through_the_journal(self, tmp_path):
        journal = EventJournal(tmp_path / "events.jsonl")
        rec = EventRecorder("s", logger=EventLogger(journal))
        event = rec.record(EventType.NAVIGATE, name="/chat")
        assert event.seq == 1
        assert [e.name for e in journal.events()] == ["/chat"]

    def test_seq_comes_from_the_logger_not_the_recorder(self):
        rec = EventRecorder("s")
        first = rec.record(EventType.CLICK, name="a")
        second = rec.record(EventType.CLICK, name="b")
        assert (first.seq, second.seq) == (1, 2)

    def test_view_is_seeded_from_the_journal_on_reopen(self, tmp_path):
        path = tmp_path / "events.jsonl"
        first = EventRecorder("s", logger=EventLogger(EventJournal(path)))
        first.record(EventType.CLICK, name="old-1")
        first.record(EventType.CLICK, name="old-2")
        first.logger.close()

        reopened = EventRecorder("s", logger=EventLogger(EventJournal(path)))
        assert len(reopened) == 2
        assert reopened.record(EventType.CLICK, name="new").seq == 3

    def test_clear_clears_the_view_not_the_journal(self, tmp_path):
        journal = EventJournal(tmp_path / "events.jsonl")
        rec = EventRecorder("s", logger=EventLogger(journal))
        rec.record(EventType.CLICK, name="kept-forever")
        rec.clear()
        assert len(rec) == 0
        assert [e.name for e in journal.events()] == ["kept-forever"]


class TestStructuredLoggerFields:
    def test_fields_reach_the_stdlib_record(self):
        capture = _Capture()
        py_logger = logging.getLogger("arken.t_fields")
        py_logger.setLevel(logging.DEBUG)
        py_logger.addHandler(capture)
        try:
            log = StructuredLogger("t_fields")
            log.set_context(run="1")
            log.info("hello", foo="bar")
        finally:
            py_logger.removeHandler(capture)
            py_logger.setLevel(logging.NOTSET)
        record = capture.records[0]
        assert record.getMessage() == "hello"
        assert record.avion_fields["run"] == "1"
        assert record.avion_fields["foo"] == "bar"
        assert record.avion_fields["kind"] == "info"

    def test_warning_and_error_fields_reach_the_record(self):
        capture = _Capture()
        py_logger = logging.getLogger("arken.t_fields_lvl")
        py_logger.addHandler(capture)
        try:
            log = StructuredLogger("t_fields_lvl")
            log.warning("careful", code=17)
            log.error("broke", reason="disk")
        finally:
            py_logger.removeHandler(capture)
        assert capture.records[0].avion_fields["code"] == 17
        assert capture.records[1].avion_fields["reason"] == "disk"
        assert capture.records[0].levelno == logging.WARNING
        assert capture.records[1].levelno == logging.ERROR

    def test_in_memory_entries_still_keep_every_field(self):
        log = StructuredLogger("t_entries")
        log.set_context(run="1")
        log.info("hello", foo="bar")
        entry = log.entries[0]
        assert entry["message"] == "hello"
        assert entry["run"] == "1"
        assert entry["foo"] == "bar"

    def test_journals_when_attached(self, tmp_path):
        journal = EventJournal(tmp_path / "events.jsonl")
        log = StructuredLogger("tj", logger=EventLogger(journal))
        log.set_context(run="42")
        log.info("hello", foo="bar")
        log.error("broke")
        events = journal.events()
        assert [e.type for e in events] == [EventType.CUSTOM, EventType.CUSTOM]
        assert [e.name for e in events] == ["info", "error"]
        assert events[0].data["message"] == "hello"
        assert events[0].data["foo"] == "bar"
        assert events[0].data["run"] == "42"
        assert events[0].success is True
        assert events[1].success is False
        assert events[1].error == "broke"

    def test_explicit_success_field_wins_over_level(self, tmp_path):
        journal = EventJournal(tmp_path / "events.jsonl")
        log = StructuredLogger("ts", logger=EventLogger(journal))
        log.navigation("/chat", False)  # warning level, explicit success=False
        log.task_end("chat", True)  # info level, explicit success=True
        events = journal.events()
        assert events[0].success is False
        assert events[1].success is True

    def test_without_a_logger_no_journal_side_effect(self, tmp_path):
        capture = _Capture()
        py_logger = logging.getLogger("arken.t_nojournal")
        py_logger.setLevel(logging.DEBUG)
        py_logger.addHandler(capture)
        try:
            StructuredLogger("t_nojournal").info("x")
        finally:
            py_logger.removeHandler(capture)
            py_logger.setLevel(logging.NOTSET)
        assert len(capture.records) == 1  # exactly one line, no duplication


class TestSessionWiring:
    """The session must route BOTH views through ONE logger."""

    def test_views_share_one_event_logger_and_journal(self, tmp_path):
        from avion.core.session import Arken

        arken = Arken(journal_path=str(tmp_path / "events.jsonl"))
        assert arken.recorder.logger is arken.event_logger
        assert arken.logger.logger is arken.event_logger
        assert arken.event_logger.journal is not None
        assert str(arken.event_logger.journal.path).endswith("events.jsonl")

    def test_default_session_is_in_memory_only(self):
        from avion.core.session import Arken

        arken = Arken()
        assert arken.event_logger.journal is None
        assert arken.event_logger.journal_failures == 0

    def test_session_events_land_in_the_journal(self, tmp_path):
        from avion.core.session import Arken

        path = tmp_path / "events.jsonl"
        arken = Arken(journal_path=str(path))
        arken.recorder.record(EventType.CLICK, name="btn")
        arken.logger.info("did a thing", step=3)
        names = [e.name for e in arken.event_logger.events()]
        assert "btn" in names
        assert "info" in names
        # one seq space: journal seqs are strictly increasing
        seqs = [e.seq for e in arken.event_logger.events()]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
