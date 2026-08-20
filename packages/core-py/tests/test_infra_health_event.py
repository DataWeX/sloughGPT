"""Tests for domains.infrastructure.health_flow — Severity, Diagnosis, HealthFlowResult; domains.infrastructure.event_bus — EventPriority, Event, Subscription."""

from domains.infrastructure.health_flow import Severity, Diagnosis, HealthFlowResult
from domains.infrastructure.event_bus import EventPriority, Event, Subscription


class TestSeverity:
    def test_all_members(self):
        assert len(Severity) == 4
    def test_values(self):
        assert Severity.OK.value == "ok"
        assert Severity.INFO.value == "info"
        assert Severity.WARN.value == "warn"
        assert Severity.CRITICAL.value == "critical"


class TestDiagnosis:
    def test_fields(self):
        d = Diagnosis(check="errors", severity=Severity.OK, score=100.0, message="all good")
        assert d.check == "errors"
        assert d.severity == Severity.OK
        assert d.score == 100.0

    def test_defaults(self):
        d = Diagnosis(check="x", severity=Severity.INFO, score=50.0, message="m")
        assert d.detail == ""


class TestHealthFlowResult:
    def test_fields(self):
        r = HealthFlowResult(score=90, status="healthy", summary="looks good")
        assert r.score == 90
        assert r.status == "healthy"
        assert r.model_loaded is False

    def test_defaults(self):
        r = HealthFlowResult(score=100, status="healthy", summary="ok")
        assert r.diagnoses == []
        assert r.model_type == ""


class TestEventPriority:
    def test_all_members(self):
        assert len(EventPriority) == 4
    def test_values(self):
        assert EventPriority.MONITOR.value == 0
        assert EventPriority.NORMAL.value == 1
        assert EventPriority.HIGH.value == 2
        assert EventPriority.CRITICAL.value == 3


class TestEvent:
    def test_fields(self):
        e = Event(name="test", data={"key": "val"}, source="unit")
        assert e.name == "test"
        assert e.data["key"] == "val"
        assert e.source == "unit"

    def test_defaults(self):
        e = Event(name="test")
        assert e.data == {}
        assert e.id.startswith("evt_")
        assert e.timestamp > 0


class TestSubscription:
    def test_defaults(self):
        s = Subscription(handler=lambda: None)
        assert s.priority == EventPriority.NORMAL
        assert s.once is False

    def test_custom(self):
        s = Subscription(handler=lambda: None, priority=EventPriority.HIGH, once=True)
        assert s.priority == EventPriority.HIGH
        assert s.once is True
