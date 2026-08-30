"""Tests for domains.infrastructure.health_flow — Severity, Diagnosis, HealthFlowResult; domains.infrastructure.event_bus — EventPriority, Event, Subscription."""

import asyncio
import threading
import time

from domains.infrastructure.health_flow import (
    Severity,
    Diagnosis,
    HealthFlowResult,
    _check_errors,
    _check_latency,
    _check_throughput,
    _check_model,
    _check_uptime,
    _check_resources,
    run_health_flow,
)
from domains.infrastructure.event_bus import (
    EventPriority,
    Event,
    Subscription,
    EventBus,
    get_event_bus,
    set_event_bus,
    reset_event_bus,
)


class TestSeverity:
    def test_all_members(self):
        assert len(Severity) == 4

    def test_values(self):
        assert Severity.OK.value == "ok"
        assert Severity.INFO.value == "info"
        assert Severity.WARN.value == "warn"
        assert Severity.CRITICAL.value == "critical"

    def test_is_str_enum(self):
        assert isinstance(Severity.OK, str)
        assert Severity.OK == "ok"

    def test_comparison_with_non_severity(self):
        assert Severity.OK != "other"

    def test_severity_repr(self):
        assert repr(Severity.OK) == "<Severity.OK: 'ok'>"

    def test_severity_in_set(self):
        s = {Severity.OK, Severity.INFO, Severity.WARN, Severity.CRITICAL}
        assert len(s) == 4

    def test_severity_as_dict_key(self):
        d = {Severity.OK: "good", Severity.CRITICAL: "bad"}
        assert d[Severity.OK] == "good"

    def test_severity_from_value(self):
        assert Severity("ok") is Severity.OK
        assert Severity("critical") is Severity.CRITICAL

    def test_severity_invalid_value(self):
        import pytest
        with pytest.raises(ValueError):
            Severity("nonexistent")

    def test_severity_all_values_are_strings(self):
        for s in Severity:
            assert isinstance(s.value, str)

    def test_severity_unique_values(self):
        values = [s.value for s in Severity]
        assert len(values) == len(set(values))

    def test_severity_name_attribute(self):
        assert Severity.OK.name == "OK"
        assert Severity.WARN.name == "WARN"

    def test_severity_members_are_ordered(self):
        assert list(Severity) == [Severity.OK, Severity.INFO, Severity.WARN, Severity.CRITICAL]


class TestDiagnosis:
    def test_fields(self):
        d = Diagnosis(check="errors", severity=Severity.OK, score=100.0, message="all good")
        assert d.check == "errors"
        assert d.severity == Severity.OK
        assert d.score == 100.0

    def test_defaults(self):
        d = Diagnosis(check="x", severity=Severity.INFO, score=50.0, message="m")
        assert d.detail == ""

    def test_detail_override(self):
        d = Diagnosis(check="c", severity=Severity.WARN, score=70.0, message="msg", detail="extra info")
        assert d.detail == "extra info"

    def test_zero_score(self):
        d = Diagnosis(check="x", severity=Severity.CRITICAL, score=0.0, message="fail")
        assert d.score == 0.0

    def test_equality(self):
        d1 = Diagnosis(check="x", severity=Severity.OK, score=100.0, message="m")
        d2 = Diagnosis(check="x", severity=Severity.OK, score=100.0, message="m")
        assert d1 == d2

    def test_inequality_different_check(self):
        d1 = Diagnosis(check="a", severity=Severity.OK, score=100.0, message="m")
        d2 = Diagnosis(check="b", severity=Severity.OK, score=100.0, message="m")
        assert d1 != d2

    def test_inequality_different_severity(self):
        d1 = Diagnosis(check="x", severity=Severity.OK, score=100.0, message="m")
        d2 = Diagnosis(check="x", severity=Severity.WARN, score=100.0, message="m")
        assert d1 != d2

    def test_negative_score(self):
        d = Diagnosis(check="x", severity=Severity.CRITICAL, score=-10.0, message="fail")
        assert d.score == -10.0

    def test_max_score(self):
        d = Diagnosis(check="x", severity=Severity.OK, score=100.0, message="perfect")
        assert d.score == 100.0

    def test_empty_message(self):
        d = Diagnosis(check="x", severity=Severity.OK, score=100.0, message="")
        assert d.message == ""

    def test_all_severity_values(self):
        for sev in Severity:
            d = Diagnosis(check="x", severity=sev, score=50.0, message="m")
            assert d.severity == sev

    def test_score_boundary_float(self):
        d = Diagnosis(check="x", severity=Severity.OK, score=99.99, message="m")
        assert d.score == 99.99


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

    def test_with_diagnoses(self):
        d = Diagnosis(check="errors", severity=Severity.OK, score=100.0, message="fine")
        r = HealthFlowResult(score=100, status="healthy", summary="fine", diagnoses=[d])
        assert len(r.diagnoses) == 1
        assert r.diagnoses[0].check == "errors"

    def test_model_loaded(self):
        r = HealthFlowResult(score=100, status="healthy", summary="ok", model_loaded=True, model_type="soul")
        assert r.model_loaded is True
        assert r.model_type == "soul"

    def test_multiple_diagnoses(self):
        d1 = Diagnosis(check="a", severity=Severity.OK, score=100.0, message="m1")
        d2 = Diagnosis(check="b", severity=Severity.WARN, score=50.0, message="m2")
        r = HealthFlowResult(score=75, status="degraded", summary="mixed", diagnoses=[d1, d2])
        assert len(r.diagnoses) == 2

    def test_score_zero(self):
        r = HealthFlowResult(score=0, status="unhealthy", summary="dead")
        assert r.score == 0

    def test_status_degraded(self):
        r = HealthFlowResult(score=60, status="degraded", summary="slow")
        assert r.status == "degraded"

    def test_empty_string_summary(self):
        r = HealthFlowResult(score=50, status="degraded", summary="")
        assert r.summary == ""

    def test_many_diagnoses(self):
        diags = [Diagnosis(check=str(i), severity=Severity.OK, score=100.0, message=str(i)) for i in range(20)]
        r = HealthFlowResult(score=80, status="healthy", summary="ok", diagnoses=diags)
        assert len(r.diagnoses) == 20


class TestCheckErrors:
    def test_no_requests(self):
        d = _check_errors(0, 0)
        assert d.severity == Severity.OK
        assert d.score == 100.0

    def test_zero_errors(self):
        d = _check_errors(100, 0)
        assert d.severity == Severity.OK
        assert d.score == 100.0

    def test_rare_errors(self):
        d = _check_errors(1000, 5)
        assert d.severity == Severity.OK
        assert d.score > 80

    def test_degrading_errors(self):
        d = _check_errors(100, 2)
        assert d.severity in (Severity.OK, Severity.WARN)

    def test_failing_errors(self):
        d = _check_errors(100, 10)
        assert d.severity == Severity.CRITICAL
        assert d.score < 50

    def test_all_errors(self):
        d = _check_errors(100, 100)
        assert d.severity == Severity.CRITICAL
        assert d.score == 0.0

    def test_check_name(self):
        d = _check_errors(100, 0)
        assert d.check == "errors"

    def test_single_error(self):
        d = _check_errors(100, 1)
        assert d.severity == Severity.OK

    def test_half_errors(self):
        d = _check_errors(100, 50)
        assert d.severity == Severity.CRITICAL

    def test_boundary_1_percent(self):
        d = _check_errors(100, 1)
        assert d.severity == Severity.OK

    def test_message_contains_counts(self):
        d = _check_errors(200, 10)
        assert "10/200" in d.message

    def test_one_request_one_error(self):
        d = _check_errors(1, 1)
        assert d.severity == Severity.CRITICAL
        assert d.score == 0.0

    def test_zero_requests_message(self):
        d = _check_errors(0, 0)
        assert "No requests" in d.message

    def test_large_numbers(self):
        d = _check_errors(100000, 1000)
        assert d.check == "errors"

    def test_boundary_5_percent(self):
        d = _check_errors(100, 5)
        assert d.score == 0.0


class TestCheckLatency:
    def test_no_requests(self):
        d = _check_latency(0)
        assert d.severity == Severity.OK
        assert d.score == 100.0

    def test_snappy(self):
        d = _check_latency(200)
        assert d.severity == Severity.OK

    def test_fine(self):
        d = _check_latency(500)
        assert d.severity == Severity.OK

    def test_slow(self):
        d = _check_latency(1200)
        assert d.severity in (Severity.WARN, Severity.CRITICAL)

    def test_very_slow(self):
        d = _check_latency(2000)
        assert d.severity == Severity.CRITICAL
        assert d.score == 0.0

    def test_check_name(self):
        d = _check_latency(100)
        assert d.check == "latency"

    def test_boundary_300ms(self):
        d = _check_latency(300)
        assert d.severity == Severity.OK

    def test_boundary_800ms(self):
        d = _check_latency(800)
        assert d.severity in (Severity.OK, Severity.WARN)

    def test_boundary_1500ms(self):
        d = _check_latency(1500)
        assert d.severity == Severity.CRITICAL

    def test_message_contains_ms(self):
        d = _check_latency(500)
        assert "500ms" in d.message

    def test_zero_latency_message(self):
        d = _check_latency(0)
        assert "No requests" in d.message

    def test_extreme_latency(self):
        d = _check_latency(10000)
        assert d.score == 0.0

    def test_just_below_300(self):
        d = _check_latency(299)
        assert d.severity == Severity.OK

    def test_just_above_1500(self):
        d = _check_latency(1501)
        assert d.severity == Severity.CRITICAL


class TestCheckThroughput:
    def test_no_data(self):
        d = _check_throughput(0)
        assert d.severity == Severity.INFO
        assert d.score == 50

    def test_fast(self):
        d = _check_throughput(50)
        assert d.severity == Severity.OK

    def test_ok(self):
        d = _check_throughput(30)
        assert d.severity == Severity.WARN

    def test_slow(self):
        d = _check_throughput(7)
        assert d.severity == Severity.CRITICAL

    def test_very_slow(self):
        d = _check_throughput(3)
        assert d.severity == Severity.CRITICAL

    def test_check_name(self):
        d = _check_throughput(30)
        assert d.check == "throughput"

    def test_boundary_5_tps(self):
        d = _check_throughput(5)
        assert d.severity == Severity.CRITICAL

    def test_boundary_30_tps(self):
        d = _check_throughput(30)
        assert d.severity == Severity.WARN

    def test_boundary_10_tps(self):
        d = _check_throughput(10)
        assert d.severity == Severity.CRITICAL

    def test_message_contains_tps(self):
        d = _check_throughput(25)
        assert "tok/s" in d.message

    def test_high_throughput_ok(self):
        d = _check_throughput(50)
        assert d.severity == Severity.OK
        assert "Fast" in d.message

    def test_medium_throughput_warn(self):
        d = _check_throughput(28)
        assert d.severity == Severity.WARN

    def test_zero_throughput_message(self):
        d = _check_throughput(0)
        assert "No generation" in d.message

    def test_extreme_throughput(self):
        d = _check_throughput(1000)
        assert d.score == 100


class TestCheckModel:
    def test_loaded_with_type(self):
        d = _check_model(True, "soul-7b")
        assert d.severity == Severity.OK
        assert "soul-7b" in d.message

    def test_loaded_without_type(self):
        d = _check_model(True, "")
        assert d.severity == Severity.OK

    def test_not_loaded(self):
        d = _check_model(False, "")
        assert d.severity == Severity.WARN
        assert d.score == 40

    def test_check_name(self):
        d = _check_model(True, "x")
        assert d.check == "model"

    def test_loaded_message_generic(self):
        d = _check_model(True, "")
        assert "Model loaded" in d.message

    def test_not_loaded_score(self):
        d = _check_model(False, "")
        assert d.score == 40

    def test_loaded_true_false_false(self):
        d = _check_model(False, "model_name")
        assert d.severity == Severity.WARN

    def test_loaded_false_true(self):
        d = _check_model(True, "x")
        assert d.score == 100


class TestCheckUptime:
    def test_hours(self):
        d = _check_uptime(7200)
        assert d.severity == Severity.OK
        assert "2.0h" in d.message

    def test_minutes(self):
        d = _check_uptime(300)
        assert d.severity == Severity.OK

    def test_warming_up(self):
        d = _check_uptime(30)
        assert d.severity == Severity.INFO

    def test_just_booted(self):
        d = _check_uptime(5)
        assert d.severity == Severity.INFO
        assert d.score == 30

    def test_check_name(self):
        d = _check_uptime(100)
        assert d.check == "uptime"

    def test_boundary_60s(self):
        d = _check_uptime(60)
        assert d.severity == Severity.INFO

    def test_boundary_3600s(self):
        d = _check_uptime(3600)
        assert d.severity == Severity.OK

    def test_very_long_uptime(self):
        d = _check_uptime(86400)
        assert "24.0h" in d.message

    def test_zero_uptime(self):
        d = _check_uptime(0)
        assert d.score == 30

    def test_11_seconds(self):
        d = _check_uptime(11)
        assert d.severity == Severity.INFO


class TestCheckResources:
    def test_no_data(self):
        d = _check_resources(0, 0)
        assert d.severity == Severity.INFO
        assert d.score == 80

    def test_healthy(self):
        d = _check_resources(30, 50)
        assert d.severity == Severity.OK

    def test_high_cpu(self):
        d = _check_resources(95, 50)
        assert "CPU" in d.message

    def test_high_memory(self):
        d = _check_resources(30, 95)
        assert "Memory" in d.message

    def test_both_high(self):
        d = _check_resources(90, 95)
        assert d.severity == Severity.CRITICAL

    def test_score_range(self):
        d = _check_resources(75, 85)
        assert 0 <= d.score <= 100

    def test_check_name(self):
        d = _check_resources(30, 50)
        assert d.check == "resources"

    def test_moderate_cpu(self):
        d = _check_resources(60, 50)
        assert "CPU" in d.message

    def test_moderate_memory(self):
        d = _check_resources(30, 80)
        assert "Memory" in d.message

    def test_extreme_values(self):
        d = _check_resources(100, 100)
        assert d.severity == Severity.CRITICAL
        assert d.score == 0.0

    def test_low_cpu_high_memory(self):
        d = _check_resources(20, 95)
        assert "Memory" in d.message

    def test_high_cpu_low_memory(self):
        d = _check_resources(95, 30)
        assert "CPU" in d.message

    def test_both_healthy(self):
        d = _check_resources(20, 30)
        assert "headroom" in d.message

    def test_cpu_above_85_throttling(self):
        d = _check_resources(86, 50)
        assert "throttling" in d.message

    def test_memory_above_90_near_limit(self):
        d = _check_resources(30, 91)
        assert "near limit" in d.message


class TestRunHealthFlow:
    def test_healthy_system(self):
        r = run_health_flow(
            req_count=1000, err_count=5,
            avg_latency_ms=200, tokens_per_sec=40,
            uptime_seconds=7200,
            model_loaded=True, model_type="soul",
            cpu_percent=30, memory_percent=50,
        )
        assert r.status == "healthy"
        assert r.score >= 80
        assert len(r.diagnoses) == 6
        assert r.model_loaded is True
        assert r.model_type == "soul"

    def test_degraded_system(self):
        r = run_health_flow(
            req_count=1000, err_count=200,
            avg_latency_ms=1500, tokens_per_sec=5,
            uptime_seconds=30,
            model_loaded=False,
            cpu_percent=90, memory_percent=95,
        )
        assert r.status in ("degraded", "unhealthy")
        assert len(r.diagnoses) == 6

    def test_unhealthy_system(self):
        r = run_health_flow(
            req_count=100, err_count=50,
            avg_latency_ms=3000, tokens_per_sec=1,
            uptime_seconds=5,
            model_loaded=False,
            cpu_percent=99, memory_percent=99,
        )
        assert r.status == "unhealthy"
        assert r.score < 50

    def test_model_type_in_summary(self):
        r = run_health_flow(
            req_count=1000, err_count=0,
            avg_latency_ms=100, tokens_per_sec=50,
            uptime_seconds=3600,
            model_loaded=True, model_type="llama",
        )
        assert "llama" in r.summary

    def test_summary_picks_worst(self):
        r = run_health_flow(
            req_count=1000, err_count=500,
            avg_latency_ms=200, tokens_per_sec=50,
            uptime_seconds=7200,
            model_loaded=True, model_type="x",
            cpu_percent=20, memory_percent=30,
        )
        assert "errors" in r.summary.lower() or "failing" in r.summary.lower()

    def test_score_is_integer(self):
        r = run_health_flow(
            req_count=100, err_count=5,
            avg_latency_ms=300, tokens_per_sec=20,
            uptime_seconds=600,
            model_loaded=True, model_type="",
        )
        assert isinstance(r.score, int)

    def test_all_diagnoses_present(self):
        r = run_health_flow(
            req_count=100, err_count=10,
            avg_latency_ms=500, tokens_per_sec=15,
            uptime_seconds=100,
            model_loaded=True, model_type="m",
            cpu_percent=60, memory_percent=70,
        )
        checks = {d.check for d in r.diagnoses}
        assert checks == {"errors", "latency", "throughput", "model", "uptime", "resources"}

    def test_score_range(self):
        r = run_health_flow(
            req_count=100, err_count=10,
            avg_latency_ms=500, tokens_per_sec=20,
            uptime_seconds=600,
            model_loaded=True, model_type="",
            cpu_percent=50, memory_percent=60,
        )
        assert 0 <= r.score <= 100

    def test_defaults_no_resources(self):
        r = run_health_flow(
            req_count=100, err_count=0,
            avg_latency_ms=100, tokens_per_sec=30,
            uptime_seconds=3600,
            model_loaded=True, model_type="x",
        )
        assert r.status == "healthy"

    def test_worst_severity_summary(self):
        r = run_health_flow(
            req_count=100, err_count=0,
            avg_latency_ms=3000, tokens_per_sec=1,
            uptime_seconds=5,
            model_loaded=False,
            cpu_percent=99, memory_percent=99,
        )
        assert r.status == "unhealthy"

    def test_diagnoses_are_diagnosis_type(self):
        r = run_health_flow(
            req_count=50, err_count=1,
            avg_latency_ms=200, tokens_per_sec=25,
            uptime_seconds=120,
            model_loaded=True, model_type="",
        )
        for d in r.diagnoses:
            assert isinstance(d, Diagnosis)

    def test_perfect_system(self):
        r = run_health_flow(
            req_count=10000, err_count=0,
            avg_latency_ms=50, tokens_per_sec=100,
            uptime_seconds=86400,
            model_loaded=True, model_type="soul-xl",
            cpu_percent=10, memory_percent=20,
        )
        assert r.status == "healthy"
        assert r.score >= 90

    def test_all_zeros(self):
        r = run_health_flow(
            req_count=0, err_count=0,
            avg_latency_ms=0, tokens_per_sec=0,
            uptime_seconds=0,
            model_loaded=False,
        )
        assert r.score >= 0


class TestEventPriority:
    def test_all_members(self):
        assert len(EventPriority) == 4

    def test_values(self):
        assert EventPriority.MONITOR.value == 0
        assert EventPriority.NORMAL.value == 1
        assert EventPriority.HIGH.value == 2
        assert EventPriority.CRITICAL.value == 3

    def test_ordering(self):
        assert EventPriority.MONITOR < EventPriority.NORMAL < EventPriority.HIGH < EventPriority.CRITICAL

    def test_is_int_enum(self):
        assert isinstance(EventPriority.NORMAL, int)

    def test_priority_as_int(self):
        assert int(EventPriority.MONITOR) == 0

    def test_priority_in_sort(self):
        priorities = [EventPriority.NORMAL, EventPriority.CRITICAL, EventPriority.MONITOR]
        sorted_p = sorted(priorities)
        assert sorted_p == [EventPriority.MONITOR, EventPriority.NORMAL, EventPriority.CRITICAL]

    def test_unique_values(self):
        values = [p.value for p in EventPriority]
        assert len(values) == len(set(values))

    def test_name_attribute(self):
        assert EventPriority.HIGH.name == "HIGH"


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

    def test_unique_ids(self):
        e1 = Event(name="x")
        e2 = Event(name="x")
        assert e1.id != e2.id

    def test_timestamp_is_recent(self):
        before = time.time()
        e = Event(name="t")
        after = time.time()
        assert before <= e.timestamp <= after

    def test_source_default(self):
        e = Event(name="x")
        assert e.source == ""

    def test_data_default(self):
        e = Event(name="x")
        assert e.data == {}

    def test_complex_data(self):
        e = Event(name="x", data={"nested": {"a": [1, 2, 3]}})
        assert e.data["nested"]["a"] == [1, 2, 3]

    def test_id_format(self):
        e = Event(name="x")
        assert e.id.startswith("evt_")
        assert len(e.id) == 16

    def test_empty_name(self):
        e = Event(name="")
        assert e.name == ""


class TestSubscription:
    def test_defaults(self):
        s = Subscription(handler=lambda: None)
        assert s.priority == EventPriority.NORMAL
        assert s.once is False

    def test_custom(self):
        s = Subscription(handler=lambda: None, priority=EventPriority.HIGH, once=True)
        assert s.priority == EventPriority.HIGH
        assert s.once is True

    def test_handler_stored(self):
        def my_handler():
            pass
        s = Subscription(handler=my_handler)
        assert s.handler is my_handler

    def test_all_priority_levels(self):
        for p in EventPriority:
            s = Subscription(handler=lambda: None, priority=p)
            assert s.priority == p

    def test_once_false_default(self):
        s = Subscription(handler=lambda n, d: None)
        assert s.once is False


class TestEventBus:
    def test_on_and_emit_sync(self):
        bus = EventBus()
        results = []
        bus.on("test", lambda name, data: results.append(data))
        bus.emit_sync("test", {"v": 1})
        assert results == [{"v": 1}]

    def test_multiple_handlers(self):
        bus = EventBus()
        count = [0]
        bus.on("x", lambda n, d: count.__setitem__(0, count[0] + 1))
        bus.on("x", lambda n, d: count.__setitem__(0, count[0] + 1))
        bus.emit_sync("x")
        assert count[0] == 2

    def test_once_handler(self):
        bus = EventBus()
        count = [0]
        bus.once("x", lambda n, d: count.__setitem__(0, count[0] + 1))
        bus.emit_sync("x")
        bus.emit_sync("x")
        assert count[0] == 1

    def test_off(self):
        bus = EventBus()
        handler = lambda n, d: None
        bus.on("x", handler)
        assert bus.off("x", handler) is True
        assert bus.off("x", handler) is False

    def test_clear_specific_event(self):
        bus = EventBus()
        bus.on("a", lambda n, d: None)
        bus.on("b", lambda n, d: None)
        bus.clear("a")
        assert bus.subscriber_count == 1

    def test_clear_all(self):
        bus = EventBus()
        bus.on("a", lambda n, d: None)
        bus.on("b", lambda n, d: None)
        bus.clear()
        assert bus.subscriber_count == 0

    def test_priority_ordering(self):
        bus = EventBus()
        order = []
        bus.on("x", lambda n, d: order.append("normal"), priority=EventPriority.NORMAL)
        bus.on("x", lambda n, d: order.append("high"), priority=EventPriority.HIGH)
        bus.emit_sync("x")
        assert order[0] == "high"

    def test_wildcard(self):
        bus = EventBus()
        results = []
        bus.on("*", lambda n, d: results.append(n))
        bus.emit_sync("foo")
        bus.emit_sync("bar")
        assert results == ["foo", "bar"]

    def test_handler_exception_isolated(self):
        bus = EventBus()
        bus.on("x", lambda n, d: 1 / 0)
        results = []
        bus.on("x", lambda n, d: results.append("ok"))
        bus.emit_sync("x")
        assert results == ["ok"]

    def test_history(self):
        bus = EventBus()
        bus.emit_sync("a", {"v": 1})
        bus.emit_sync("b", {"v": 2})
        h = bus.history()
        assert len(h) == 2

    def test_history_filtered(self):
        bus = EventBus()
        bus.emit_sync("a")
        bus.emit_sync("b")
        bus.emit_sync("a")
        h = bus.history("a")
        assert len(h) == 2

    def test_replay(self):
        bus = EventBus()
        bus.emit_sync("a", {"v": 1})
        results = []
        bus.replay("a", lambda n, d: results.append(d))
        assert results == [{"v": 1}]

    def test_subscriber_count(self):
        bus = EventBus()
        assert bus.subscriber_count == 0
        bus.on("a", lambda n, d: None)
        bus.on("b", lambda n, d: None)
        assert bus.subscriber_count == 2

    def test_history_max_limit(self):
        bus = EventBus(max_history=3)
        for i in range(5):
            bus.emit_sync("x")
        assert len(bus.history("x")) == 3

    def test_non_callable_handler_raises(self):
        bus = EventBus()
        try:
            bus.on("x", "not_callable")
            assert False, "Should have raised TypeError"
        except TypeError:
            pass

    def test_emit_returns_count(self):
        bus = EventBus()
        bus.on("x", lambda n, d: None)
        bus.on("x", lambda n, d: None)
        count = bus.emit_sync("x")
        assert count == 2

    def test_once_removes_after_emit(self):
        bus = EventBus()
        bus.once("x", lambda n, d: None)
        assert bus.subscriber_count == 1
        bus.emit_sync("x")
        assert bus.subscriber_count == 0

    def test_off_nonexistent_returns_false(self):
        bus = EventBus()
        assert bus.off("x", lambda n, d: None) is False

    def test_clear_nonexistent_event(self):
        bus = EventBus()
        bus.clear("nonexistent")
        assert bus.subscriber_count == 0

    def test_emit_no_handlers(self):
        bus = EventBus()
        count = bus.emit_sync("empty")
        assert count == 0

    def test_history_empty(self):
        bus = EventBus()
        assert bus.history() == []

    def test_replay_no_handler(self):
        bus = EventBus()
        bus.emit_sync("a", {"v": 1})
        events = bus.replay("a")
        assert len(events) == 1

    def test_multiple_wildcards(self):
        bus = EventBus()
        results = []
        bus.on("*", lambda n, d: results.append("first"))
        bus.on("*", lambda n, d: results.append("second"))
        bus.emit_sync("test")
        assert len(results) == 2

    def test_emit_with_source(self):
        bus = EventBus()
        results = []
        bus.on("x", lambda n, d: results.append(n))
        bus.emit_sync("x", source="test_src")
        assert results == ["x"]

    def test_clear_wildcard(self):
        bus = EventBus()
        bus.on("*", lambda n, d: None)
        bus.clear("*")
        assert bus.subscriber_count == 0

    def test_off_wildcard(self):
        bus = EventBus()
        handler = lambda n, d: None
        bus.on("*", handler)
        assert bus.off("*", handler) is True

    def test_replay_all_events(self):
        bus = EventBus()
        bus.emit_sync("a", {"x": 1})
        bus.emit_sync("b", {"x": 2})
        events = bus.replay()
        assert len(events) == 2

    def test_once_on_wildcard(self):
        bus = EventBus()
        count = [0]
        bus.once("*", lambda n, d: count.__setitem__(0, count[0] + 1))
        bus.emit_sync("a")
        bus.emit_sync("b")
        assert count[0] == 1

    def test_emit_empty_data(self):
        bus = EventBus()
        results = []
        bus.on("x", lambda n, d: results.append(d))
        bus.emit_sync("x")
        assert results == [{}]

    def test_many_handlers(self):
        bus = EventBus()
        count = [0]
        for _ in range(10):
            bus.on("x", lambda n, d: count.__setitem__(0, count[0] + 1))
        bus.emit_sync("x")
        assert count[0] == 10


class TestEventBusSingleton:
    def test_get_returns_same(self):
        reset_event_bus()
        b1 = get_event_bus()
        b2 = get_event_bus()
        assert b1 is b2

    def test_set_and_get(self):
        custom = EventBus()
        set_event_bus(custom)
        assert get_event_bus() is custom
        reset_event_bus()

    def test_reset(self):
        reset_event_bus()
        b1 = get_event_bus()
        reset_event_bus()
        b2 = get_event_bus()
        assert b1 is not b2

    def test_set_none_creates_new(self):
        reset_event_bus()
        custom = EventBus()
        set_event_bus(custom)
        assert get_event_bus() is custom
        reset_event_bus()
        assert get_event_bus() is not custom
