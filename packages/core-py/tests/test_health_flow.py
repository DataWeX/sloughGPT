"""Tests for domains/infrastructure/health_flow.py — diagnostic pipeline."""

import pytest
from domains.infrastructure.health_flow import (
    Severity,
    Diagnosis,
    HealthFlowResult,
    _check_errors,
    _check_latency,
    _check_throughput,
    _check_model,
    _check_uptime,
    run_health_flow,
)


class TestCheckErrors:
    def test_no_requests(self):
        d = _check_errors(0, 0)
        assert d.severity == Severity.OK
        assert d.score == 100

    def test_zero_errors(self):
        d = _check_errors(100, 0)
        assert d.severity == Severity.OK
        assert d.score == 100
        assert "All 100" in d.message

    def test_low_error_rate(self):
        d = _check_errors(100, 1)
        assert d.severity == Severity.OK
        assert d.score == 80.0

    def test_moderate_error_rate(self):
        d = _check_errors(100, 3)
        assert d.severity == Severity.CRITICAL
        assert d.score < 50

    def test_high_error_rate(self):
        d = _check_errors(100, 10)
        assert d.severity == Severity.CRITICAL
        assert "failing" in d.message

    def test_rare_error_rate(self):
        d = _check_errors(1000, 5)
        assert d.severity == Severity.OK
        assert "rare" in d.message


class TestCheckLatency:
    def test_no_data(self):
        d = _check_latency(0)
        assert d.severity == Severity.OK
        assert d.score == 100

    def test_fast(self):
        d = _check_latency(100)
        assert d.severity == Severity.OK
        assert "Snappy" in d.message

    def test_moderate(self):
        d = _check_latency(500)
        assert d.severity == Severity.OK

    def test_slow(self):
        d = _check_latency(1200)
        assert d.severity == Severity.CRITICAL

    def test_very_slow(self):
        d = _check_latency(3000)
        assert d.severity == Severity.CRITICAL


class TestCheckThroughput:
    def test_no_data(self):
        d = _check_throughput(0)
        assert d.severity == Severity.INFO

    def test_fast(self):
        d = _check_throughput(40)
        assert d.severity == Severity.WARN
        assert "Fast" in d.message

    def test_ok(self):
        d = _check_throughput(15)
        assert d.severity == Severity.CRITICAL

    def test_slow(self):
        d = _check_throughput(7)
        assert d.severity == Severity.CRITICAL

    def test_very_slow(self):
        d = _check_throughput(2)
        assert d.severity == Severity.CRITICAL


class TestRunHealthFlow:
    def test_healthy_system(self):
        result = run_health_flow(
            req_count=100,
            err_count=0,
            avg_latency_ms=200,
            tokens_per_sec=30,
            uptime_seconds=3600,
            model_loaded=True,
            model_type="gpt2",
        )
        assert isinstance(result, HealthFlowResult)
        assert result.score >= 80
        assert result.status == "healthy"
        assert result.model_loaded is True
        assert len(result.diagnoses) > 0

    def test_unhealthy_system(self):
        result = run_health_flow(
            req_count=100,
            err_count=50,
            avg_latency_ms=3000,
            tokens_per_sec=1,
            uptime_seconds=60,
            model_loaded=False,
            model_type="",
        )
        assert result.score < 60
        assert result.status in ("unhealthy", "degraded")

    def test_no_model(self):
        result = run_health_flow(
            req_count=10,
            err_count=0,
            avg_latency_ms=100,
            tokens_per_sec=0,
            uptime_seconds=10,
            model_loaded=False,
            model_type="",
        )
        assert result.model_loaded is False

    def test_diagnoses_have_required_fields(self):
        result = run_health_flow(
            req_count=50,
            err_count=2,
            avg_latency_ms=400,
            tokens_per_sec=20,
            uptime_seconds=600,
            model_loaded=True,
            model_type="gpt2",
        )
        for d in result.diagnoses:
            assert hasattr(d, "check")
            assert hasattr(d, "severity")
            assert hasattr(d, "score")
            assert hasattr(d, "message")
            assert isinstance(d.severity, Severity)

    def test_summary_is_string(self):
        result = run_health_flow(
            req_count=0, err_count=0, avg_latency_ms=0,
            tokens_per_sec=0, uptime_seconds=0, model_loaded=False,
        )
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0


class TestCheckModel:
    def test_loaded_with_type(self):
        d = _check_model(True, "gpt2")
        assert d.severity == Severity.OK
        assert d.score == 100
        assert "gpt2 loaded" in d.message

    def test_loaded_without_type(self):
        d = _check_model(True, "")
        assert d.severity == Severity.OK
        assert d.message == "Model loaded."

    def test_not_loaded(self):
        d = _check_model(False, "")
        assert d.severity == Severity.WARN
        assert d.score == 40
        assert "No model loaded" in d.message


class TestCheckUptime:
    def test_hours(self):
        d = _check_uptime(7200)
        assert d.severity == Severity.OK
        assert "Up 2.0h" in d.message

    def test_minutes(self):
        d = _check_uptime(120)
        assert "Up 2m" in d.message

    def test_warming(self):
        d = _check_uptime(30)
        assert d.message == "Warming up."

    def test_just_booted(self):
        d = _check_uptime(5)
        assert d.message == "Just booted."
