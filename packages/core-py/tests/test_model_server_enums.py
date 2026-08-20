"""Tests for domains.infrastructure.model_server — Priority, QueueMetrics, ModelStatus, ModelMetrics, CircuitBreakerState, CircuitBreaker."""

import time
from domains.infrastructure.model_server import (
    Priority, QueueMetrics, ModelStatus, ModelMetrics, CircuitBreakerState, CircuitBreaker,
)


class TestPriority:
    def test_all_members(self):
        assert len(Priority) == 3
    def test_values(self):
        assert Priority.HIGH == 0
        assert Priority.MEDIUM == 1
        assert Priority.LOW == 2


class TestQueueMetrics:
    def test_defaults(self):
        qm = QueueMetrics()
        assert qm.depth_high == 0
        assert qm.total_depth == 0


class TestModelStatus:
    def test_all_members(self):
        assert len(ModelStatus) == 6
    def test_values(self):
        assert ModelStatus.READY.value == "ready"
        assert ModelStatus.ERROR.value == "error"


class TestModelMetrics:
    def test_defaults(self):
        mm = ModelMetrics()
        assert mm.requests_total == 0
        assert mm.requests_completed == 0

    def test_record_success(self):
        mm = ModelMetrics()
        mm.record_success(100.0, 50)
        assert mm.requests_completed == 1
        assert mm.max_generation_time_ms == 100.0

    def test_record_failure(self):
        mm = ModelMetrics()
        mm.record_failure("timeout")
        assert mm.requests_failed == 1
        assert mm.last_error == "timeout"


class TestCircuitBreakerState:
    def test_all_members(self):
        assert len(CircuitBreakerState) == 3
    def test_values(self):
        assert CircuitBreakerState.CLOSED.value == "closed"
        assert CircuitBreakerState.OPEN.value == "open"
        assert CircuitBreakerState.HALF_OPEN.value == "half_open"


class TestCircuitBreaker:
    def test_init(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_record_failure_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_record_failure_trips_open(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=100)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
