"""
Tests for model_server.py — CircuitBreaker, ModelMetrics, IdleManager, Priority.

Pure state-machine tests (no model/server dependencies).

Covers:
    - CircuitBreaker: state transitions (CLOSED→OPEN→HALF_OPEN→CLOSED),
      failure threshold, recovery timeout, allow_request, callbacks
    - ModelMetrics: record_success/failure/timeout, computed properties, snapshot
    - IdleManager: register/touch/unregister, idle detection, reload trigger
    - Priority: ordering
    - QueueMetrics: data defaults
"""

import time
import sys
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock
import pytest

_CORE_PY = Path(__file__).resolve().parents[1]
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

from domains.infrastructure.model_server import (
    CircuitBreaker,
    CircuitBreakerState,
    ModelMetrics,
    ModelStatus,
    IdleManager,
    Priority,
    QueueMetrics,
)


# ── CircuitBreaker ────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.allow_request() is True

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED  # 2 < 3
        assert cb.allow_request() is True
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN  # 3 >= 3
        assert cb.allow_request() is False

    def test_success_resets_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED  # only 2 since reset

    def test_success_resets_to_closed(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.allow_request() is True

    def test_failure_in_half_open_goes_back_to_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.allow_request() is False

    def test_success_in_half_open_closes(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_state_change_callback(self):
        cb = CircuitBreaker(failure_threshold=1)
        changes = []
        cb._on_state_change = lambda old, new: changes.append((old, new))
        cb.record_failure()
        assert changes == [(CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN)]

    def test_no_callback_on_same_state(self):
        cb = CircuitBreaker(failure_threshold=3)
        changes = []
        cb._on_state_change = lambda old, new: changes.append((old, new))
        cb.record_failure()  # stays CLOSED
        assert changes == []

    def test_threshold_1(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN


# ── ModelMetrics ──────────────────────────────────────────────────────


class TestModelMetrics:
    def test_defaults(self):
        m = ModelMetrics()
        assert m.requests_total == 0
        assert m.requests_completed == 0
        assert m.requests_failed == 0
        assert m.avg_generation_time_ms == 0.0
        assert m.error_rate == 0.0

    def test_record_success(self):
        m = ModelMetrics()
        m.record_success(100.0, 50)
        assert m.requests_completed == 1
        assert m.total_generation_time_ms == 100.0
        assert m.max_generation_time_ms == 100.0
        assert m.min_generation_time_ms == 100.0
        assert m.tokens_generated_total == 50
        assert m.consecutive_failures == 0

    def test_record_failure(self):
        m = ModelMetrics()
        m.record_failure("boom")
        assert m.requests_failed == 1
        assert m.last_error == "boom"
        assert m.consecutive_failures == 1

    def test_record_timeout(self):
        m = ModelMetrics()
        m.record_timeout()
        assert m.requests_timed_out == 1
        assert m.consecutive_failures == 1

    def test_consecutive_failures_reset_on_success(self):
        m = ModelMetrics()
        m.record_failure("e1")
        m.record_failure("e2")
        assert m.consecutive_failures == 2
        m.record_success(10.0, 1)
        assert m.consecutive_failures == 0

    def test_avg_generation_time(self):
        m = ModelMetrics()
        m.record_success(100.0, 10)
        m.record_success(200.0, 20)
        assert m.avg_generation_time_ms == 150.0

    def test_error_rate(self):
        m = ModelMetrics()
        m.requests_total = 10
        m.requests_failed = 3
        assert m.error_rate == 0.3

    def test_min_max_tracking(self):
        m = ModelMetrics()
        m.record_success(50.0, 1)
        m.record_success(200.0, 1)
        m.record_success(100.0, 1)
        assert m.min_generation_time_ms == 50.0
        assert m.max_generation_time_ms == 200.0

    def test_snapshot(self):
        m = ModelMetrics()
        m.record_success(100.0, 50)
        snap = m.snapshot()
        assert snap["requests_completed"] == 1
        assert snap["tokens_generated_total"] == 50
        assert snap["avg_generation_time_ms"] == 100.0
        assert snap["max_generation_time_ms"] == 100.0

    def test_snapshot_empty_min_is_zero(self):
        m = ModelMetrics()
        snap = m.snapshot()
        assert snap["min_generation_time_ms"] == 0.0


# ── IdleManager ───────────────────────────────────────────────────────


class TestIdleManager:
    def test_register_and_touch(self):
        mgr = IdleManager(idle_timeout_s=300, check_interval_s=1)
        mgr.register("m1")
        assert mgr.touch("m1") is False
        mgr.shutdown()

    def test_unregister(self):
        mgr = IdleManager()
        mgr.register("m1")
        mgr.unregister("m1")
        assert mgr.get_idle_info("m1") is None
        mgr.shutdown()

    def test_get_idle_info(self):
        mgr = IdleManager(idle_timeout_s=300)
        mgr.register("m1")
        info = mgr.get_idle_info("m1")
        assert info is not None
        assert info["idle_timeout_s"] == 300
        assert info["unloaded"] is False
        assert info["remaining_s"] > 0
        mgr.shutdown()

    def test_get_idle_info_unknown(self):
        mgr = IdleManager()
        assert mgr.get_idle_info("unknown") is None
        mgr.shutdown()

    def test_is_idle_unloaded(self):
        mgr = IdleManager()
        mgr.register("m1")
        assert mgr.is_idle_unloaded("m1") is False
        mgr.shutdown()

    def test_touch_triggers_reload(self):
        mgr = IdleManager(idle_timeout_s=0.01, check_interval_s=0.01)
        reload_called = []
        mgr.register("m1", unload_fn=lambda: None, reload_fn=lambda: reload_called.append(1))
        # Force idle unload by simulating expired touch
        with mgr._lock:
            mgr._models["m1"]["last_touch"] = time.time() - 10
            mgr._models["m1"]["unloaded_at"] = time.time()
        mgr.touch("m1")
        assert reload_called == [1]
        mgr.shutdown()

    def test_touch_unknown_returns_false(self):
        mgr = IdleManager()
        assert mgr.touch("unknown") is False
        mgr.shutdown()

    def test_shutdown_stops_thread(self):
        mgr = IdleManager(check_interval_s=0.01)
        mgr.register("m1")
        time.sleep(0.05)
        mgr.shutdown()
        assert not mgr._running


# ── Priority ──────────────────────────────────────────────────────────


class TestPriority:
    def test_ordering(self):
        assert Priority.HIGH < Priority.MEDIUM < Priority.LOW

    def test_values(self):
        assert Priority.HIGH == 0
        assert Priority.MEDIUM == 1
        assert Priority.LOW == 2


# ── QueueMetrics ──────────────────────────────────────────────────────


class TestQueueMetrics:
    def test_defaults(self):
        q = QueueMetrics()
        assert q.depth_high == 0
        assert q.total_depth == 0
        assert q.avg_wait_ms == 0.0


# ── ModelStatus ───────────────────────────────────────────────────────


class TestModelStatus:
    def test_all_values(self):
        statuses = [s.value for s in ModelStatus]
        assert "uninitialized" in statuses
        assert "loading" in statuses
        assert "ready" in statuses
        assert "degraded" in statuses
        assert "error" in statuses
        assert "unloaded" in statuses
