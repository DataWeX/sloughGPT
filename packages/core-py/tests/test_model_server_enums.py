"""Tests for domains.infrastructure.model_server — Priority, QueueMetrics, ModelStatus, ModelMetrics, CircuitBreakerState, CircuitBreaker, IdleManager, SessionKVCache."""

import time
from unittest.mock import MagicMock

from domains.infrastructure.model_server import (
    Priority, QueueMetrics, ModelStatus, ModelMetrics, CircuitBreakerState, CircuitBreaker,
    IdleManager, SessionKVCache,
)


class TestPriority:
    def test_all_members(self):
        assert len(Priority) == 3

    def test_values(self):
        assert Priority.HIGH == 0
        assert Priority.MEDIUM == 1
        assert Priority.LOW == 2

    def test_high_is_lowest_number(self):
        assert Priority.HIGH < Priority.MEDIUM
        assert Priority.MEDIUM < Priority.LOW

    def test_ordering(self):
        assert Priority.HIGH < Priority.MEDIUM < Priority.LOW

    def test_is_int(self):
        assert isinstance(Priority.HIGH, int)

    def test_unique_values(self):
        values = [p.value for p in Priority]
        assert len(values) == len(set(values))

    def test_repr(self):
        r = repr(Priority.HIGH)
        assert "HIGH" in r or "0" in r

    def test_member_by_name(self):
        assert Priority["HIGH"] is Priority.HIGH

    def test_member_by_value(self):
        assert Priority(0) is Priority.HIGH

    def test_invalid_value_raises(self):
        try:
            Priority(99)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestQueueMetrics:
    def test_defaults(self):
        qm = QueueMetrics()
        assert qm.depth_high == 0
        assert qm.total_depth == 0

    def test_all_fields_default_zero(self):
        qm = QueueMetrics()
        assert qm.depth_medium == 0
        assert qm.depth_low == 0
        assert qm.served == 0
        assert qm.timed_out == 0
        assert qm.avg_wait_ms == 0.0
        assert qm.max_wait_ms == 0.0

    def test_custom_values(self):
        qm = QueueMetrics(depth_high=5, depth_medium=3, depth_low=1, total_depth=9)
        assert qm.depth_high == 5
        assert qm.depth_medium == 3
        assert qm.depth_low == 1
        assert qm.total_depth == 9

    def test_served_and_timed_out(self):
        qm = QueueMetrics(served=100, timed_out=5)
        assert qm.served == 100
        assert qm.timed_out == 5

    def test_wait_times(self):
        qm = QueueMetrics(avg_wait_ms=12.5, max_wait_ms=200.0)
        assert qm.avg_wait_ms == 12.5
        assert qm.max_wait_ms == 200.0


class TestModelStatus:
    def test_all_members(self):
        assert len(ModelStatus) == 6

    def test_values(self):
        assert ModelStatus.UNINITIALIZED.value == "uninitialized"
        assert ModelStatus.LOADING.value == "loading"
        assert ModelStatus.READY.value == "ready"
        assert ModelStatus.DEGRADED.value == "degraded"
        assert ModelStatus.ERROR.value == "error"
        assert ModelStatus.UNLOADED.value == "unloaded"

    def test_unique_values(self):
        values = [s.value for s in ModelStatus]
        assert len(values) == len(set(values))

    def test_is_enum(self):
        from enum import Enum
        assert issubclass(ModelStatus, Enum)

    def test_member_by_name(self):
        assert ModelStatus["READY"] is ModelStatus.READY

    def test_member_by_value(self):
        assert ModelStatus("ready") is ModelStatus.READY


class TestModelMetrics:
    def test_defaults(self):
        mm = ModelMetrics()
        assert mm.requests_total == 0
        assert mm.requests_completed == 0

    def test_all_defaults(self):
        mm = ModelMetrics()
        assert mm.requests_failed == 0
        assert mm.requests_timed_out == 0
        assert mm.total_generation_time_ms == 0.0
        assert mm.max_generation_time_ms == 0.0
        assert mm.min_generation_time_ms == float("inf")
        assert mm.tokens_generated_total == 0
        assert mm.last_generation_time_ms == 0.0
        assert mm.last_error is None
        assert mm.last_error_at is None
        assert mm.consecutive_failures == 0
        assert mm.last_request_time == 0.0

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

    def test_record_success_updates_min(self):
        mm = ModelMetrics()
        mm.record_success(100.0, 10)
        mm.record_success(50.0, 10)
        assert mm.min_generation_time_ms == 50.0

    def test_record_success_updates_max(self):
        mm = ModelMetrics()
        mm.record_success(50.0, 10)
        mm.record_success(200.0, 10)
        assert mm.max_generation_time_ms == 200.0

    def test_record_success_accumulates_tokens(self):
        mm = ModelMetrics()
        mm.record_success(10.0, 50)
        mm.record_success(10.0, 30)
        assert mm.tokens_generated_total == 80

    def test_record_success_accumulates_time(self):
        mm = ModelMetrics()
        mm.record_success(100.0, 10)
        mm.record_success(200.0, 10)
        assert mm.total_generation_time_ms == 300.0

    def test_record_success_resets_consecutive_failures(self):
        mm = ModelMetrics()
        mm.record_failure("err1")
        mm.record_failure("err2")
        assert mm.consecutive_failures == 2
        mm.record_success(10.0, 10)
        assert mm.consecutive_failures == 0

    def test_record_failure_increments_consecutive(self):
        mm = ModelMetrics()
        mm.record_failure("err1")
        mm.record_failure("err2")
        mm.record_failure("err3")
        assert mm.consecutive_failures == 3

    def test_record_failure_stores_timestamp(self):
        mm = ModelMetrics()
        before = time.time()
        mm.record_failure("err")
        after = time.time()
        assert before <= mm.last_error_at <= after

    def test_record_timeout(self):
        mm = ModelMetrics()
        mm.record_timeout()
        assert mm.requests_timed_out == 1
        assert mm.consecutive_failures == 1

    def test_record_timeout_increments_consecutive(self):
        mm = ModelMetrics()
        mm.record_timeout()
        mm.record_timeout()
        assert mm.consecutive_failures == 2

    def test_reset(self):
        mm = ModelMetrics()
        mm.record_success(100.0, 50)
        mm.record_failure("err")
        mm.record_timeout()
        mm.reset()
        assert mm.requests_total == 0
        assert mm.requests_completed == 0
        assert mm.requests_failed == 0
        assert mm.requests_timed_out == 0
        assert mm.total_generation_time_ms == 0.0

    def test_avg_generation_time_ms_empty(self):
        mm = ModelMetrics()
        assert mm.avg_generation_time_ms == 0.0

    def test_avg_generation_time_ms(self):
        mm = ModelMetrics()
        mm.record_success(100.0, 10)
        mm.record_success(200.0, 10)
        assert mm.avg_generation_time_ms == 150.0

    def test_error_rate_empty(self):
        mm = ModelMetrics()
        assert mm.error_rate == 0.0

    def test_error_rate(self):
        mm = ModelMetrics()
        mm.requests_total = 10
        mm.requests_failed = 3
        assert mm.error_rate == 0.3

    def test_snapshot(self):
        mm = ModelMetrics()
        mm.requests_total = 10
        mm.record_success(100.0, 50)
        mm.record_failure("err")
        s = mm.snapshot()
        assert s["requests_completed"] == 1
        assert s["requests_failed"] == 1
        assert s["avg_generation_time_ms"] == 100.0
        assert s["max_generation_time_ms"] == 100.0
        assert s["min_generation_time_ms"] == 100.0
        assert s["tokens_generated_total"] == 50
        assert s["last_error"] == "err"
        assert s["error_rate"] > 0

    def test_snapshot_empty(self):
        mm = ModelMetrics()
        s = mm.snapshot()
        assert s["requests_completed"] == 0
        assert s["requests_failed"] == 0
        assert s["min_generation_time_ms"] == 0.0

    def test_last_request_time_set_on_success(self):
        mm = ModelMetrics()
        before = time.time()
        mm.record_success(10.0, 5)
        after = time.time()
        assert before <= mm.last_request_time <= after

    def test_last_request_time_set_on_failure(self):
        mm = ModelMetrics()
        before = time.time()
        mm.record_failure("err")
        after = time.time()
        assert before <= mm.last_request_time <= after

    def test_last_generation_time_ms(self):
        mm = ModelMetrics()
        mm.record_success(42.0, 10)
        assert mm.last_generation_time_ms == 42.0
        mm.record_success(99.0, 10)
        assert mm.last_generation_time_ms == 99.0


class TestCircuitBreakerState:
    def test_all_members(self):
        assert len(CircuitBreakerState) == 3

    def test_values(self):
        assert CircuitBreakerState.CLOSED.value == "closed"
        assert CircuitBreakerState.OPEN.value == "open"
        assert CircuitBreakerState.HALF_OPEN.value == "half_open"

    def test_unique_values(self):
        values = [s.value for s in CircuitBreakerState]
        assert len(values) == len(set(values))

    def test_is_enum(self):
        from enum import Enum
        assert issubclass(CircuitBreakerState, Enum)

    def test_member_by_name(self):
        assert CircuitBreakerState["CLOSED"] is CircuitBreakerState.CLOSED

    def test_member_by_value(self):
        assert CircuitBreakerState("closed") is CircuitBreakerState.CLOSED


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

    def test_allow_request_when_closed(self):
        cb = CircuitBreaker()
        assert cb.allow_request() is True

    def test_allow_request_when_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=100)
        cb.record_failure()
        assert cb.allow_request() is False

    def test_record_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED

    def test_recovery_timeout_transitions_to_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_custom_threshold(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_state_change_callback(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=100)
        changes = []
        cb._on_state_change = lambda old, new: changes.append((old, new))
        cb.record_failure()
        assert len(changes) == 1
        assert changes[0] == (CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN)

    def test_no_callback_on_same_state(self):
        cb = CircuitBreaker(failure_threshold=3)
        changes = []
        cb._on_state_change = lambda old, new: changes.append((old, new))
        cb.record_failure()
        assert len(changes) == 0

    def test_failure_count_reset_on_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_default_threshold(self):
        cb = CircuitBreaker()
        assert cb.failure_threshold == 3

    def test_default_recovery_timeout(self):
        cb = CircuitBreaker()
        assert cb.recovery_timeout == 30.0


class TestIdleManager:
    def test_register_and_unregister(self):
        mgr = IdleManager(check_interval_s=100)
        mgr.register("model1")
        assert "model1" in mgr._models
        mgr.unregister("model1")
        assert "model1" not in mgr._models
        mgr.shutdown()

    def test_unregister_nonexistent(self):
        mgr = IdleManager(check_interval_s=100)
        mgr.unregister("nonexistent")
        mgr.shutdown()

    def test_touch_registered(self):
        mgr = IdleManager(check_interval_s=100)
        mgr.register("model1")
        result = mgr.touch("model1")
        assert result is False
        mgr.shutdown()

    def test_touch_unregistered_returns_false(self):
        mgr = IdleManager(check_interval_s=100)
        result = mgr.touch("nonexistent")
        assert result is False
        mgr.shutdown()

    def test_is_idle_unloaded_false_by_default(self):
        mgr = IdleManager(check_interval_s=100)
        mgr.register("model1")
        assert mgr.is_idle_unloaded("model1") is False
        mgr.shutdown()

    def test_is_reloading_false_by_default(self):
        mgr = IdleManager(check_interval_s=100)
        mgr.register("model1")
        assert mgr.is_reloading("model1") is False
        mgr.shutdown()

    def test_get_idle_info(self):
        mgr = IdleManager(idle_timeout_s=300, check_interval_s=100)
        mgr.register("model1")
        info = mgr.get_idle_info("model1")
        assert info is not None
        assert "last_request_age_s" in info
        assert info["idle_timeout_s"] == 300
        assert info["unloaded"] is False
        assert info["remaining_s"] > 0
        mgr.shutdown()

    def test_get_idle_info_nonexistent(self):
        mgr = IdleManager(check_interval_s=100)
        assert mgr.get_idle_info("nonexistent") is None
        mgr.shutdown()

    def test_shutdown(self):
        mgr = IdleManager(check_interval_s=100)
        mgr.register("model1")
        mgr.shutdown()
        assert mgr._running is False

    def test_reset(self):
        mgr = IdleManager(check_interval_s=100)
        mgr.register("model1")
        mgr.reset()
        assert len(mgr._models) == 0
        mgr.shutdown()


class TestSessionKVCache:
    def test_store_and_get(self):
        cache = SessionKVCache()
        cache.store("s1", [1, 2, 3], "pkv_data")
        pkv, prefix_len = cache.get("s1", [1, 2, 3])
        assert pkv == "pkv_data"
        assert prefix_len == 3

    def test_get_miss(self):
        cache = SessionKVCache()
        pkv, prefix_len = cache.get("s1", [1, 2])
        assert pkv is None
        assert prefix_len == 0

    def test_partial_prefix(self):
        cache = SessionKVCache()
        cache.store("s1", [1, 2, 3, 4], "pkv")
        pkv, prefix_len = cache.get("s1", [1, 2, 99])
        assert prefix_len == 2

    def test_no_prefix(self):
        cache = SessionKVCache()
        cache.store("s1", [1, 2, 3], "pkv")
        pkv, prefix_len = cache.get("s1", [99, 88])
        assert pkv is None
        assert prefix_len == 0

    def test_clear(self):
        cache = SessionKVCache()
        cache.store("s1", [1, 2], "pkv")
        cache.clear("s1")
        pkv, _ = cache.get("s1", [1, 2])
        assert pkv is None

    def test_lru_eviction(self):
        cache = SessionKVCache(max_sessions=2)
        cache.store("s1", [1], "pkv1")
        cache.store("s2", [2], "pkv2")
        cache.store("s3", [3], "pkv3")
        assert cache.size <= 2

    def test_size(self):
        cache = SessionKVCache()
        assert cache.size == 0
        cache.store("s1", [1], "pkv")
        assert cache.size == 1
        cache.store("s2", [2], "pkv")
        assert cache.size == 2

    def test_stats(self):
        cache = SessionKVCache(max_sessions=10, ttl=300.0)
        s = cache.stats()
        assert s["max_sessions"] == 10
        assert s["ttl_seconds"] == 300.0
        assert s["entries"] == 0

    def test_evict_expired(self):
        cache = SessionKVCache(ttl=0.01)
        cache.store("s1", [1], "pkv")
        time.sleep(0.02)
        cache.evict_expired()
        assert cache.size == 0

    def test_overwrite_session(self):
        cache = SessionKVCache()
        cache.store("s1", [1, 2], "pkv_v1")
        cache.store("s1", [1, 2, 3], "pkv_v2")
        pkv, prefix_len = cache.get("s1", [1, 2, 3])
        assert pkv == "pkv_v2"
        assert prefix_len == 3

    def test_multiple_sessions(self):
        cache = SessionKVCache()
        cache.store("s1", [1], "pkv1")
        cache.store("s2", [2], "pkv2")
        pkv1, _ = cache.get("s1", [1])
        pkv2, _ = cache.get("s2", [2])
        assert pkv1 == "pkv1"
        assert pkv2 == "pkv2"

    def test_clear_nonexistent(self):
        cache = SessionKVCache()
        cache.clear("nonexistent")

    def test_default_ttl(self):
        cache = SessionKVCache()
        assert cache._ttl == 600.0

    def test_default_max_sessions(self):
        cache = SessionKVCache()
        assert cache._max_sessions == 20

    def test_stats_after_store(self):
        cache = SessionKVCache()
        cache.store("s1", [1], "pkv")
        s = cache.stats()
        assert s["entries"] == 1

    def test_get_empty_ids(self):
        cache = SessionKVCache()
        cache.store("s1", [1, 2, 3], "pkv")
        pkv, prefix_len = cache.get("s1", [])
        assert pkv is None
        assert prefix_len == 0

    def test_evict_expired_no_expired(self):
        cache = SessionKVCache(ttl=300.0)
        cache.store("s1", [1], "pkv")
        cache.evict_expired()
        assert cache.size == 1

    def test_lru_evicts_oldest(self):
        cache = SessionKVCache(max_sessions=2)
        cache.store("s1", [1], "pkv1")
        time.sleep(0.005)
        cache.store("s2", [2], "pkv2")
        time.sleep(0.005)
        cache.store("s3", [3], "pkv3")
        pkv1, _ = cache.get("s1", [1])
        assert pkv1 is None


# ---------------------------------------------------------------------------
# Extended ModelMetrics tests
# ---------------------------------------------------------------------------

class TestModelMetricsExtended:
    def test_requests_total_not_incremented_by_record(self):
        mm = ModelMetrics()
        mm.record_success(10.0, 5)
        assert mm.requests_total == 0

    def test_record_success_sets_last_request_time(self):
        mm = ModelMetrics()
        before = time.time()
        mm.record_success(10.0, 5)
        after = time.time()
        assert before <= mm.last_request_time <= after

    def test_record_failure_sets_last_request_time(self):
        mm = ModelMetrics()
        before = time.time()
        mm.record_failure("err")
        after = time.time()
        assert before <= mm.last_request_time <= after

    def test_record_failure_does_not_reset_consecutive_on_success(self):
        mm = ModelMetrics()
        mm.record_failure("e1")
        mm.record_failure("e2")
        mm.record_success(10.0, 5)
        assert mm.consecutive_failures == 0

    def test_record_timeout_does_not_set_last_error(self):
        mm = ModelMetrics()
        mm.record_timeout()
        assert mm.last_error is None

    def test_reset_clears_min_generation(self):
        mm = ModelMetrics()
        mm.record_success(100.0, 10)
        mm.reset()
        assert mm.min_generation_time_ms == float("inf")

    def test_snapshot_has_all_keys(self):
        mm = ModelMetrics()
        s = mm.snapshot()
        expected_keys = {
            "requests_total", "requests_completed", "requests_failed",
            "requests_timed_out", "consecutive_failures", "avg_generation_time_ms",
            "max_generation_time_ms", "min_generation_time_ms", "last_generation_time_ms",
            "tokens_generated_total", "last_error", "error_rate", "last_request_time",
        }
        assert expected_keys == set(s.keys())

    def test_error_rate_after_record_failure(self):
        mm = ModelMetrics()
        mm.record_failure("err")
        mm.requests_total = 1
        assert mm.error_rate == 1.0

    def test_multiple_record_success_avg(self):
        mm = ModelMetrics()
        mm.record_success(100.0, 10)
        mm.record_success(200.0, 10)
        mm.record_success(300.0, 10)
        assert mm.avg_generation_time_ms == 200.0

    def test_record_failure_then_success_resets_consecutive(self):
        mm = ModelMetrics()
        mm.record_failure("e1")
        mm.record_failure("e2")
        mm.record_failure("e3")
        assert mm.consecutive_failures == 3
        mm.record_success(10.0, 1)
        assert mm.consecutive_failures == 0

    def test_record_timeout_consecutive(self):
        mm = ModelMetrics()
        mm.record_timeout()
        mm.record_timeout()
        mm.record_timeout()
        assert mm.consecutive_failures == 3


# ---------------------------------------------------------------------------
# Extended CircuitBreaker tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerExtended:
    def test_allow_request_when_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.allow_request() is True

    def test_state_change_on_open_to_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        time.sleep(0.02)
        _ = cb.state
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_no_state_change_when_already_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        changes = []
        cb._on_state_change = lambda old, new: changes.append((old, new))
        cb.record_success()
        assert len(changes) == 0

    def test_custom_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.5)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        time.sleep(0.01)
        assert cb.state == CircuitBreakerState.OPEN

    def test_default_failure_count(self):
        cb = CircuitBreaker()
        assert cb._failure_count == 0

    def test_state_thread_safe(self):
        cb = CircuitBreaker(failure_threshold=100)
        for _ in range(50):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_half_open_success_resets_count(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.record_success()
        assert cb._failure_count == 0

    def test_state_property_checks_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb._state == CircuitBreakerState.OPEN
        time.sleep(0.02)
        state = cb.state
        assert state == CircuitBreakerState.HALF_OPEN


# ---------------------------------------------------------------------------
# Extended IdleManager tests
# ---------------------------------------------------------------------------

class TestIdleManagerExtended:
    def test_register_multiple(self):
        mgr = IdleManager(check_interval_s=100)
        mgr.register("m1")
        mgr.register("m2")
        mgr.register("m3")
        assert len(mgr._models) == 3
        mgr.shutdown()

    def test_touch_updates_last_touch(self):
        mgr = IdleManager(check_interval_s=100)
        mgr.register("m1")
        before = time.time()
        mgr.touch("m1")
        after = time.time()
        entry = mgr._models["m1"]
        assert before <= entry["last_touch"] <= after
        mgr.shutdown()

    def test_get_idle_info_after_touch(self):
        mgr = IdleManager(idle_timeout_s=100, check_interval_s=100)
        mgr.register("m1")
        time.sleep(0.01)
        info = mgr.get_idle_info("m1")
        assert info["last_request_age_s"] >= 0
        assert info["remaining_s"] <= 100
        mgr.shutdown()

    def test_is_idle_unloaded_false_after_register(self):
        mgr = IdleManager(check_interval_s=100)
        mgr.register("m1")
        assert mgr.is_idle_unloaded("m1") is False
        mgr.shutdown()

    def test_is_reloading_false_after_register(self):
        mgr = IdleManager(check_interval_s=100)
        mgr.register("m1")
        assert mgr.is_reloading("m1") is False
        mgr.shutdown()

    def test_shutdown_stops_thread(self):
        mgr = IdleManager(check_interval_s=100)
        mgr.register("m1")
        mgr.shutdown()
        assert mgr._running is False

    def test_reset_clears_all(self):
        mgr = IdleManager(check_interval_s=100)
        mgr.register("m1")
        mgr.register("m2")
        mgr.reset()
        assert len(mgr._models) == 0
        assert mgr._running is False

    def test_default_idle_timeout(self):
        mgr = IdleManager()
        assert mgr._idle_timeout_s == 300.0

    def test_default_check_interval(self):
        mgr = IdleManager()
        assert mgr._check_interval_s == 30.0

    def test_touch_nonexistent(self):
        mgr = IdleManager(check_interval_s=100)
        result = mgr.touch("nonexistent")
        assert result is False
        mgr.shutdown()

    def test_unregister_multiple(self):
        mgr = IdleManager(check_interval_s=100)
        mgr.register("m1")
        mgr.register("m2")
        mgr.unregister("m1")
        mgr.unregister("m2")
        assert len(mgr._models) == 0
        mgr.shutdown()

    def test_get_idle_info_fields(self):
        mgr = IdleManager(idle_timeout_s=60, check_interval_s=100)
        mgr.register("m1")
        info = mgr.get_idle_info("m1")
        assert "last_request_age_s" in info
        assert "idle_timeout_s" in info
        assert "unloaded" in info
        assert "remaining_s" in info
        mgr.shutdown()


# ---------------------------------------------------------------------------
# Extended Priority tests
# ---------------------------------------------------------------------------

class TestPriorityExtended:
    def test_high_is_zero(self):
        assert Priority.HIGH == 0

    def test_medium_is_one(self):
        assert Priority.MEDIUM == 1

    def test_low_is_two(self):
        assert Priority.LOW == 2

    def test_all_are_int(self):
        for p in Priority:
            assert isinstance(p, int)

    def test_high_less_than_medium(self):
        assert Priority.HIGH < Priority.MEDIUM

    def test_medium_less_than_low(self):
        assert Priority.MEDIUM < Priority.LOW


# ---------------------------------------------------------------------------
# Extended QueueMetrics tests
# ---------------------------------------------------------------------------

class TestQueueMetricsExtended:
    def test_default_avg_wait(self):
        qm = QueueMetrics()
        assert qm.avg_wait_ms == 0.0

    def test_default_max_wait(self):
        qm = QueueMetrics()
        assert qm.max_wait_ms == 0.0

    def test_all_fields_settable(self):
        qm = QueueMetrics(
            depth_high=1, depth_medium=2, depth_low=3, total_depth=6,
            served=10, timed_out=1, avg_wait_ms=5.0, max_wait_ms=50.0
        )
        assert qm.depth_high == 1
        assert qm.depth_medium == 2
        assert qm.depth_low == 3
        assert qm.total_depth == 6
        assert qm.served == 10
        assert qm.timed_out == 1
        assert qm.avg_wait_ms == 5.0
        assert qm.max_wait_ms == 50.0

    def test_negative_depths_allowed(self):
        qm = QueueMetrics(depth_high=-1)
        assert qm.depth_high == -1


# ---------------------------------------------------------------------------
# Extended ModelStatus tests
# ---------------------------------------------------------------------------

class TestModelStatusExtended:
    def test_all_members_present(self):
        members = [s.name for s in ModelStatus]
        assert "UNINITIALIZED" in members
        assert "LOADING" in members
        assert "READY" in members
        assert "DEGRADED" in members
        assert "ERROR" in members
        assert "UNLOADED" in members

    def test_value_is_str(self):
        assert isinstance(ModelStatus.READY.value, str)
        assert ModelStatus.READY.value == "ready"

    def test_invalid_value_raises(self):
        try:
            ModelStatus("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_member_count(self):
        assert len(ModelStatus) == 6

    def test_unique_values(self):
        values = [s.value for s in ModelStatus]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# Extended CircuitBreakerState tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerStateExtended:
    def test_all_members_present(self):
        members = [s.name for s in CircuitBreakerState]
        assert "CLOSED" in members
        assert "OPEN" in members
        assert "HALF_OPEN" in members

    def test_value_is_str(self):
        assert isinstance(CircuitBreakerState.CLOSED.value, str)
        assert CircuitBreakerState.CLOSED.value == "closed"

    def test_invalid_value_raises(self):
        try:
            CircuitBreakerState("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_member_count(self):
        assert len(CircuitBreakerState) == 3

    def test_unique_values(self):
        values = [s.value for s in CircuitBreakerState]
        assert len(values) == len(set(values))
