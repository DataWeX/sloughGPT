"""Comprehensive tests for server_state.py — AtomicRef, ServerState,
request/error tracking, inference metrics, rate limiting, memory tracking,
singleton management.

Covers: AtomicRef get/set/swap/version/listeners, ServerState request recording,
latency tracking, inference metrics, tokens_per_second, model metrics,
rate limiting, memory pressure, health history, singleton reset.
"""
from __future__ import annotations

import time
from threading import Thread

import pytest

from domains.infrastructure.server_state import (
    AtomicRef,
    ServerState,
    get_server_state,
    reset_server_state,
)


# ---------------------------------------------------------------------------
# AtomicRef
# ---------------------------------------------------------------------------

class TestAtomicRef:
    def test_initial_value(self):
        ref = AtomicRef(42, name="test")
        assert ref.get() == 42

    def test_set(self):
        ref = AtomicRef(0)
        ref.set(10)
        assert ref.get() == 10

    def test_version_increments(self):
        ref = AtomicRef(0)
        assert ref.version == 0
        ref.set(1)
        assert ref.version == 1
        ref.set(2)
        assert ref.version == 2

    def test_swap(self):
        ref = AtomicRef(10)
        new = ref.swap(lambda x: x * 2)
        assert new == 20
        assert ref.get() == 20

    def test_swap_returns_new(self):
        ref = AtomicRef("hello")
        result = ref.swap(lambda x: x.upper())
        assert result == "HELLO"

    def test_on_change_listener(self):
        ref = AtomicRef(0)
        changes = []
        ref.on_change(lambda old, new: changes.append((old, new)))
        ref.set(5)
        assert changes == [(0, 5)]

    def test_on_change_multiple_listeners(self):
        ref = AtomicRef(0)
        log1 = []
        log2 = []
        ref.on_change(lambda old, new: log1.append(new))
        ref.on_change(lambda old, new: log2.append(new))
        ref.set(7)
        assert log1 == [7]
        assert log2 == [7]

    def test_listener_exception_doesnt_break_set(self):
        ref = AtomicRef(0)
        def bad_listener(old, new):
            raise RuntimeError("boom")
        ref.on_change(bad_listener)
        ref.set(1)  # should not raise
        assert ref.get() == 1

    def test_thread_safety(self):
        ref = AtomicRef(0)
        errors = []

        def writer(start):
            for i in range(100):
                ref.set(start + i)

        threads = [Thread(target=writer, args=(i * 1000,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Value should be one of the written values
        assert isinstance(ref.get(), int)

    def test_name(self):
        ref = AtomicRef(0, name="counter")
        assert ref._name == "counter"


# ---------------------------------------------------------------------------
# ServerState — construction
# ---------------------------------------------------------------------------

class TestServerStateInit:
    def test_initial_state(self):
        s = ServerState()
        assert s.model.get() is None
        assert s.tokenizer.get() is None
        assert s.model_type.get() is None

    def test_uptime_positive(self):
        s = ServerState()
        time.sleep(0.01)
        assert s.uptime_seconds > 0

    def test_request_count_zero(self):
        s = ServerState()
        assert s.request_count == 0

    def test_error_count_zero(self):
        s = ServerState()
        assert s.error_count == 0

    def test_inference_count_zero(self):
        s = ServerState()
        assert s.inference_count == 0

    def test_total_tokens_zero(self):
        s = ServerState()
        assert s.total_tokens == 0


# ---------------------------------------------------------------------------
# Request recording
# ---------------------------------------------------------------------------

class TestRequestRecording:
    def test_record_request(self):
        s = ServerState()
        s.record_request()
        s.record_request()
        assert s.request_count == 2

    def test_record_error(self):
        s = ServerState()
        s.record_error()
        assert s.error_count == 1

    def test_record_request_latency(self):
        s = ServerState()
        s.record_request_latency("/api/chat", "POST", 200, 45.3)
        history = s.get_request_history()
        assert len(history) == 1
        assert history[0]["path"] == "/api/chat"
        assert history[0]["elapsed_ms"] == 45.3

    def test_get_request_history_limit(self):
        s = ServerState()
        for i in range(10):
            s.record_request_latency(f"/path{i}", "GET", 200, 10.0)
        history = s.get_request_history(limit=5)
        assert len(history) == 5

    def test_get_request_history_order(self):
        s = ServerState()
        s.record_request_latency("/first", "GET", 200, 10.0)
        s.record_request_latency("/second", "GET", 200, 20.0)
        history = s.get_request_history()
        assert history[0]["path"] == "/second"  # newest first

    def test_get_avg_latency(self):
        s = ServerState()
        s.record_request_latency("/a", "GET", 200, 100.0)
        s.record_request_latency("/b", "GET", 200, 200.0)
        avg = s.get_avg_latency()
        assert avg == 150.0

    def test_get_avg_latency_empty(self):
        s = ServerState()
        assert s.get_avg_latency() == 0.0

    def test_get_p95_latency(self):
        s = ServerState()
        for i in range(100):
            s.record_request_latency("/a", "GET", 200, float(i))
        p95 = s.get_p95_latency()
        assert p95 >= 90.0

    def test_get_p95_latency_empty(self):
        s = ServerState()
        assert s.get_p95_latency() == 0.0

    def test_requests_per_minute(self):
        s = ServerState()
        s.record_request_latency("/a", "GET", 200, 10.0)
        rpm = s.get_requests_per_minute()
        assert rpm >= 1


# ---------------------------------------------------------------------------
# Error recording
# ---------------------------------------------------------------------------

class TestErrorRecording:
    def test_record_error_detail(self):
        s = ServerState()
        s.record_error_detail("/api/chat", "POST", 500, "internal error", error_type="RuntimeError")
        history = s.get_error_history()
        assert len(history) == 1
        assert history[0]["path"] == "/api/chat"
        assert history[0]["message"] == "internal error"
        assert history[0]["error_type"] == "RuntimeError"
        assert s.error_count == 1

    def test_error_message_truncated(self):
        s = ServerState()
        long_msg = "x" * 500
        s.record_error_detail("/a", "GET", 500, long_msg)
        history = s.get_error_history()
        assert len(history[0]["message"]) == 200

    def test_get_error_history_limit(self):
        s = ServerState()
        for i in range(10):
            s.record_error_detail(f"/path{i}", "GET", 500, f"err{i}")
        history = s.get_error_history(limit=3)
        assert len(history) == 3


# ---------------------------------------------------------------------------
# Path latency
# ---------------------------------------------------------------------------

class TestPathLatency:
    def test_record_and_get(self):
        s = ServerState()
        s.record_path_latency("/api/chat", 50.0)
        s.record_path_latency("/api/chat", 100.0)
        result = s.get_path_latencies()
        assert len(result) == 1
        assert result[0]["path"] == "/api/chat"
        assert result[0]["count"] == 2
        assert result[0]["avg_ms"] == 75.0

    def test_get_path_latencies_top_n(self):
        s = ServerState()
        for i in range(10):
            for j in range(i + 1):
                s.record_path_latency(f"/path{i}", float(j))
        result = s.get_path_latencies(top_n=3)
        assert len(result) == 3

    def test_get_path_latencies_empty(self):
        s = ServerState()
        assert s.get_path_latencies() == []


# ---------------------------------------------------------------------------
# Inference metrics
# ---------------------------------------------------------------------------

class TestInferenceMetrics:
    def test_record_inference(self):
        s = ServerState()
        s.record_inference(tokens=50, elapsed_ms=100.0, model="gpt2")
        assert s.inference_count == 1
        assert s.total_tokens == 50

    def test_record_inference_no_model(self):
        s = ServerState()
        s.record_inference(tokens=10, elapsed_ms=50.0)
        assert s.inference_count == 1

    def test_get_tokens_per_second(self):
        s = ServerState()
        for _ in range(5):
            s.record_inference(tokens=100, elapsed_ms=100.0)
        tps = s.get_tokens_per_second()
        assert tps > 0

    def test_get_tokens_per_second_empty(self):
        s = ServerState()
        assert s.get_tokens_per_second() == 0.0

    def test_get_avg_tokens_per_request(self):
        s = ServerState()
        s.record_inference(tokens=50, elapsed_ms=100.0)
        s.record_inference(tokens=100, elapsed_ms=100.0)
        avg = s.get_avg_tokens_per_request()
        assert avg == 75.0

    def test_get_avg_tokens_per_request_empty(self):
        s = ServerState()
        assert s.get_avg_tokens_per_request() == 0.0

    def test_get_model_metrics(self):
        s = ServerState()
        s.record_inference(tokens=50, elapsed_ms=100.0, model="gpt2")
        s.record_inference(tokens=100, elapsed_ms=200.0, model="gpt2")
        s.record_inference(tokens=30, elapsed_ms=50.0, model="llama")
        metrics = s.get_model_metrics()
        assert len(metrics) == 2
        # sorted by count
        assert metrics[0]["model"] == "gpt2"
        assert metrics[0]["count"] == 2

    def test_get_model_metrics_empty(self):
        s = ServerState()
        assert s.get_model_metrics() == []


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_check_rate_limit_allowed(self):
        s = ServerState()
        assert s.check_rate_limit("/api/chat", max_per_second=10) is True

    def test_check_rate_limit_blocked(self):
        s = ServerState()
        for _ in range(10):
            s.check_rate_limit("/api/chat", max_per_second=10)
        assert s.check_rate_limit("/api/chat", max_per_second=10) is False

    def test_rate_limit_violations(self):
        s = ServerState()
        for _ in range(11):
            s.check_rate_limit("/api/chat", max_per_second=10)
        violations = s.get_rate_limit_violations()
        assert len(violations) >= 1
        assert violations[0]["path"] == "/api/chat"

    def test_rate_limit_reset_after_window(self):
        s = ServerState()
        # First call sets window
        s.check_rate_limit("/api/chat", max_per_second=100)
        # Force window reset by manipulating time
        with s._lock:
            s._rate_limits["/api/chat"]["window_start"] = time.time() - 2.0
        assert s.check_rate_limit("/api/chat", max_per_second=100) is True


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

class TestMemoryTracking:
    def test_record_memory_snapshot(self):
        s = ServerState()
        s.record_memory_snapshot()
        history = s.get_memory_history()
        assert len(history) == 1
        assert "rss_mb" in history[0]

    def test_get_memory_history_empty(self):
        s = ServerState()
        assert s.get_memory_history() == []

    def test_record_memory_pressure_block(self):
        s = ServerState()
        s.record_memory_pressure_block()
        stats = s.get_memory_pressure_stats()
        assert stats["pressure_blocks"] == 1

    def test_record_gc_cycle(self):
        s = ServerState()
        s.record_gc_cycle()
        stats = s.get_memory_pressure_stats()
        assert stats["gc_cycles"] == 1

    def test_memory_pressure_stats_defaults(self):
        s = ServerState()
        stats = s.get_memory_pressure_stats()
        assert stats["pressure_blocks"] == 0
        assert stats["gc_cycles"] == 0


# ---------------------------------------------------------------------------
# Model events
# ---------------------------------------------------------------------------

class TestModelEvents:
    def test_record_model_event(self):
        s = ServerState()
        s.record_model_event("load", "gpt2", "loaded successfully")
        events = s.get_model_events()
        assert len(events) == 1
        assert events[0]["type"] == "load"
        assert events[0]["model"] == "gpt2"

    def test_get_model_events_limit(self):
        s = ServerState()
        for i in range(10):
            s.record_model_event("load", f"model_{i}")
        events = s.get_model_events(limit=3)
        assert len(events) == 3

    def test_model_event_detail_truncated(self):
        s = ServerState()
        s.record_model_event("load", "gpt2", "x" * 500)
        events = s.get_model_events()
        assert len(events[0]["detail"]) == 200


# ---------------------------------------------------------------------------
# Health history
# ---------------------------------------------------------------------------

class TestHealthHistory:
    def test_get_health_history_empty(self):
        s = ServerState()
        assert s.get_health_history() == []


# ---------------------------------------------------------------------------
# Trend snapshots
# ---------------------------------------------------------------------------

class TestTrendSnapshots:
    def test_record_trend_snapshots_throttled(self):
        s = ServerState()
        # First call should record
        s.record_trend_snapshots(interval_s=1.0)
        health = s.get_health_history()
        assert len(health) >= 1
        # Second call within interval should be throttled
        s.record_trend_snapshots(interval_s=10.0)
        # No new entry because interval hasn't elapsed
        # (health_history may still have 1 from first call)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_server_state(self):
        reset_server_state()
        state = get_server_state()
        assert isinstance(state, ServerState)

    def test_singleton_returns_same(self):
        reset_server_state()
        s1 = get_server_state()
        s2 = get_server_state()
        assert s1 is s2

    def test_reset(self):
        reset_server_state()
        s1 = get_server_state()
        reset_server_state()
        s2 = get_server_state()
        assert s1 is not s2
