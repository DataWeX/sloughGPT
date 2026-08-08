"""Tests for domains/infrastructure/server_state.py — thread-safe state management."""

import time
import threading
import pytest
from domains.infrastructure.server_state import AtomicRef, ServerState, get_server_state


class TestAtomicRef:
    def test_get_set(self):
        ref = AtomicRef(42, "test")
        assert ref.get() == 42
        ref.set(99)
        assert ref.get() == 99

    def test_swap(self):
        ref = AtomicRef(10, "test")
        result = ref.swap(lambda x: x * 2)
        assert result == 20
        assert ref.get() == 20

    def test_version_increments(self):
        ref = AtomicRef(0, "test")
        assert ref.version == 0
        ref.set(1)
        assert ref.version == 1
        ref.swap(lambda x: x + 1)
        assert ref.version == 2

    def test_on_change_listener(self):
        ref = AtomicRef(0, "test")
        changes = []
        ref.on_change(lambda old, new: changes.append((old, new)))
        ref.set(5)
        assert changes == [(0, 5)]

    def test_listener_receives_old_and_new(self):
        ref = AtomicRef("a", "test")
        changes = []
        ref.on_change(lambda old, new: changes.append((old, new)))
        ref.set("b")
        ref.set("c")
        assert changes == [("a", "b"), ("b", "c")]

    def test_listener_exception_does_not_propagate(self):
        ref = AtomicRef(0, "test")
        ref.on_change(lambda old, new: 1 / 0)
        ref.set(1)
        assert ref.get() == 1

    def test_thread_safety(self):
        ref = AtomicRef(0, "test")
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    ref.set(n * 1000 + i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert isinstance(ref.get(), int)


class TestServerState:
    def test_fresh_state(self):
        state = ServerState()
        assert state.model.get() is None
        assert state.tokenizer.get() is None
        assert state.request_count == 0
        assert state.error_count == 0

    def test_uptime_positive(self):
        state = ServerState()
        assert state.uptime_seconds >= 0

    def test_record_request(self):
        state = ServerState()
        state.record_request()
        state.record_request()
        assert state.request_count == 2

    def test_record_error(self):
        state = ServerState()
        state.record_error()
        assert state.error_count == 1

    def test_request_history(self):
        state = ServerState()
        state.record_request_latency("/chat", "POST", 200, 150.0)
        state.record_request_latency("/models", "GET", 200, 50.0)
        history = state.get_request_history()
        assert len(history) == 2
        assert history[0]["path"] == "/models"

    def test_request_history_ring_buffer(self):
        state = ServerState()
        for i in range(60):
            state.record_request_latency(f"/path{i}", "GET", 200, 10.0)
        history = state.get_request_history(limit=100)
        assert len(history) == 50

    def test_avg_latency(self):
        state = ServerState()
        state.record_request_latency("/a", "GET", 200, 100.0)
        state.record_request_latency("/b", "GET", 200, 200.0)
        assert state.get_avg_latency() == 150.0

    def test_avg_latency_empty(self):
        state = ServerState()
        assert state.get_avg_latency() == 0.0

    def test_error_history(self):
        state = ServerState()
        state.record_error_detail("/chat", "POST", 500, "boom", "RuntimeError")
        errors = state.get_error_history()
        assert len(errors) == 1
        assert errors[0]["message"] == "boom"
        assert errors[0]["error_type"] == "RuntimeError"

    def test_error_message_truncated(self):
        state = ServerState()
        long_msg = "x" * 300
        state.record_error_detail("/chat", "POST", 500, long_msg)
        errors = state.get_error_history()
        assert len(errors[0]["message"]) == 200

    def test_path_latencies(self):
        state = ServerState()
        for _ in range(10):
            state.record_path_latency("/chat", 100.0)
        for _ in range(5):
            state.record_path_latency("/models", 50.0)
        top = state.get_path_latencies()
        assert top[0]["path"] == "/chat"
        assert top[0]["count"] == 10

    def test_requests_per_minute(self):
        state = ServerState()
        state.record_request_latency("/a", "GET", 200, 1.0)
        rpm = state.get_requests_per_minute()
        assert rpm >= 1

    def test_inference_tracking(self):
        state = ServerState()
        state.record_inference(50, 1000.0, "gpt2")
        state.record_inference(30, 500.0, "gpt2")
        assert state.inference_count == 2
        assert state.total_tokens == 80
        # 80 tokens / (1500ms / 1000) = 53.3 tok/s
        assert abs(state.get_tokens_per_second() - 53.3) < 0.1

    def test_tokens_per_second_no_data(self):
        state = ServerState()
        assert state.get_tokens_per_second() == 0.0

    def test_avg_tokens_per_request(self):
        state = ServerState()
        state.record_inference(100, 1000.0)
        state.record_inference(200, 1000.0)
        assert state.get_avg_tokens_per_request() == 150.0

    def test_model_metrics(self):
        state = ServerState()
        state.record_inference(50, 1000.0, "gpt2")
        state.record_inference(100, 500.0, "qwen")
        metrics = state.get_model_metrics()
        assert len(metrics) == 2
        model_names = {m["model"] for m in metrics}
        assert "gpt2" in model_names
        assert "qwen" in model_names

    def test_model_events(self):
        state = ServerState()
        state.record_model_event("load", "gpt2", "loaded")
        state.record_model_event("unload", "gpt2")
        events = state.get_model_events()
        assert len(events) == 2
        assert events[0]["type"] == "unload"

    def test_rate_limit_allows_normal(self):
        state = ServerState()
        assert state.check_rate_limit("/chat", max_per_second=30) is True

    def test_rate_limit_blocks_excess(self):
        state = ServerState()
        for _ in range(31):
            state.check_rate_limit("/chat", max_per_second=30)
        assert state.check_rate_limit("/chat", max_per_second=30) is False

    def test_rate_limit_violations(self):
        state = ServerState()
        for _ in range(35):
            state.check_rate_limit("/chat", max_per_second=30)
        violations = state.get_rate_limit_violations()
        assert len(violations) >= 1

    def test_trend_snapshots_record(self):
        state = ServerState()
        state.record_trend_snapshots(interval_s=0)
        assert len(state.get_health_history()) >= 1
        assert len(state.get_memory_history()) >= 1

    def test_trend_snapshots_throttled(self):
        state = ServerState()
        state.record_trend_snapshots(interval_s=60)
        assert len(state.get_health_history()) >= 1
        state.record_trend_snapshots(interval_s=60)
        assert len(state.get_health_history()) == 1

    def test_trend_snapshots_oldest_first(self):
        state = ServerState()
        for _ in range(3):
            state.record_trend_snapshots(interval_s=0)
            time.sleep(0.01)
        history = state.get_health_history(20)
        assert history == sorted(history, key=lambda h: h["ts"])

    def test_singleton(self):
        s1 = get_server_state()
        s2 = get_server_state()
        assert s1 is s2


class TestServerStateAdvanced:
    def test_swap_listener_exception_tolerated(self):
        ref = AtomicRef(0, "test")
        ref.on_change(lambda old, new: 1 / 0)
        assert ref.swap(lambda x: x + 1) == 1
        assert ref.get() == 1

    def test_error_history_ring_buffer(self):
        state = ServerState()
        for i in range(25):
            state.record_error_detail(f"/p{i}", "GET", 500, "e")
        assert len(state.get_error_history(limit=100)) == 20

    def test_path_latency_ring_buffer(self):
        state = ServerState()
        for _ in range(120):
            state.record_path_latency("/chat", 1.0)
        lat = state.get_path_latencies(top_n=5)
        assert lat[0]["count"] == 100

    def test_path_latencies_skips_empty(self):
        state = ServerState()
        state.record_path_latency("/a", 10.0)
        state._path_latencies["/empty"] = []
        top = state.get_path_latencies()
        assert [t["path"] for t in top] == ["/a"]

    def test_avg_tokens_per_request_no_data(self):
        state = ServerState()
        assert state.get_avg_tokens_per_request() == 0.0

    def test_tokens_per_request_ring_buffer(self):
        state = ServerState()
        for _ in range(60):
            state.record_inference(10, 100.0)
        assert state.total_tokens == 600
        assert len(state._tokens_per_request) == 50

    def test_model_events_ring_buffer(self):
        state = ServerState()
        for i in range(35):
            state.record_model_event("load", f"m{i}")
        assert len(state.get_model_events(limit=100)) == 30

    def test_health_score_shape(self):
        state = ServerState()
        state.model.set(type("Fake", (), {"name_or_path": "gpt2"})())
        state.record_request_latency("/a", "GET", 200, 10.0)
        state.record_inference(100, 1000.0, "gpt2")
        h = state.get_health_score()
        assert "score" in h and "status" in h and "summary" in h and "diagnoses" in h

    def test_health_snapshot_and_history(self):
        state = ServerState()
        state.record_health_snapshot()
        state.record_health_snapshot()
        hist = state.get_health_history()
        assert len(hist) == 2
        assert hist[0]["ts"] <= hist[1]["ts"]

    def test_health_history_ring_buffer(self):
        state = ServerState()
        for _ in range(40):
            state.record_health_snapshot()
        assert len(state.get_health_history(limit=100)) == 30

    def test_memory_snapshot_and_history(self):
        state = ServerState()
        state.record_memory_snapshot()
        hist = state.get_memory_history()
        assert len(hist) == 1
        assert "rss_mb" in hist[0]
        assert "virtual_mb" in hist[0]

    def test_memory_history_ring_buffer(self):
        state = ServerState()
        for _ in range(40):
            state.record_memory_snapshot()
        assert len(state.get_memory_history(limit=100)) == 30

    def test_memory_snapshot_resource_fallback_and_psutil(self, monkeypatch):
        import sys
        import types as types_mod

        class FakeMem:
            percent = 42.5

        class FakeProcInfo:
            rss = 100 * 1024 * 1024
            vms = 200 * 1024 * 1024

        class FakeProcess:
            def memory_info(self):
                return FakeProcInfo()

        fake = types_mod.SimpleNamespace(virtual_memory=lambda: FakeMem(), Process=FakeProcess)
        monkeypatch.setitem(sys.modules, "psutil", fake)
        import resource

        def boom():
            raise OSError("no rusage")

        monkeypatch.setattr(resource, "getrusage", boom)
        state = ServerState()
        state.record_memory_snapshot()
        hist = state.get_memory_history()
        assert hist[0]["rss_mb"] == pytest.approx(100.0)
        assert hist[0]["virtual_mb"] == pytest.approx(200.0)
        assert hist[0]["system_percent"] == pytest.approx(42.5)

    def test_rate_limit_window_resets(self):
        state = ServerState()
        state.check_rate_limit("/chat", max_per_second=30)
        state._rate_limits["/chat"]["window_start"] = time.time() - 2.0
        assert state.check_rate_limit("/chat", max_per_second=30) is True

    def test_rate_limit_violation_ring_buffer(self):
        state = ServerState()
        for _ in range(25):
            for __ in range(40):
                state.check_rate_limit("/x", max_per_second=30)
            state._rate_limits["/x"]["window_start"] = time.time() - 2.0
        assert len(state.get_rate_limit_violations(limit=100)) == 20
