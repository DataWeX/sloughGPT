"""Tests for domains.infrastructure.server_state — AtomicRef and ServerState."""

import threading
import time

import pytest


class TestAtomicRef:
    def test_get_initial(self):
        from domains.infrastructure.server_state import AtomicRef
        ref = AtomicRef(42, name="test")
        assert ref.get() == 42

    def test_set_and_get(self):
        from domains.infrastructure.server_state import AtomicRef
        ref = AtomicRef(0, name="test")
        ref.set(99)
        assert ref.get() == 99

    def test_swap(self):
        from domains.infrastructure.server_state import AtomicRef
        ref = AtomicRef(10, name="test")
        result = ref.swap(lambda x: x * 3)
        assert result == 30
        assert ref.get() == 30

    def test_version_increments(self):
        from domains.infrastructure.server_state import AtomicRef
        ref = AtomicRef("a")
        assert ref.version == 0
        ref.set("b")
        assert ref.version == 1
        ref.set("c")
        assert ref.version == 2

    def test_swap_increments_version(self):
        from domains.infrastructure.server_state import AtomicRef
        ref = AtomicRef(1)
        ref.swap(lambda x: x + 1)
        assert ref.version == 1

    def test_on_change_listener(self):
        from domains.infrastructure.server_state import AtomicRef
        ref = AtomicRef(1)
        changes = []
        ref.on_change(lambda old, new: changes.append((old, new)))
        ref.set(2)
        ref.set(3)
        assert changes == [(1, 2), (2, 3)]

    def test_swap_triggers_listener(self):
        from domains.infrastructure.server_state import AtomicRef
        ref = AtomicRef("x")
        changes = []
        ref.on_change(lambda old, new: changes.append((old, new)))
        ref.swap(lambda old: old + "y")
        assert changes == [("x", "xy")]

    def test_listener_exception_does_not_crash(self):
        from domains.infrastructure.server_state import AtomicRef
        ref = AtomicRef(0)
        ref.on_change(lambda old, new: 1 / 0)
        ref.set(1)  # should not raise
        assert ref.get() == 1

    def test_thread_safety(self):
        from domains.infrastructure.server_state import AtomicRef
        ref = AtomicRef(0)
        errors = []

        def increment(n):
            try:
                for _ in range(100):
                    ref.swap(lambda x: x + 1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert ref.get() == 1000  # 10 threads × 100 increments
        assert errors == []


class TestServerState:
    def _fresh_state(self):
        from domains.infrastructure.server_state import ServerState
        return ServerState()

    def test_initial_values(self):
        s = self._fresh_state()
        assert s.model.get() is None
        assert s.current_soul.get() is None
        assert s.gen_config.get() is None
        assert s.request_count == 0
        assert s.error_count == 0

    def test_uptime_positive(self):
        s = self._fresh_state()
        assert s.uptime_seconds >= 0

    def test_record_request(self):
        s = self._fresh_state()
        s.record_request()
        s.record_request()
        assert s.request_count == 2

    def test_record_error(self):
        s = self._fresh_state()
        s.record_error()
        assert s.error_count == 1

    def test_request_history_ring_buffer(self):
        s = self._fresh_state()
        s._request_history_max = 3
        for i in range(5):
            s.record_request_latency(f"/p{i}", "GET", 200, 1.0)
        history = s.get_request_history()
        assert len(history) == 3
        assert history[0]["path"] == "/p4"  # newest first
        assert history[2]["path"] == "/p2"

    def test_get_request_history_limit(self):
        s = self._fresh_state()
        for i in range(10):
            s.record_request_latency(f"/p{i}", "GET", 200, 1.0)
        assert len(s.get_request_history(limit=5)) == 5

    def test_get_avg_latency_empty(self):
        s = self._fresh_state()
        assert s.get_avg_latency() == 0.0

    def test_get_avg_latency(self):
        s = self._fresh_state()
        s.record_request_latency("/a", "GET", 200, 10.0)
        s.record_request_latency("/b", "GET", 200, 30.0)
        assert s.get_avg_latency() == 20.0

    def test_error_history_ring_buffer(self):
        s = self._fresh_state()
        s._error_history_max = 3
        for i in range(5):
            s.record_error_detail(f"/p{i}", "GET", 500, f"err {i}")
        assert len(s.get_error_history()) == 3
        assert s.get_error_history()[0]["message"] == "err 4"

    def test_error_detail_message_truncated(self):
        s = self._fresh_state()
        s.record_error_detail("/x", "GET", 500, "x" * 300)
        assert len(s.get_error_history()[0]["message"]) == 200

    def test_path_latencies(self):
        s = self._fresh_state()
        s.record_path_latency("/api", 10.0)
        s.record_path_latency("/api", 20.0)
        s.record_path_latency("/other", 5.0)
        top = s.get_path_latencies(top_n=1)
        assert len(top) == 1
        assert top[0]["path"] == "/api"
        assert top[0]["count"] == 2
        assert top[0]["avg_ms"] == 15.0

    def test_requests_per_minute(self):
        s = self._fresh_state()
        s.record_request_latency("/x", "GET", 200, 1.0)
        rpm = s.get_requests_per_minute()
        assert rpm >= 1

    def test_record_inference(self):
        s = self._fresh_state()
        s.record_inference(100, 500.0, model="gpt2")
        s.record_inference(200, 1000.0, model="gpt2")
        assert s.inference_count == 2
        assert s.total_tokens == 300
        assert s.get_tokens_per_second() == pytest.approx(200.0, abs=1.0)
        assert s.get_avg_tokens_per_request() == 150.0

    def test_tokens_per_request_ring_buffer(self):
        s = self._fresh_state()
        s._tokens_per_request_max = 3
        for i in range(5):
            s.record_inference(i * 10, 100.0)
        assert len(s._tokens_per_request) == 3

    def test_model_metrics(self):
        s = self._fresh_state()
        s.record_inference(100, 500.0, "gpt2")
        s.record_inference(50, 200.0, "qwen")
        metrics = s.get_model_metrics()
        assert len(metrics) == 2
        assert metrics[0]["model"] == "gpt2"  # more count
        assert metrics[0]["count"] == 1

    def test_get_tokens_per_second_no_inference(self):
        s = self._fresh_state()
        assert s.get_tokens_per_second() == 0.0

    def test_model_events_ring_buffer(self):
        s = self._fresh_state()
        s._model_events_max = 3
        for i in range(5):
            s.record_model_event("load", f"model-{i}")
        events = s.get_model_events()
        assert len(events) == 3
        assert events[0]["model"] == "model-4"  # newest first

    def test_health_score(self):
        s = self._fresh_state()
        score = s.get_health_score()
        assert "score" in score
        assert "status" in score
        assert "summary" in score
        assert "diagnoses" in score

    def test_health_history(self):
        s = self._fresh_state()
        s.record_health_snapshot()
        history = s.get_health_history()
        assert len(history) == 1
        assert "score" in history[0]

    def test_memory_history(self):
        s = self._fresh_state()
        s.record_memory_snapshot()
        history = s.get_memory_history()
        assert len(history) == 1
        assert "rss_mb" in history[0]

    def test_rate_limit(self):
        s = self._fresh_state()
        assert s.check_rate_limit("/api", max_per_second=3) is True
        assert s.check_rate_limit("/api", max_per_second=3) is True
        assert s.check_rate_limit("/api", max_per_second=3) is True
        assert s.check_rate_limit("/api", max_per_second=3) is False
        violations = s.get_rate_limit_violations()
        assert len(violations) == 1
        assert violations[0]["path"] == "/api"

    def test_rate_limit_window_reset(self):
        s = self._fresh_state()
        s._rate_limits["/api"] = {"window_start": time.time() - 2.0, "count": 99}
        assert s.check_rate_limit("/api", max_per_second=30) is True

    def test_singleton_get_server_state(self):
        from domains.infrastructure.server_state import get_server_state, _server_state_lock
        import domains.infrastructure.server_state as mod
        with _server_state_lock:
            old = mod._server_state
            mod._server_state = None
        try:
            s1 = get_server_state()
            s2 = get_server_state()
            assert s1 is s2
        finally:
            with _server_state_lock:
                mod._server_state = old

    def test_path_latencies_p95(self):
        s = self._fresh_state()
        for i in range(20):
            s.record_path_latency("/x", float(i))
        top = s.get_path_latencies(top_n=1)
        assert top[0]["p95_ms"] > 0
