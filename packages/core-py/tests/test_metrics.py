"""
Tests for the Prometheus metrics collector.
"""

import time
import pytest
from domains.infrastructure.metrics import MetricsCollector, get_metrics_collector, reset_metrics_collector


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_render_returns_string(self):
        c = MetricsCollector()
        out = c.render()
        assert isinstance(out, str)
        assert "sloughgpt_uptime_seconds" in out

    def test_record_request_counter(self):
        c = MetricsCollector()
        c.record_request("/chat", 200, 0.5)
        c.record_request("/chat", 200, 0.3)
        out = c.render()
        assert 'sloughgpt_requests_total{path="/chat"} 2' in out

    def test_record_request_error(self):
        c = MetricsCollector()
        c.record_request("/chat", 500, 1.0)
        out = c.render()
        assert 'sloughgpt_request_errors_total{path="/chat"} 1' in out

    def test_request_latency_percentiles(self):
        c = MetricsCollector()
        for i in range(100):
            c.record_request("/api", 200, i * 0.01)
        out = c.render()
        assert 'sloughgpt_request_duration_seconds{path="/api",quantile="0.5"' in out

    def test_inference_counter(self):
        c = MetricsCollector()
        c.record_inference(1.5, tokens=50)
        c.record_inference(2.0, tokens=30)
        out = c.render()
        assert "sloughgpt_inferences_total 2" in out
        assert "sloughgpt_tokens_generated_total 80" in out

    def test_active_requests_gauge(self):
        c = MetricsCollector()
        c.set_active_requests(5)
        out = c.render()
        assert "sloughgpt_active_requests 5" in out

    def test_model_loaded_gauge(self):
        c = MetricsCollector()
        c.set_model_info(True, "gpt2")
        out = c.render()
        assert "sloughgpt_model_loaded 1" in out

    def test_model_not_loaded(self):
        c = MetricsCollector()
        c.set_model_info(False)
        out = c.render()
        assert "sloughgpt_model_loaded 0" in out

    def test_singleton(self):
        c1 = get_metrics_collector()
        c2 = get_metrics_collector()
        assert c1 is c2

    def test_uptime_increases(self):
        c = MetricsCollector()
        out1 = c.render()
        time.sleep(0.05)
        out2 = c.render()
        t1 = float([l for l in out1.split("\n") if "uptime" in l and not l.startswith("#")][0].split()[-1])
        t2 = float([l for l in out2.split("\n") if "uptime" in l and not l.startswith("#")][0].split()[-1])
        assert t2 >= t1

    def test_thread_safety(self):
        """Record 1000 requests from multiple threads."""
        import threading
        c = MetricsCollector()
        errors = []

        def _worker():
            try:
                for _ in range(200):
                    c.record_request("/test", 200, 0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        out = c.render()
        assert 'sloughgpt_requests_total{path="/test"} 1000' in out

    def test_multiple_paths(self):
        c = MetricsCollector()
        c.record_request("/chat", 200, 0.1)
        c.record_request("/inference", 200, 0.5)
        out = c.render()
        assert 'path="/chat"' in out
        assert 'path="/inference"' in out

    def test_error_status_codes(self):
        c = MetricsCollector()
        for code in [400, 404, 500, 502, 503]:
            c.record_request(f"/err{code}", code, 0.1)
        out = c.render()
        for code in [400, 404, 500, 502, 503]:
            assert f'sloughgpt_request_errors_total{{path="/err{code}"}} 1' in out

    def test_success_not_counted_as_error(self):
        c = MetricsCollector()
        c.record_request("/ok", 200, 0.1)
        c.record_request("/ok", 201, 0.1)
        c.record_request("/ok", 204, 0.1)
        out = c.render()
        # No error counter is emitted for /ok since status < 400
        assert 'sloughgpt_request_errors_total{path="/ok"}' not in out

    def test_inference_zero_tokens(self):
        c = MetricsCollector()
        c.record_inference(0.5, tokens=0)
        out = c.render()
        assert "sloughgpt_tokens_generated_total 0" in out

    def test_inference_many_calls(self):
        c = MetricsCollector()
        for _ in range(100):
            c.record_inference(0.01, tokens=1)
        out = c.render()
        assert "sloughgpt_inferences_total 100" in out
        assert "sloughgpt_tokens_generated_total 100" in out

    def test_active_requests_zero(self):
        c = MetricsCollector()
        c.set_active_requests(0)
        out = c.render()
        assert "sloughgpt_active_requests 0" in out

    def test_active_requests_negative(self):
        c = MetricsCollector()
        c.set_active_requests(-5)
        out = c.render()
        assert "sloughgpt_active_requests -5" in out

    def test_model_name_stored(self):
        c = MetricsCollector()
        c.set_model_info(True, "mymodel")
        assert c._model_name == "mymodel"

    def test_render_prometheus_format(self):
        c = MetricsCollector()
        c.record_request("/api", 200, 0.1)
        out = c.render()
        lines = out.strip().split("\n")
        help_lines = [l for l in lines if l.startswith("# HELP")]
        type_lines = [l for l in lines if l.startswith("# TYPE")]
        assert len(help_lines) > 0
        assert len(type_lines) > 0

    def test_request_count_multiple_paths(self):
        c = MetricsCollector()
        c.record_request("/a", 200, 0.1)
        c.record_request("/b", 200, 0.1)
        c.record_request("/a", 200, 0.1)
        out = c.render()
        assert 'sloughgpt_requests_total{path="/a"} 2' in out
        assert 'sloughgpt_requests_total{path="/b"} 1' in out

    def test_inference_latency_percentiles(self):
        c = MetricsCollector()
        for i in range(50):
            c.record_inference(i * 0.01)
        out = c.render()
        assert "sloughgpt_inference_duration_seconds" in out

    def test_single_inference_latency(self):
        c = MetricsCollector()
        c.record_inference(1.0)
        out = c.render()
        assert "sloughgpt_inference_duration_seconds" in out

    def test_reset_singleton(self):
        reset_metrics_collector()
        c1 = get_metrics_collector()
        c2 = get_metrics_collector()
        assert c1 is c2

    def test_get_active_requests(self):
        c = MetricsCollector()
        c.set_active_requests(7)
        assert c.get_active_requests() == 7

    def test_get_active_requests_thread_safe(self):
        c = MetricsCollector()
        c.set_active_requests(10)
        import threading
        results = []
        def _read():
            results.append(c.get_active_requests())
        threads = [threading.Thread(target=_read) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r == 10 for r in results)

    def test_set_model_info_unloaded(self):
        c = MetricsCollector()
        c.set_model_info(True, "model_a")
        c.set_model_info(False)
        out = c.render()
        assert "sloughgpt_model_loaded 0" in out

    def test_request_latency_single_value(self):
        c = MetricsCollector()
        c.record_request("/single", 200, 0.5)
        out = c.render()
        assert 'sloughgpt_request_duration_seconds{path="/single",quantile="0.5"' in out

    def test_request_latency_all_same(self):
        c = MetricsCollector()
        for _ in range(10):
            c.record_request("/same", 200, 1.0)
        out = c.render()
        assert 'quantile="0.5"' in out

    def test_uptime_starts_near_zero(self):
        c = MetricsCollector()
        out = c.render()
        uptime = float([l for l in out.split("\n") if "uptime" in l and not l.startswith("#")][0].split()[-1])
        assert uptime < 1.0

    def test_concurrent_increments(self):
        import threading
        c = MetricsCollector()
        def _inc():
            for _ in range(100):
                c.record_request("/conc", 200, 0.01)
        threads = [threading.Thread(target=_inc) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        out = c.render()
        assert 'sloughgpt_requests_total{path="/conc"} 400' in out
