"""
Tests for the Prometheus metrics collector.
"""

import time
import pytest
from domains.infrastructure.metrics import MetricsCollector, get_metrics_collector


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
