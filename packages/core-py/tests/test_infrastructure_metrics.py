"""Tests for MetricsCollector — Prometheus-style metrics."""
from __future__ import annotations

from domains.infrastructure.metrics import MetricsCollector, get_metrics_collector, reset_metrics_collector


class TestMetricsCollector:
    def test_record_request(self):
        mc = MetricsCollector()
        mc.record_request("/chat", 200, 1.5)
        rendered = mc.render()
        assert "sloughgpt_requests_total" in rendered
        assert "/chat" in rendered

    def test_record_error(self):
        mc = MetricsCollector()
        mc.record_request("/api", 500, 0.1)
        rendered = mc.render()
        assert "sloughgpt_request_errors_total" in rendered

    def test_record_inference(self):
        mc = MetricsCollector()
        mc.record_inference(0.5, tokens=100)
        rendered = mc.render()
        assert "sloughgpt_inferences_total 1" in rendered
        assert "sloughgpt_tokens_generated_total 100" in rendered

    def test_set_active_requests(self):
        mc = MetricsCollector()
        mc.set_active_requests(5)
        assert mc.get_active_requests() == 5
        rendered = mc.render()
        assert "sloughgpt_active_requests 5" in rendered

    def test_set_model_info(self):
        mc = MetricsCollector()
        mc.set_model_info(True, "gpt2")
        rendered = mc.render()
        assert "sloughgpt_model_loaded 1" in rendered

    def test_render_includes_uptime(self):
        mc = MetricsCollector()
        rendered = mc.render()
        assert "sloughgpt_uptime_seconds" in rendered

    def test_percentiles(self):
        mc = MetricsCollector()
        for i in range(100):
            mc.record_request("/test", 200, float(i))
        rendered = mc.render()
        assert "quantile" in rendered

    def test_empty_render(self):
        mc = MetricsCollector()
        rendered = mc.render()
        assert "sloughgpt_uptime_seconds" in rendered
        # HELP/TYPE lines for requests appear even with no data
        assert "sloughgpt_inferences_total 0" in rendered


class TestSingleton:
    def test_get_metrics_collector(self):
        reset_metrics_collector()
        a = get_metrics_collector()
        b = get_metrics_collector()
        assert a is b
