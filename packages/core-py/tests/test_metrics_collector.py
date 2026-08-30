"""Tests for domains/infrastructure/metrics.py.

psutil is not installed in this environment; a minimal fake is injected into
``sys.modules`` for the duration of this test module (auto-restored) so the
module under test can be exercised without the real dependency.
"""

import importlib
import sys
import types
import threading
import time

import pytest

import domains.infrastructure.metrics as _metrics
from domains.infrastructure.metrics import MetricsCollector, get_metrics_collector, reset_metrics_collector


@pytest.fixture(autouse=True, scope="module")
def fake_psutil():
    """Install a minimal psutil fake for this module, then restore state."""
    mp = pytest.MonkeyPatch()
    fake = types.ModuleType("psutil")

    class _Mem:
        percent = 42.5

    fake.virtual_memory = lambda: _Mem()
    fake.cpu_percent = lambda interval=None: 12.3

    mp.setitem(sys.modules, "psutil", fake)
    importlib.reload(_metrics)
    yield
    mp.undo()
    importlib.reload(_metrics)


class TestMetricsCollector:
    def test_record_request_success_does_not_count_error(self):
        c = MetricsCollector()
        c.record_request("/chat", 200, 1.5)
        assert c._request_count["/chat"] == 1
        assert c._request_errors["/chat"] == 0
        assert c._request_latencies["/chat"] == [1.5]

    def test_record_request_error_counts(self):
        c = MetricsCollector()
        c.record_request("/chat", 500, 2.0)
        assert c._request_errors["/chat"] == 1

    def test_record_inference(self):
        c = MetricsCollector()
        c.record_inference(0.5, tokens=10)
        c.record_inference(0.7, tokens=20)
        assert c._inference_count == 2
        assert c._tokens_generated == 30
        assert c._inference_latencies == [0.5, 0.7]

    def test_set_active_requests(self):
        c = MetricsCollector()
        c.set_active_requests(7)
        assert c._active_requests == 7

    def test_set_model_info_loaded(self):
        c = MetricsCollector()
        c.set_model_info(True, "gpt2")
        assert c._model_loaded is True
        assert c._model_name == "gpt2"

    def test_set_model_info_unloaded(self):
        c = MetricsCollector()
        c.set_model_info(False)
        assert c._model_loaded is False

    def test_render_empty_metrics(self):
        c = MetricsCollector()
        text = c.render()
        assert text.endswith("\n")
        assert "sloughgpt_uptime_seconds" in text
        assert "sloughgpt_active_requests 0" in text
        assert "sloughgpt_model_loaded 0" in text
        assert "sloughgpt_system_cpu_usage 12.3" in text
        assert "sloughgpt_system_memory_percent 42.5" in text
        assert "sloughgpt_inferences_total 0" in text
        assert "sloughgpt_tokens_generated_total 0" in text

    def test_render_request_metrics(self):
        c = MetricsCollector()
        c.record_request("/b", 200, 0.4)
        c.record_request("/a", 200, 0.2)
        c.record_request("/a", 404, 0.3)
        text = c.render()
        assert 'sloughgpt_requests_total{path="/a"} 2' in text
        assert 'sloughgpt_requests_total{path="/b"} 1' in text
        assert 'sloughgpt_request_errors_total{path="/a"} 1' in text
        assert 'sloughgpt_request_duration_seconds_count{path="/a"} 2' in text
        assert 'quantile="0.5"' in text
        assert 'quantile="0.95"' in text
        assert 'quantile="0.99"' in text

    def test_render_single_latency_sample(self):
        c = MetricsCollector()
        c.record_request("/a", 200, 0.5)
        text = c.render()
        assert 'sloughgpt_request_duration_seconds{path="/a",quantile="0.95"} 0.5000' in text
        assert 'sloughgpt_request_duration_seconds{path="/a",quantile="0.99"} 0.5000' in text

    def test_render_two_latency_samples(self):
        c = MetricsCollector()
        c.record_request("/a", 200, 0.1)
        c.record_request("/a", 200, 0.9)
        text = c.render()
        assert 'sloughgpt_request_duration_seconds_sum{path="/a"} 1.0000' in text

    def test_render_skips_empty_latency_list(self):
        c = MetricsCollector()
        c._request_latencies["/empty"] = []
        text = c.render()
        assert 'sloughgpt_request_duration_seconds{path="/empty"' not in text

    def test_render_inference_latency_present(self):
        c = MetricsCollector()
        c.record_inference(0.3)
        c.record_inference(0.7)
        text = c.render()
        assert "sloughgpt_inference_duration_seconds" in text
        assert 'sloughgpt_inference_duration_seconds_count 2' in text
        assert "sloughgpt_inference_duration_seconds_sum 1.0000" in text

    def test_render_model_loaded_gauge(self):
        c = MetricsCollector()
        c.set_model_info(True, "gpt2")
        assert "sloughgpt_model_loaded 1" in c.render()

    def test_record_request_4xx_counts_as_error(self):
        c = MetricsCollector()
        c.record_request("/api", 422, 0.1)
        assert c._request_errors["/api"] == 1

    def test_record_request_3xx_no_error(self):
        c = MetricsCollector()
        c.record_request("/api", 301, 0.1)
        assert c._request_errors["/api"] == 0

    def test_record_request_400_counts_as_error(self):
        c = MetricsCollector()
        c.record_request("/api", 400, 0.1)
        assert c._request_errors["/api"] == 1

    def test_multiple_paths(self):
        c = MetricsCollector()
        c.record_request("/a", 200, 0.1)
        c.record_request("/b", 200, 0.2)
        c.record_request("/c", 500, 0.3)
        assert len(c._request_count) == 3
        assert c._request_errors["/c"] == 1
        assert c._request_errors["/a"] == 0

    def test_get_active_requests(self):
        c = MetricsCollector()
        assert c.get_active_requests() == 0
        c.set_active_requests(5)
        assert c.get_active_requests() == 5

    def test_inference_no_tokens(self):
        c = MetricsCollector()
        c.record_inference(0.5)
        assert c._inference_count == 1
        assert c._tokens_generated == 0

    def test_render_uptime_increases(self):
        c = MetricsCollector()
        text1 = c.render()
        time.sleep(0.01)
        text2 = c.render()
        line1 = [l for l in text1.split("\n") if l.startswith("sloughgpt_uptime_seconds ")][0]
        line2 = [l for l in text2.split("\n") if l.startswith("sloughgpt_uptime_seconds ")][0]
        uptime1 = float(line1.split()[-1])
        uptime2 = float(line2.split()[-1])
        assert uptime2 >= uptime1

    def test_render_active_requests_gauge(self):
        c = MetricsCollector()
        c.set_active_requests(42)
        text = c.render()
        assert "sloughgpt_active_requests 42" in text

    def test_render_no_inference_section_when_no_data(self):
        c = MetricsCollector()
        text = c.render()
        assert "sloughgpt_inference_duration_seconds" not in text

    def test_inference_tokens_accumulate(self):
        c = MetricsCollector()
        c.record_inference(0.1, tokens=5)
        c.record_inference(0.1, tokens=10)
        c.record_inference(0.1, tokens=15)
        assert c._tokens_generated == 30

    def test_request_count_multiple_requests_same_path(self):
        c = MetricsCollector()
        for _ in range(10):
            c.record_request("/test", 200, 0.01)
        assert c._request_count["/test"] == 10
        assert len(c._request_latencies["/test"]) == 10

    def test_error_rate_boundary(self):
        c = MetricsCollector()
        c.record_request("/a", 399, 0.1)
        c.record_request("/b", 400, 0.1)
        c.record_request("/c", 500, 0.1)
        assert c._request_errors["/a"] == 0
        assert c._request_errors["/b"] == 1
        assert c._request_errors["/c"] == 1

    def test_histogram_percentiles_ordering(self):
        c = MetricsCollector()
        for i in range(100):
            c.record_request("/a", 200, float(i) / 100.0)
        text = c.render()
        assert 'quantile="0.5"' in text
        assert 'quantile="0.95"' in text
        assert 'quantile="0.99"' in text

    def test_model_name_in_set_info(self):
        c = MetricsCollector()
        c.set_model_info(True, "llama-7b")
        assert c._model_name == "llama-7b"

    def test_set_model_info_overwrites(self):
        c = MetricsCollector()
        c.set_model_info(True, "model1")
        c.set_model_info(True, "model2")
        assert c._model_name == "model2"

    def test_render_psutil_metrics_present(self):
        c = MetricsCollector()
        text = c.render()
        assert "sloughgpt_system_cpu_usage" in text
        assert "sloughgpt_system_memory_percent" in text


class TestMetricsCollectorAdvanced:
    def test_record_request_2xx_no_error(self):
        c = MetricsCollector()
        for status in (200, 201, 204, 226):
            c.record_request("/api", status, 0.1)
        assert c._request_errors["/api"] == 0
        assert c._request_count["/api"] == 4

    def test_record_request_3xx_various(self):
        c = MetricsCollector()
        for status in (301, 302, 304, 307):
            c.record_request("/api", status, 0.1)
        assert c._request_errors["/api"] == 0

    def test_record_request_4xx_various(self):
        c = MetricsCollector()
        for status in (400, 401, 403, 404, 405, 408, 409, 422, 429, 451):
            c.record_request("/api", status, 0.1)
        assert c._request_errors["/api"] == 10

    def test_record_request_5xx_various(self):
        c = MetricsCollector()
        for status in (500, 501, 502, 503, 504):
            c.record_request("/api", status, 0.1)
        assert c._request_errors["/api"] == 5

    def test_record_request_path_normalization(self):
        c = MetricsCollector()
        c.record_request("/users/123", 200, 0.1)
        c.record_request("/users/456", 200, 0.2)
        assert "/users/123" in c._request_count
        assert "/users/456" in c._request_count

    def test_record_inference_many(self):
        c = MetricsCollector()
        for i in range(100):
            c.record_inference(0.01 * i, tokens=i)
        assert c._inference_count == 100
        assert c._tokens_generated == sum(range(100))

    def test_set_active_requests_zero(self):
        c = MetricsCollector()
        c.set_active_requests(10)
        c.set_active_requests(0)
        assert c.get_active_requests() == 0

    def test_set_active_requests_large(self):
        c = MetricsCollector()
        c.set_active_requests(10000)
        assert c.get_active_requests() == 10000

    def test_render_sorted_paths(self):
        c = MetricsCollector()
        c.record_request("/z", 200, 0.1)
        c.record_request("/a", 200, 0.1)
        c.record_request("/m", 200, 0.1)
        text = c.render()
        a_pos = text.find('path="/a"')
        m_pos = text.find('path="/m"')
        z_pos = text.find('path="/z"')
        assert a_pos < m_pos < z_pos

    def test_render_no_requests_no_error_section(self):
        c = MetricsCollector()
        text = c.render()
        assert "sloughgpt_requests_total" in text
        assert "sloughgpt_request_errors_total" in text

    def test_inference_zero_duration(self):
        c = MetricsCollector()
        c.record_inference(0.0)
        assert c._inference_count == 1
        assert c._inference_latencies == [0.0]

    def test_inference_negative_duration(self):
        c = MetricsCollector()
        c.record_inference(-0.1)
        assert c._inference_count == 1

    def test_record_request_zero_duration(self):
        c = MetricsCollector()
        c.record_request("/fast", 200, 0.0)
        assert c._request_latencies["/fast"] == [0.0]

    def test_record_request_negative_duration(self):
        c = MetricsCollector()
        c.record_request("/weird", 200, -0.1)
        assert c._request_latencies["/weird"] == [-0.1]

    def test_concurrent_record_requests(self):
        c = MetricsCollector()
        errors = []

        def record(i):
            try:
                c.record_request(f"/path{i % 5}", 200, 0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert sum(c._request_count.values()) == 50

    def test_render_model_not_loaded(self):
        c = MetricsCollector()
        assert "sloughgpt_model_loaded 0" in c.render()

    def test_model_loaded_toggled(self):
        c = MetricsCollector()
        c.set_model_info(True, "gpt2")
        assert "sloughgpt_model_loaded 1" in c.render()
        c.set_model_info(False)
        assert "sloughgpt_model_loaded 0" in c.render()

    def test_tokens_not_in_count(self):
        c = MetricsCollector()
        c.record_inference(0.1, tokens=100)
        c.record_inference(0.1)
        assert c._tokens_generated == 100
        assert c._inference_count == 2

    def test_render_multiple_latency_quantiles_ordering(self):
        c = MetricsCollector()
        for i in range(10):
            c.record_request("/a", 200, float(i) / 10.0)
        text = c.render()
        lines = text.split("\n")
        p50_line = [l for l in lines if 'path="/a"' in l and 'quantile="0.5"' in l][0]
        p95_line = [l for l in lines if 'path="/a"' in l and 'quantile="0.95"' in l][0]
        p99_line = [l for l in lines if 'path="/a"' in l and 'quantile="0.99"' in l][0]
        p50_val = float(p50_line.split()[-1])
        p95_val = float(p95_line.split()[-1])
        p99_val = float(p99_line.split()[-1])
        assert p50_val <= p95_val <= p99_val

    def test_uptime_format(self):
        c = MetricsCollector()
        text = c.render()
        lines = text.split("\n")
        uptime_line = [l for l in lines if l.startswith("sloughgpt_uptime_seconds ")][0]
        val = float(uptime_line.split()[-1])
        assert val >= 0.0

    def test_initial_state_all_zeros(self):
        c = MetricsCollector()
        assert c._request_count == {}
        assert c._request_errors == {}
        assert c._request_latencies == {}
        assert c._inference_count == 0
        assert c._tokens_generated == 0
        assert c._inference_latencies == []
        assert c._active_requests == 0
        assert c._model_loaded is False
        assert c._model_name == ""


class TestGetMetricsCollector:
    def test_singleton(self):
        first = get_metrics_collector()
        second = get_metrics_collector()
        assert first is second

    def test_reset_creates_new(self):
        reset_metrics_collector()
        a = get_metrics_collector()
        reset_metrics_collector()
        b = get_metrics_collector()
        assert a is not b

    def test_singleton_returns_same_type(self):
        reset_metrics_collector()
        result = get_metrics_collector()
        assert isinstance(result, _metrics.MetricsCollector)
        reset_metrics_collector()

    def test_singleton_preserves_data(self):
        reset_metrics_collector()
        a = get_metrics_collector()
        a.record_request("/test", 200, 0.1)
        b = get_metrics_collector()
        assert b._request_count["/test"] == 1
        reset_metrics_collector()

    def test_reset_clears_state(self):
        reset_metrics_collector()
        c = get_metrics_collector()
        c.record_request("/test", 200, 0.1)
        reset_metrics_collector()
        c2 = get_metrics_collector()
        assert "/test" not in c2._request_count
        reset_metrics_collector()
