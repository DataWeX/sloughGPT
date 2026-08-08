"""Tests for domains/infrastructure/metrics.py.

psutil is not installed in this environment; a minimal fake is injected into
``sys.modules`` for the duration of this test module (auto-restored) so the
module under test can be exercised without the real dependency.
"""

import importlib
import sys
import types

import pytest

import domains.infrastructure.metrics as _metrics
from domains.infrastructure.metrics import MetricsCollector, get_metrics_collector


@pytest.fixture(autouse=True, scope="module")
def fake_psutil():
    """Install a minimal psutil fake for this module, then restore state.

    The fake must be present before ``domains.infrastructure.metrics`` binds
    ``psutil``, so the module is reloaded while the fake is installed and
    reloaded again afterward to drop the binding. The ``sys.modules`` entry is
    removed when this module's tests finish.
    """
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


class TestGetMetricsCollector:
    def test_singleton(self):
        first = get_metrics_collector()
        second = get_metrics_collector()
        assert first is second
