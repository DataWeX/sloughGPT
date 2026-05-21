"""
Tests for llama.cpp inference engine features.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time


class TestGPUDetection:
    """Tests for GPU detection."""

    def test_detect_gpu_returns_info(self):
        """Test that detect_gpu returns GPUInfo or None."""
        from domains.inference.llama_engine import detect_gpu

        result = detect_gpu()
        assert result is None or hasattr(result, "name")
        assert result is None or hasattr(result, "backend")
        assert result is None or hasattr(result, "recommended")

    def test_auto_select_backend_returns_int(self):
        """Test that auto_select_backend returns integer."""
        from domains.inference.llama_engine import auto_select_backend

        result = auto_select_backend()
        assert isinstance(result, int)
        assert result >= 0


class TestInferenceProfiler:
    """Tests for InferenceProfiler."""

    def test_profiler_start_end(self):
        """Test profiler start and end."""
        from domains.inference.llama_engine import InferenceProfiler

        p = InferenceProfiler()
        assert p.start("test").__class__.__name__ == "InferenceProfiler"

        elapsed = p.end()
        assert isinstance(elapsed, float)
        assert elapsed >= 0

    def test_profiler_get_profiles(self):
        """Test getting profile results."""
        from domains.inference.llama_engine import InferenceProfiler

        p = InferenceProfiler()
        p.start("op1")
        time.sleep(0.001)
        p.end()

        profiles = p.get_profiles()
        assert len(profiles) == 1
        assert profiles[0]["name"] == "op1"
        assert "elapsed_ms" in profiles[0]

    def test_profiler_summary(self):
        """Test profiler summary."""
        from domains.inference.llama_engine import InferenceProfiler

        p = InferenceProfiler()
        for i in range(3):
            p.start(f"op{i}")
            time.sleep(0.001)
            p.end()

        summary = p.summary()
        assert summary["total_profiles"] == 3
        assert summary["total_ms"] > 0
        assert summary["avg_ms"] > 0
        assert summary["min_ms"] <= summary["max_ms"]

    def test_profiler_clear(self):
        """Test clearing profiles."""
        from domains.inference.llama_engine import InferenceProfiler

        p = InferenceProfiler()
        p.start("test")
        p.end()

        p.clear()
        assert len(p.get_profiles()) == 0
        assert p.summary() == {}


class TestMemoryTracking:
    """Tests for memory tracking."""

    def test_get_memory_usage(self):
        """Test memory usage retrieval."""
        from domains.inference.llama_engine import get_memory_usage

        mem = get_memory_usage()

        if "error" in mem:
            pytest.skip("psutil not installed")

        assert "rss_mb" in mem
        assert "vms_mb" in mem
        assert mem["rss_mb"] > 0


class TestModelCaching:
    """Tests for model caching functionality."""

    def test_get_inference_stats(self):
        """Test getting inference stats."""
        from domains.inference.llama_engine import get_inference_stats

        stats = get_inference_stats()
        assert "total_requests" in stats
        assert "total_tokens" in stats
        assert "total_time" in stats
        assert "avg_tokens_per_second" in stats
        assert "cached_models" in stats


class TestLlamaInferenceConfig:
    """Tests for LlamaInferenceConfig."""

    def test_config_init(self):
        """Test config initialization."""
        from domains.inference.llama_engine import LlamaInferenceConfig

        config = LlamaInferenceConfig(model_path="/tmp/test.gguf")
        assert config.model_path == "/tmp/test.gguf"
        assert config.n_ctx == 4096
        assert config.n_threads == 6
        assert config.n_gpu_layers >= 0  # auto-detected (may be 0 or positive)

    def test_config_custom_values(self):
        """Test custom config values."""
        from domains.inference.llama_engine import LlamaInferenceConfig

        config = LlamaInferenceConfig(
            model_path="/tmp/test.gguf",
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
        )
        assert config.n_ctx == 2048
        assert config.n_threads == 4
        assert config.n_gpu_layers == 0


class TestFindGGUFModels:
    """Tests for finding GGUF models."""

    def test_find_gguf_models_returns_list(self):
        """Test that find_gguf_models returns a list."""
        from domains.inference.llama_engine import find_gguf_models

        models = find_gguf_models()
        assert isinstance(models, list)


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_gpu_info_creation(self):
        """Test GPUInfo creation."""
        from domains.inference.llama_engine import GPUInfo

        info = GPUInfo(
            name="Test GPU",
            backend="cuda",
            vram_mb=8192,
            has_tensor_ops=True,
            recommended=True,
            reason="Test GPU",
        )

        assert info.name == "Test GPU"
        assert info.backend == "cuda"
        assert info.vram_mb == 8192
        assert info.has_tensor_ops is True
        assert info.recommended is True
        assert info.reason == "Test GPU"


class TestBatchGenerate:
    """Tests for batch generation."""

    def test_batch_generate_is_method(self):
        """Test that batch_generate exists on LlamaInferenceEngine."""
        from domains.inference.llama_engine import LlamaInferenceEngine, LlamaInferenceConfig

        assert hasattr(LlamaInferenceEngine, "batch_generate")
        assert hasattr(LlamaInferenceEngine, "warmup")
        assert hasattr(LlamaInferenceEngine, "batch_stream_generate")


class TestLatencyHistogram:
    """Tests for latency histogram."""

    def test_get_latency_histogram(self):
        """Test latency histogram function."""
        from domains.inference.llama_engine import get_latency_histogram

        histogram = get_latency_histogram()
        assert "count" in histogram
        assert "p50" in histogram
        assert "p90" in histogram
        assert "p99" in histogram


class TestResetStats:
    """Tests for reset functions."""

    def test_reset_inference_stats(self):
        """Test reset inference stats function."""
        from domains.inference.llama_engine import reset_inference_stats, get_inference_stats

        reset_inference_stats()
        stats = get_inference_stats()
        assert stats["total_requests"] == 0
        assert stats["total_tokens"] == 0
        assert stats["total_errors"] == 0


class TestMetricsSummary:
    """Tests for metrics summary."""

    def test_get_metrics_summary(self):
        """Test get_metrics_summary function."""
        from domains.inference.llama_engine import get_metrics_summary

        summary = get_metrics_summary()
        assert "requests" in summary
        assert "performance" in summary
        assert "memory" in summary
        assert "last_error" in summary
        assert "total" in summary["requests"]
        assert "latency_p50_ms" in summary["performance"]


class TestModelInfo:
    """Tests for model info."""

    def test_get_model_info_missing(self):
        """Test get_model_info with missing file."""
        from domains.inference.llama_engine import get_model_info

        info = get_model_info("/nonexistent/model.gguf")
        assert "error" in info

    def test_get_model_info_exists(self):
        """Test get_model_info with existing file."""
        from domains.inference.llama_engine import get_model_info
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"test model data" * 1000)
            temp_path = f.name

        info = get_model_info(temp_path)
        os.unlink(temp_path)

        assert "error" not in info
        assert info["exists"] is True
        assert info["size_bytes"] > 0
        assert "size_mb" in info
        assert "size_gb" in info


class TestCacheManagement:
    """Tests for cache management."""

    def test_list_cached_models(self):
        """Test list_cached_models function."""
        from domains.inference.llama_engine import list_cached_models

        models = list_cached_models()
        assert isinstance(models, list)

    def test_preload_models(self):
        """Test preload_models function."""
        from domains.inference.llama_engine import preload_models

        results = preload_models([])
        assert isinstance(results, dict)

    def test_is_model_cached(self):
        """Test is_model_cached function."""
        from domains.inference.llama_engine import is_model_cached

        result = is_model_cached("/nonexistent.gguf")
        assert result is False

    def test_unload_model(self):
        """Test unload_model function."""
        from domains.inference.llama_engine import unload_model

        result = unload_model("/nonexistent.gguf")
        assert result is False

    def test_get_cache_stats(self):
        """Test get_cache_stats function."""
        from domains.inference.llama_engine import get_cache_stats

        stats = get_cache_stats()
        assert "total_cached" in stats
        assert "loaded" in stats
        assert "unloaded" in stats
        assert "models" in stats
        assert "max_size" in stats

    def test_set_max_cache_size(self):
        """Test set_max_cache_size function."""
        from domains.inference.llama_engine import set_max_cache_size, get_cache_stats

        set_max_cache_size(5)
        stats = get_cache_stats()
        assert stats["max_size"] == 5

        set_max_cache_size(10)
        stats = get_cache_stats()
        assert stats["max_size"] == 10


class TestInferenceHealth:
    """Tests for inference health check."""

    def test_get_inference_health(self):
        """Test get_inference_health function."""
        from domains.inference.llama_engine import get_inference_health

        health = get_inference_health()
        assert "status" in health
        assert "gpu_available" in health
        assert "cached_models" in health
        assert "total_requests" in health
        assert health["status"] == "healthy"
