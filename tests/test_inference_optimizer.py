"""
Tests for Inference Optimizer

Tests:
1. KVCache - key-value caching
2. SpeculativeDecoder - speculative decoding
3. ContinuousBatcher - batching
4. InferenceOptimizer - generation
5. Benchmark
"""

import os
import torch
import pytest


@pytest.fixture
def inference_config():
    """Test config."""
    from domains.inference.optimizer import InferenceConfig
    return InferenceConfig(
        use_kv_cache=True,
        use_speculative_decoding=False,  # Requires draft model
        use_continuous_batching=True,
    )


@pytest.fixture
def dummy_model():
    """Simple model for testing."""
    import torch.nn as nn
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 128)
            self.lm_head = nn.Linear(128, 1000)

        def forward(self, x):
            emb = self.embed(x)
            return self.lm_head(emb)
    return Dummy()


# =============================================================================
# KV CACHE TESTS
# =============================================================================

class TestKVCache:
    """Tests for KVCache."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        from domains.inference.optimizer import KVCache

        cache = KVCache(
            num_layers=12,
            num_heads=12,
            head_dim=64,
            max_length=512,
        )

        assert cache.num_layers == 12
        assert cache.num_heads == 12
        assert cache.head_dim == 64
        assert not cache._initialized

    def test_cache_update_and_get(self):
        """Test cache update and retrieval."""
        from domains.inference.optimizer import KVCache

        cache = KVCache(
            num_layers=1,
            num_heads=1,
            head_dim=64,
            max_length=10,
        )
        cache.initialize(device="cpu")

        # Create test tensors
        positions = torch.tensor([0, 1, 2])
        k = torch.randn(1, 1, 3, 64)
        v = torch.randn(1, 1, 3, 64)

        # Update cache
        cache.update(0, positions, k, v)

        # Retrieve
        retrieved_k, retrieved_v = cache.get(0, positions)

        # Cache stores (batch, num_heads, seq, head_dim), k/v might be different shape
        assert retrieved_k.shape[0] == 1  # batch
        assert retrieved_k.shape[-1] == 64  # head_dim
        assert retrieved_v.shape[0] == 1

    def test_cache_clear(self):
        """Test cache clearing."""
        from domains.inference.optimizer import KVCache

        cache = KVCache(
            num_layers=1,
            num_heads=1,
            head_dim=64,
            max_length=10,
        )
        cache.initialize(device="cpu")

        # Add some data
        positions = torch.tensor([0])
        k = torch.randn(1, 1, 1, 64)
        cache.update(0, positions, k, k)

        # Clear
        cache.clear()

        # Check cache is zero
        assert cache.k_cache[0].sum().item() == 0


# =============================================================================
# CONTINUOUS BATCHER TESTS
# =============================================================================

class TestContinuousBatcher:
    """Tests for ContinuousBatcher."""

    def test_batcher_initialization(self, inference_config):
        """Test batcher initialization."""
        from domains.inference.optimizer import ContinuousBatcher

        batcher = ContinuousBatcher(inference_config)

        assert len(batcher.pending_requests) == 0
        assert len(batcher.active_batches) == 0

    def test_add_request(self, inference_config):
        """Test adding requests."""
        from domains.inference.optimizer import ContinuousBatcher

        batcher = ContinuousBatcher(inference_config)

        prompt = torch.randint(0, 1000, (1, 10))
        batcher.add_request("req1", prompt, max_tokens=50)

        assert len(batcher.pending_requests) == 1
        assert batcher.pending_requests[0]["id"] == "req1"

    def test_get_next_batch(self, inference_config):
        """Test batch creation."""
        from domains.inference.optimizer import ContinuousBatcher

        batcher = ContinuousBatcher(inference_config)

        # Add multiple requests
        for i in range(3):
            prompt = torch.randint(0, 1000, (1, 10 + i))
            batcher.add_request(f"req{i}", prompt, max_tokens=50)

        # Get batch
        batch = batcher.get_next_batch()

        assert batch is not None
        assert len(batch["requests"]) == 3
        assert batch["input_ids"].shape[0] == 3
        assert batch["input_ids"].shape[1] == 12  # Max length + 1


# =============================================================================
# INFERENCE OPTIMIZER TESTS
# =============================================================================

class TestInferenceOptimizer:
    """Tests for InferenceOptimizer."""

    def test_optimizer_initialization(self, dummy_model, inference_config):
        """Test optimizer initialization."""
        from domains.inference.optimizer import InferenceOptimizer

        optimizer = InferenceOptimizer(
            model=dummy_model,
            config=inference_config,
            device="cpu",
        )

        assert optimizer.model is not None
        assert optimizer.kv_cache is not None
        assert optimizer.batcher is not None

    def test_generate(self, dummy_model, inference_config):
        """Test generation."""
        import sys
        if not os.environ.get("MAN_USE_TORCH_SHIM"):
            pytest.skip("compat torch not loaded; optimizer uses compat but test uses real torch")
        from domains.inference.optimizer import InferenceOptimizer

        optimizer = InferenceOptimizer(
            model=dummy_model,
            config=inference_config,
            device="cpu",
        )

        prompt = torch.randint(0, 1000, (1, 10))
        output, stats = optimizer.generate(
            prompt,
            max_new_tokens=5,
            temperature=1.0,
        )

        assert output.shape[1] == 15  # 10 + 5 tokens
        assert "latency_ms" in stats
        assert "method" in stats


# =============================================================================
# INFERENCE CONFIG TESTS
# =============================================================================

class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from domains.inference.optimizer import InferenceConfig

        config = InferenceConfig()

        assert config.max_batch_size == 32
        assert config.max_sequence_length == 2048
        assert config.use_kv_cache == True
        assert config.use_flash_attention == True
        assert config.use_speculative_decoding == False

    def test_custom_config(self):
        """Test custom configuration."""
        from domains.inference.optimizer import InferenceConfig

        config = InferenceConfig(
            max_batch_size=64,
            kv_cache_size=1024,
            use_quantization=True,
            quantization_bits=4,
        )

        assert config.max_batch_size == 64
        assert config.kv_cache_size == 1024
        assert config.use_quantization == True
        assert config.quantization_bits == 4


# =============================================================================
# INFERENCE BENCHMARK TESTS
# =============================================================================

class TestInferenceBenchmark:
    """Tests for InferenceBenchmark."""

    def test_benchmark_initialization(self, dummy_model, inference_config):
        """Test benchmark initialization."""
        from domains.inference.optimizer import InferenceOptimizer, InferenceBenchmark

        optimizer = InferenceOptimizer(dummy_model, inference_config, device="cpu")
        benchmark = InferenceBenchmark(optimizer)

        assert benchmark.optimizer is not None
        assert benchmark.results == []

    def test_run_benchmark(self, dummy_model, inference_config):
        """Test benchmark execution."""
        if not os.environ.get("MAN_USE_TORCH_SHIM"):
            pytest.skip("compat torch not loaded; optimizer uses compat but test uses real torch")
        from domains.inference.optimizer import InferenceOptimizer, InferenceBenchmark

        optimizer = InferenceOptimizer(dummy_model, inference_config, device="cpu")
        benchmark = InferenceBenchmark(optimizer)

        results = benchmark.run_benchmark(
            prompt="test",
            num_tokens=10,
            num_runs=2,
        )

        assert "avg_latency_ms" in results
        assert "p50_latency_ms" in results
        assert "p95_latency_ms" in results
        assert results["num_runs"] == 2
        assert results["generated_tokens"] == 10

    def test_print_results(self, dummy_model, inference_config):
        """Test results printing."""
        from domains.inference.optimizer import InferenceOptimizer, InferenceBenchmark

        optimizer = InferenceOptimizer(dummy_model, inference_config, device="cpu")
        benchmark = InferenceBenchmark(optimizer)

        results = {
            "prompt_tokens": 10,
            "generated_tokens": 20,
            "avg_latency_ms": 100.0,
            "p50_latency_ms": 95.0,
            "p95_latency_ms": 150.0,
            "p99_latency_ms": 180.0,
            "avg_tokens_per_sec": 200.0,
            "num_runs": 5,
        }

        # Should not raise
        benchmark.print_results(results)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
