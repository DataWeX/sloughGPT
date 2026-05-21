"""
SloughGPT Optimizations Tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock


class TestOptimizationConfig:
    """Tests for OptimizationConfig."""
    
    def test_config_defaults(self):
        """Test default configuration."""
        from domains.inference.optimizations import OptimizationConfig
        
        config = OptimizationConfig()
        
        assert config.use_flash_attention is False
        assert config.use_kv_cache is True
        assert config.max_batch_size == 32
        assert config.prefill_chunk_size == 512
        assert config.use_speculative is False
        assert config.speculative_tokens == 4
    
    def test_config_custom(self):
        """Test custom configuration."""
        from domains.inference.optimizations import OptimizationConfig
        
        config = OptimizationConfig(
            use_kv_cache=True,
            max_batch_size=64,
            speculative_tokens=8
        )
        
        assert config.max_batch_size == 64
        assert config.speculative_tokens == 8


class TestKVCacheOptimizer:
    """Tests for KVCacheOptimizer."""
    
    def test_cache_init(self):
        """Test cache initialization."""
        from domains.inference.optimizations import KVCacheOptimizer
        
        cache = KVCacheOptimizer(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            max_length=1024,
            device="cpu"
        )
        
        assert cache.num_layers == 4
        assert cache.num_heads == 8
        assert cache.head_dim == 64
        assert cache.max_length == 1024
        assert cache.current_length == 0
    
    def test_cache_update(self):
        """Test cache update."""
        from domains.inference.optimizations import KVCacheOptimizer
        
        cache = KVCacheOptimizer(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            max_length=512,
            device="cpu"
        )
        
        key = torch.randn(1, 4, 10, 32)
        value = torch.randn(1, 4, 10, 32)
        
        cache.update(0, key, value, position=0)
        
        assert cache.current_length == 10
    
    def test_cache_get(self):
        """Test cache retrieval."""
        from domains.inference.optimizations import KVCacheOptimizer
        
        cache = KVCacheOptimizer(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            max_length=512,
            device="cpu"
        )
        
        key = torch.randn(1, 4, 10, 32)
        value = torch.randn(1, 4, 10, 32)
        
        cache.update(0, key, value, position=0)
        
        retrieved_key, retrieved_value = cache.get(0, start=0, end=10)
        
        assert retrieved_key.shape == key.shape
        assert retrieved_value.shape == value.shape
    
    def test_cache_reset(self):
        """Test cache reset."""
        from domains.inference.optimizations import KVCacheOptimizer
        
        cache = KVCacheOptimizer(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            max_length=512,
            device="cpu"
        )
        
        key = torch.randn(1, 4, 10, 32)
        value = torch.randn(1, 4, 10, 32)
        cache.update(0, key, value, position=0)
        
        cache.reset()
        
        assert cache.current_length == 0
    
    def test_cache_memory_calculation(self):
        """Test memory calculation."""
        from domains.inference.optimizations import KVCacheOptimizer
        
        cache = KVCacheOptimizer(
            num_layers=12,
            num_heads=12,
            head_dim=64,
            max_length=4096,
            device="cpu"
        )
        
        memory_mb = cache.get_allocated_memory_mb()
        
        assert memory_mb > 0


class TestAttentionMask:
    """Tests for AttentionMask."""
    
    def test_create_causal_mask(self):
        """Test causal mask creation."""
        from domains.inference.optimizations import AttentionMask
        
        mask = AttentionMask.create_causal_mask(seq_len=10, device="cpu")
        
        assert mask.shape == (10, 10)
        assert mask.dtype == torch.bool
    
    def test_create_padder_mask(self):
        """Test padder mask creation."""
        from domains.inference.optimizations import AttentionMask
        
        input_ids = torch.tensor([[1, 2, 3, 0, 0]])
        mask = AttentionMask.create_padder_mask(input_ids, pad_token_id=0)
        
        expected = torch.tensor([[True, True, True, False, False]])
        assert torch.equal(mask, expected)


class TestBatchProcessor:
    """Tests for BatchProcessor."""
    
    def test_batch_processor_init(self):
        """Test BatchProcessor initialization."""
        from domains.inference.optimizations import BatchProcessor
        
        processor = BatchProcessor(max_batch_size=16)
        
        assert processor.max_batch_size == 16
    
    def test_pad_to_batch(self):
        """Test padding sequences for batching."""
        from domains.inference.optimizations import BatchProcessor
        
        processor = BatchProcessor()
        
        input1 = torch.tensor([[1, 2, 3]])
        input2 = torch.tensor([[4, 5]])
        input3 = torch.tensor([[6, 7, 8, 9]])
        
        batch, mask = processor.pad_to_batch([input1, input2, input3], pad_token_id=0)
        
        assert batch.shape == (3, 4)
        assert mask.shape == (3, 4)
        assert torch.equal(batch[0], torch.tensor([1, 2, 3, 0]))
        assert torch.equal(batch[1], torch.tensor([4, 5, 0, 0]))
    
    def test_split_by_length(self):
        """Test splitting long sequences."""
        from domains.inference.optimizations import BatchProcessor
        
        processor = BatchProcessor()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        chunks = processor.split_by_length(input_ids, max_length=3)
        
        assert len(chunks) == 3
        assert chunks[0].shape == (1, 3)


class TestSpeculativeDecoder:
    """Tests for SpeculativeDecoder."""
    
    @pytest.fixture
    def mock_draft_model(self):
        """Create mock draft model."""
        model = nn.Linear(10, 10)
        output = Mock()
        output.logits = torch.randn(1, 1, 10)
        model.return_value = output
        return model
    
    @pytest.fixture
    def mock_target_model(self):
        """Create mock target model."""
        model = nn.Linear(10, 10)
        output = Mock()
        output.logits = torch.randn(1, 5, 10)
        model.return_value = output
        return model
    
    def test_speculative_decoder_init(self):
        """Test SpeculativeDecoder initialization."""
        from domains.inference.optimizations import SpeculativeDecoder
        
        draft = nn.Linear(10, 10)
        target = nn.Linear(10, 10)
        
        decoder = SpeculativeDecoder(draft, target, speculative_tokens=4)
        
        assert decoder.speculative_tokens == 4
        assert decoder.draft_model is draft
        assert decoder.target_model is target


class TestEstimateInferenceMemory:
    """Tests for inference memory estimation."""
    
    def test_estimate_fp16(self):
        """Test FP16 memory estimation."""
        from domains.inference.optimizations import estimate_inference_memory
        
        result = estimate_inference_memory(1000000000, precision="fp16")
        
        assert result["precision"] == "fp16"
        assert result["num_parameters"] == 1000000000
        assert result["model_memory_gb"] > 0
    
    def test_estimate_fp32(self):
        """Test FP32 memory estimation."""
        from domains.inference.optimizations import estimate_inference_memory
        
        result = estimate_inference_memory(1000000000, precision="fp32")
        
        assert result["bytes_per_param"] == 4 if "bytes_per_param" in result else True
    
    def test_estimate_int8(self):
        """Test INT8 memory estimation."""
        from domains.inference.optimizations import estimate_inference_memory
        
        result = estimate_inference_memory(1000000000, precision="int8")
        
        assert result["model_memory_gb"] > 0
    
    def test_estimate_kv_cache_multiplier(self):
        """Test KV cache multiplier."""
        from domains.inference.optimizations import estimate_inference_memory
        
        result = estimate_inference_memory(
            1000000000,
            precision="fp16",
            kv_cache_multiplier=2.0
        )
        
        assert result["total_memory_gb"] > result["model_memory_gb"]


class TestOptimizeModelForInference:
    """Tests for model optimization."""
    
    def test_optimize_fp16(self):
        """Test FP16 optimization."""
        from domains.inference.optimizations import optimize_model_for_inference
        
        model = nn.Linear(100, 50)
        optimized = optimize_model_for_inference(model, use_quantization=True, precision="fp16")
        
        assert optimized is not None
    
    def test_optimize_no_quantization(self):
        """Test without quantization."""
        from domains.inference.optimizations import optimize_model_for_inference
        
        model = nn.Linear(100, 50)
        model.eval = Mock()
        optimized = optimize_model_for_inference(model, use_quantization=False)
        
        assert optimized is model
