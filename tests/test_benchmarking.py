"""
SloughGPT Benchmarking Tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""
    
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult creation."""
        from domains.ml_infrastructure.benchmarking import BenchmarkResult
        
        result = BenchmarkResult(
            model_name="test-model",
            num_parameters=1000000,
            memory_mb=50.0,
            inference_time_ms=100.0,
            throughput_tokens_per_sec=50.0
        )
        
        assert result.model_name == "test-model"
        assert result.num_parameters == 1000000
        assert result.memory_mb == 50.0
    
    def test_benchmark_result_to_dict(self):
        """Test BenchmarkResult to_dict."""
        from domains.ml_infrastructure.benchmarking import BenchmarkResult
        
        result = BenchmarkResult(
            model_name="test-model",
            num_parameters=1000000,
            memory_mb=50.0,
            inference_time_ms=100.0,
            throughput_tokens_per_sec=50.0,
            latency_p50_ms=90.0,
            latency_p95_ms=120.0
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["model_name"] == "test-model"
        assert result_dict["memory_mb"] == 50.0
        assert result_dict["latency_p50_ms"] == 90.0


class TestBenchmarker:
    """Tests for Benchmarker class."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        model = nn.Linear(100, 100)
        model.name = "test-model"
        return model
    
    @pytest.fixture
    def simple_tokenizer(self):
        """Create simple tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
        tokenizer.decode = Mock(return_value="test")
        tokenizer.pad_token_id = 0
        return tokenizer
    
    def test_benchmarker_init(self, simple_model):
        """Test Benchmarker initialization."""
        from domains.ml_infrastructure.benchmarking import Benchmarker
        
        benchmarker = Benchmarker(simple_model, device="cpu")
        
        assert benchmarker.device == "cpu"
        assert benchmarker.warmup_steps == 3
    
    def test_count_parameters(self, simple_model):
        """Test parameter counting."""
        from domains.ml_infrastructure.benchmarking import Benchmarker
        
        benchmarker = Benchmarker(simple_model, device="cpu")
        count = benchmarker.count_parameters()
        
        assert count > 0
    
    def test_measure_memory(self, simple_model):
        """Test memory measurement."""
        from domains.ml_infrastructure.benchmarking import Benchmarker
        
        benchmarker = Benchmarker(simple_model, device="cpu")
        memory = benchmarker.measure_memory()
        
        assert memory > 0
    
    def test_benchmark_inference_no_tokenizer(self, simple_model):
        """Test inference benchmark without tokenizer."""
        from domains.ml_infrastructure.benchmarking import Benchmarker
        
        benchmarker = Benchmarker(simple_model, device="cpu")
        
        result = benchmarker.benchmark_inference(
            prompt="test",
            max_new_tokens=5,
            num_runs=2
        )
        
        assert result.num_parameters > 0
        assert result.memory_mb > 0
    
    def test_benchmark_batch_no_tokenizer(self, simple_model):
        """Test batch benchmark without tokenizer."""
        from domains.ml_infrastructure.benchmarking import Benchmarker
        
        benchmarker = Benchmarker(simple_model, device="cpu")
        
        result = benchmarker.benchmark_batch(
            prompts=["a", "b"],
            batch_size=2,
            max_new_tokens=2
        )
        
        assert "error" in result
    
    def test_calculate_perplexity_no_tokenizer(self, simple_model):
        """Test perplexity without tokenizer."""
        from domains.ml_infrastructure.benchmarking import Benchmarker
        
        benchmarker = Benchmarker(simple_model, device="cpu")
        
        ppl = benchmarker.calculate_perplexity("test text")
        
        assert ppl is None


class TestBenchmarkFunctions:
    """Tests for benchmark utility functions."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        model = nn.Linear(50, 50)
        model.name = "test"
        return model
    
    def test_benchmark_model_function(self, simple_model):
        """Test benchmark_model function."""
        from domains.ml_infrastructure.benchmarking import benchmark_model
        
        result = benchmark_model(simple_model, device="cpu")
        
        assert result.model_name == "test"
        assert result.num_parameters > 0
    
    def test_compare_models(self, simple_model):
        """Test compare_models function."""
        from domains.ml_infrastructure.benchmarking import compare_models
        
        model1 = nn.Linear(50, 50)
        model1.name = "model1"
        model2 = nn.Linear(50, 50)
        model2.name = "model2"
        
        results = compare_models(
            {"model1": model1, "model2": model2},
            device="cpu"
        )
        
        assert len(results) == 2
        assert results[0].model_name == "model1"
        assert results[1].model_name == "model2"


class TestBenchmarkImports:
    """Tests for module imports."""
    
    def test_import_all_exports(self):
        """Test that all exports are importable."""
        from domains.ml_infrastructure.benchmarking import (
            BenchmarkResult,
            Benchmarker,
            benchmark_model,
            compare_models,
        )
        
        assert BenchmarkResult is not None
        assert Benchmarker is not None
        assert callable(benchmark_model)
        assert callable(compare_models)
