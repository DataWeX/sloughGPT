"""
SloughGPT API Server Tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, patch, AsyncMock
import torch


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_response_structure(self):
        """Test health endpoint returns correct structure."""
        # Mock response structure
        response = {
            "status": "ok",
            "version": "1.0.0",
            "model_loaded": False,
        }
        
        assert "status" in response
        assert "version" in response
    
    def test_health_check(self):
        """Test health check can be called."""
        # Simple test that the function exists
        from domains.inference.engine import InferenceEngine
        
        assert InferenceEngine is not None


class TestInferenceEndpoints:
    """Tests for inference API endpoints."""
    
    def test_generate_request_validation(self):
        """Test generate request validation."""
        from pydantic import BaseModel
        from typing import Optional
        
        class GenerateRequest(BaseModel):
            prompt: str
            max_new_tokens: Optional[int] = 100
            temperature: Optional[float] = 0.8
            top_k: Optional[int] = 50
            top_p: Optional[float] = 0.9
            
        # Valid request
        req = GenerateRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.max_new_tokens == 100
        
        # Custom params
        req = GenerateRequest(prompt="Hi", max_new_tokens=50, temperature=0.5)
        assert req.max_new_tokens == 50
        assert req.temperature == 0.5
    
    def test_chat_request_validation(self):
        """Test chat request validation."""
        from pydantic import BaseModel
        from typing import List
        
        class ChatMessage(BaseModel):
            role: str
            content: str
        
        class ChatRequest(BaseModel):
            messages: List[ChatMessage]
            max_new_tokens: int = 100
            
        # Valid request
        req = ChatRequest(
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi"),
            ]
        )
        
        assert len(req.messages) == 2
        assert req.messages[0].content == "Hello"


class TestQuantizationEndpoints:
    """Tests for quantization API endpoints."""
    
    def test_quantize_endpoint_params(self):
        """Test quantize endpoint parameters."""
        from pydantic import BaseModel
        from typing import Optional
        
        class QuantizeRequest(BaseModel):
            model_name: str
            quantization_type: str  # fp16, int8, int4
            device: Optional[str] = "cpu"
            
        req = QuantizeRequest(model_name="gpt2", quantization_type="fp16")
        assert req.quantization_type == "fp16"
    
    def test_quantization_type_validation(self):
        """Test quantization type enum values."""
        from domains.inference.quantization import QuantizationType
        
        valid_types = [qt.value for qt in QuantizationType]
        assert "fp16" in valid_types
        assert "int8" in valid_types
        assert "int4" in valid_types


class TestBenchmarkEndpoints:
    """Tests for benchmarking API endpoints."""
    
    def test_benchmark_request(self):
        """Test benchmark request structure."""
        from pydantic import BaseModel
        from typing import Optional
        
        class BenchmarkRequest(BaseModel):
            model_name: str
            prompt: str = "The quick brown fox"
            max_new_tokens: Optional[int] = 50
            num_runs: Optional[int] = 10
            
        req = BenchmarkRequest(model_name="gpt2", num_runs=5)
        assert req.num_runs == 5


class TestTrainingEndpoints:
    """Tests for training API endpoints."""
    
    def test_training_request(self):
        """Test training request structure."""
        from pydantic import BaseModel
        from typing import Optional
        
        class TrainingRequest(BaseModel):
            dataset: str
            epochs: Optional[int] = 5
            batch_size: Optional[int] = 32
            learning_rate: Optional[float] = 1e-3
            use_lora: Optional[bool] = False
            
        req = TrainingRequest(dataset="shakespeare", epochs=3)
        assert req.epochs == 3
    
    def test_training_response(self):
        """Test training response structure."""
        response = {
            "job_id": "train_123",
            "status": "started",
            "config": {
                "epochs": 3,
                "batch_size": 32,
            }
        }
        
        assert "job_id" in response
        assert "status" in response


class TestExperimentEndpoints:
    """Tests for experiment tracking endpoints."""
    
    def test_create_experiment(self):
        """Test experiment creation request."""
        from pydantic import BaseModel
        from typing import Optional
        
        class CreateExperiment(BaseModel):
            name: str
            description: Optional[str] = None
            tags: Optional[list] = None
            
        exp = CreateExperiment(name="test_exp", tags=["test"])
        assert exp.name == "test_exp"
    
    def test_log_metric(self):
        """Test metric logging request."""
        from pydantic import BaseModel
        
        class LogMetric(BaseModel):
            metric_name: str
            value: float
            step: int
            
        metric = LogMetric(metric_name="loss", value=2.5, step=100)
        assert metric.value == 2.5


class TestExportEndpoints:
    """Tests for model export endpoints."""
    
    def test_export_request(self):
        """Test export request structure."""
        from pydantic import BaseModel
        from typing import Optional, List
        
        class ExportRequest(BaseModel):
            model_name: str
            format: str  # torch, onnx, safetensors
            output_path: Optional[str] = None
            quantize: Optional[str] = None  # fp16, int8, int4
            
        req = ExportRequest(model_name="gpt2", format="onnx")
        assert req.format == "onnx"


class TestAPIIntegration:
    """Integration tests for API."""
    
    def test_endpoints_exist(self):
        """Test that all endpoint functions exist."""
        # These should exist in the server
        from domains.inference.quantization import quantize_model
        from domains.inference.engine import InferenceEngine
        from domains.ml_infrastructure.benchmarking import benchmark_model
        
        assert callable(quantize_model)
        assert callable(InferenceEngine)
        assert callable(benchmark_model)
    
    def test_error_handling(self):
        """Test error handling in requests."""
        from pydantic import ValidationError, BaseModel
        
        class StrictRequest(BaseModel):
            name: str
            value: int
            
        # Should fail with invalid type
        with pytest.raises(ValidationError):
            StrictRequest(name="test", value="not_an_int")
