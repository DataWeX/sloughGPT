"""
SloughGPT Inference Engine Tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from unittest.mock import Mock, MagicMock


class TestKVCache:
    """Tests for KVCache."""
    
    def test_kv_cache_init(self):
        """Test KVCache initialization."""
        from domains.inference.engine import KVCache
        
        cache = KVCache(num_layers=4, dtype=torch.float32)
        assert cache.num_layers == 4
        assert cache.dtype == torch.float32
        assert cache.max_length == 0
    
    def test_kv_cache_update(self):
        """Test updating cache."""
        from domains.inference.engine import KVCache
        
        cache = KVCache(num_layers=2, dtype=torch.float32)
        key = torch.randn(1, 2, 5, 8)
        value = torch.randn(1, 2, 5, 8)
        
        cache.update(0, key, value)
        assert cache.max_length == 5
        
        retrieved_key, retrieved_value = cache.get(0)
        assert retrieved_key.shape == key.shape
    
    def test_kv_cache_reset(self):
        """Test cache reset."""
        from domains.inference.engine import KVCache
        
        cache = KVCache(num_layers=2)
        cache.key_cache[0] = torch.randn(1, 2, 10, 8)
        cache.max_length = 10
        
        cache.reset()
        assert cache.max_length == 0
        assert cache.key_cache[0] is None


class TestGenerationRequest:
    """Tests for GenerationRequest."""
    
    def test_request_creation(self):
        """Test request creation."""
        from domains.inference.engine import GenerationRequest
        
        request = GenerationRequest(
            id="test-1",
            prompt="Hello world",
            max_new_tokens=100
        )
        
        assert request.id == "test-1"
        assert request.prompt == "Hello world"
        assert request.max_new_tokens == 100
        assert request.finished is False
        assert request.error is None


class TestInferenceEngine:
    """Tests for InferenceEngine."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=torch.tensor([[1, 2, 3]]))
        tokenizer.decode = Mock(return_value="Hello world")
        tokenizer.eos_token_id = 0
        tokenizer.pad_token = Mock()
        tokenizer.pad_token_id = 0
        return tokenizer
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.eval = Mock()
        model.to = Mock()
        model.config = Mock()
        model.config.num_hidden_layers = 4
        return model
    
    def test_engine_init(self, mock_model, mock_tokenizer):
        """Test engine initialization."""
        from domains.inference.engine import InferenceEngine
        
        engine = InferenceEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu"
        )
        
        assert engine.device == torch.device("cpu")
        assert engine.max_batch_size == 32
    
    def test_encode(self, mock_model, mock_tokenizer):
        """Test text encoding."""
        from domains.inference.engine import InferenceEngine
        
        engine = InferenceEngine(mock_model, mock_tokenizer, device="cpu")
        tokens = engine.encode("test")
        
        mock_tokenizer.encode.assert_called_once_with("test", return_tensors="pt")
    
    def test_decode(self, mock_model, mock_tokenizer):
        """Test token decoding."""
        from domains.inference.engine import InferenceEngine
        
        engine = InferenceEngine(mock_model, mock_tokenizer, device="cpu")
        text = engine.decode([1, 2, 3])
        
        mock_tokenizer.decode.assert_called_once()
    
    def test_sample_token_greedy(self, mock_model, mock_tokenizer):
        """Test greedy token sampling."""
        from domains.inference.engine import InferenceEngine
        
        engine = InferenceEngine(mock_model, mock_tokenizer, device="cpu")
        
        logits = torch.randn(10)
        token = engine._sample_token(logits, temperature=0.0, top_k=0, top_p=0.0)
        
        expected = logits.argmax().item()
        assert token == expected
    
    def test_sample_token_with_temperature(self, mock_model, mock_tokenizer):
        """Test sampling with temperature."""
        from domains.inference.engine import InferenceEngine
        
        engine = InferenceEngine(mock_model, mock_tokenizer, device="cpu")
        
        logits = torch.tensor([1.0, 2.0, 3.0])
        token = engine._sample_token(logits, temperature=1.0, top_k=0, top_p=0.0)
        
        assert 0 <= token < 3
    
    def test_sample_token_with_top_k(self, mock_model, mock_tokenizer):
        """Test top-k sampling."""
        from domains.inference.engine import InferenceEngine
        
        engine = InferenceEngine(mock_model, mock_tokenizer, device="cpu")
        
        logits = torch.tensor([1.0, 5.0, 3.0, 7.0, 2.0])
        token = engine._sample_token(logits, temperature=1.0, top_k=2, top_p=1.0)
        
        assert token in [1, 3]
    
    def test_apply_repetition_penalty_positive(self, mock_model, mock_tokenizer):
        """Test repetition penalty with positive logits."""
        from domains.inference.engine import InferenceEngine
        
        engine = InferenceEngine(mock_model, mock_tokenizer, device="cpu")
        
        logits = torch.ones(10)
        prev_tokens = torch.tensor([0])
        
        penalized = engine._apply_repetition_penalty(logits, prev_tokens, penalty=2.0)
        
        assert penalized[0] == 2.0
    
    def test_apply_repetition_penalty_negative(self, mock_model, mock_tokenizer):
        """Test repetition penalty with negative logits."""
        from domains.inference.engine import InferenceEngine
        
        engine = InferenceEngine(mock_model, mock_tokenizer, device="cpu")
        
        logits = torch.tensor([-1.0, 1.0, 2.0])
        prev_tokens = torch.tensor([0])
        
        penalized = engine._apply_repetition_penalty(logits, prev_tokens, penalty=2.0)
        
        assert penalized[0] == -0.5
    
    def test_apply_repetition_penalty_no_op(self, mock_model, mock_tokenizer):
        """Test repetition penalty no-op when penalty is 1.0."""
        from domains.inference.engine import InferenceEngine
        
        engine = InferenceEngine(mock_model, mock_tokenizer, device="cpu")
        
        logits = torch.ones(10)
        prev_tokens = torch.tensor([0])
        
        penalized = engine._apply_repetition_penalty(logits, prev_tokens, penalty=1.0)
        
        assert torch.equal(penalized, logits)
    
    def test_get_stats(self, mock_model, mock_tokenizer):
        """Test statistics retrieval."""
        from domains.inference.engine import InferenceEngine
        
        engine = InferenceEngine(mock_model, mock_tokenizer, device="cpu")
        stats = engine.get_stats()
        
        assert "requests_processed" in stats
        assert "tokens_generated" in stats
        assert "total_time" in stats
    
    def test_reset_stats(self, mock_model, mock_tokenizer):
        """Test statistics reset."""
        from domains.inference.engine import InferenceEngine
        
        engine = InferenceEngine(mock_model, mock_tokenizer, device="cpu")
        engine._stats["requests_processed"] = 10
        
        engine.reset_stats()
        
        assert engine._stats["requests_processed"] == 0


class TestInferenceEngineGeneration:
    """Tests for text generation."""
    
    @pytest.fixture
    def mock_model_with_output(self):
        """Create mock model that returns proper output."""
        model = Mock()
        model.eval = Mock()
        model.to = Mock()
        model.config = Mock()
        model.config.num_hidden_layers = 2
        
        output = Mock()
        output.logits = torch.randn(1, 10, 50257)
        model.return_value = output
        return model
    
    @pytest.fixture
    def simple_tokenizer(self):
        """Create simple tokenizer for testing."""
        tokenizer = Mock()
        vocab = {chr(i + 97): i for i in range(26)}
        vocab['<eos>'] = 0
        rev_vocab = {v: k for k, v in vocab.items()}
        
        def encode(text, return_tensors=None):
            return torch.tensor([[vocab.get(c, 1) for c in text]])
        tokenizer.encode = encode
        
        def decode(ids, skip_special_tokens=True):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return ''.join([rev_vocab.get(i, '') for i in ids])
        tokenizer.decode = decode
        
        tokenizer.eos_token_id = 0
        tokenizer.pad_token_id = 0
        
        return tokenizer
    
    def test_generate_single_basic(self, simple_tokenizer):
        """Test basic single prompt generation."""
        from domains.inference.engine import InferenceEngine
        
        model = Mock()
        model.eval = Mock()
        model.to = Mock()
        model.config = Mock()
        model.config.num_hidden_layers = 2
        
        logits = torch.zeros(1, 5, 26)
        logits[0, -1, 1] = 10.0
        
        output = Mock()
        output.logits = logits
        model.return_value = output
        
        engine = InferenceEngine(model, simple_tokenizer, device="cpu", use_cache=False)
        result = engine.generate_single("abc", max_new_tokens=3)
        
        assert isinstance(result, str)


class TestCreateEngine:
    """Tests for create_engine factory function."""
    
    def test_create_engine_import(self):
        """Test that create_engine is importable."""
        from domains.inference.engine import create_engine
        
        assert callable(create_engine)
