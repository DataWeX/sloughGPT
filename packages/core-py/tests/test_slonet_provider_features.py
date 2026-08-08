"""
Tests for SloNetChatProvider new features.

FEATURE: slonet-provider-tests — Tests for stop sequences, logprobs, batch,
embed, metadata, seed control, tokenize/detokenize. DO NOT DELETE.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class MockTokenizer:
    """Minimal tokenizer mock for testing."""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.pad_token_id = 0

    def encode(self, text):
        # Simple hash-based encoding for deterministic results
        return [abs(hash(c)) % self.vocab_size for c in text[:20]]

    def decode(self, token_ids):
        return "".join(chr(65 + (tid % 26)) for tid in token_ids)

    def apply_chat_template(self, messages):
        if messages and isinstance(messages[-1], dict):
            return messages[-1].get("content", "")
        return ""


class MockModel:
    """Minimal model mock that yields sequential tokens."""

    def __init__(self, vocab_size=1000, max_seq_len=512):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._config = {
            "n_embd": 64,
            "n_head": 4,
            "n_layer": 2,
            "vocab_size": vocab_size,
        }
        self._counter = 0
        self._layers = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        # Mock layers[0] as embedding
        self._layers[0] = MagicMock()
        self._layers[0].weight = MagicMock()
        self._layers[0].weight.shape = [vocab_size, 64]
        self._layers[0].forward_numpy = MagicMock(return_value=np.random.randn(1, 10, 64))
        # Mock transformer blocks to return (x, cache)
        for i in range(1, 3):
            self._layers[i].forward_numpy = MagicMock(
                return_value=(np.random.randn(1, 10, 64), None)
            )
        # Mock norm layer (layers[-2]) — returns plain array, not tuple
        self._layers[-2].forward_numpy = MagicMock(return_value=np.random.randn(1, 10, 64))
        # Mock pos_emb
        self.pos_emb = MagicMock()
        self.pos_emb.num_embeddings = 512
        self.pos_emb.forward_numpy = MagicMock(return_value=np.random.randn(1, 10, 64))

    @property
    def layers(self):
        return self._layers

    def parameters(self):
        for layer in self._layers:
            if hasattr(layer, 'weight'):
                yield layer.weight

    def generate_numpy_stream(self, input_ids, max_new_tokens=50, eos_token=0,
                              temperature=1.0, top_k=None, top_p=None,
                              repetition_penalty=1.0, extra_stop_ids=None):
        """Yields sequential token IDs, stops at eos, extra stop ids, or max."""
        stop_ids = {eos_token} | set(extra_stop_ids or ())
        for i in range(max_new_tokens):
            tok = (self._counter + i) % self.vocab_size
            self._counter += 1
            if tok in stop_ids:
                break
            yield tok

    def generate_numpy(self, input_ids, max_new_tokens=50, temperature=1.0,
                       top_k=None, top_p=None, repetition_penalty=1.0,
                       eos_token=0, extra_stop_ids=None):
        """Returns all generated token IDs as array."""
        tokens = list(self.generate_numpy_stream(
            input_ids, max_new_tokens, eos_token, temperature, top_k, top_p,
            repetition_penalty, extra_stop_ids,
        ))
        if not tokens:
            return input_ids
        return np.array([tokens], dtype=np.int64)


@pytest.fixture
def mock_provider():
    """Create a SloNetChatProvider with mocked internals."""
    from domains.inference.slonet_provider import SloNetChatProvider

    provider = SloNetChatProvider.__new__(SloNetChatProvider)
    provider._hf_model_id = "test-model"
    provider._model_id = "test-model"
    provider._device = "cpu"
    provider._model = MockModel()
    provider._tokenizer = MockTokenizer()
    provider._quant_engine = None
    provider._parser = None
    return provider


class TestGenerateWithStop:
    """Test stop sequence support."""

    def test_no_stop_generates_full(self, mock_provider):
        result = mock_provider.generate_with_stop("hello", max_tokens=5)
        assert len(result) > 0

    def test_stop_string_truncates(self, mock_provider):
        # Override tokenizer to produce a predictable sequence
        mock_provider._tokenizer.decode = lambda ids: "ABCHELLOWORLD"
        result = mock_provider.generate_with_stop("hello", max_tokens=10, stop="HELLO")
        assert "HELLO" not in result

    def test_stop_list(self, mock_provider):
        mock_provider._tokenizer.decode = lambda ids: "ABCSTOPMEEND"
        result = mock_provider.generate_with_stop("hello", max_tokens=10, stop=["STOPME", "END"])
        assert "STOPME" not in result
        assert "END" not in result

    def test_stop_none_generates_full(self, mock_provider):
        result = mock_provider.generate_with_stop("hello", max_tokens=5, stop=None)
        assert len(result) > 0


class TestGenerateBatch:
    """Test batch inference."""

    def test_batch_single_prompt(self, mock_provider):
        results = mock_provider.generate_batch(["hello"], max_tokens=3)
        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_batch_multiple_prompts(self, mock_provider):
        prompts = ["hello", "world", "test"]
        results = mock_provider.generate_batch(prompts, max_tokens=3)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, str)

    def test_batch_empty_list(self, mock_provider):
        results = mock_provider.generate_batch([], max_tokens=3)
        assert results == []


class TestEmbed:
    """Test embedding extraction."""

    def test_embed_returns_array(self, mock_provider):
        emb = mock_provider.embed("hello world")
        assert isinstance(emb, np.ndarray)

    def test_embed_shape(self, mock_provider):
        emb = mock_provider.embed("test")
        assert emb.ndim == 1  # (hidden_dim,)
        assert emb.shape[0] > 0

    def test_deterministic(self, mock_provider):
        emb1 = mock_provider.embed("same text")
        emb2 = mock_provider.embed("same text")
        np.testing.assert_array_equal(emb1, emb2)


class TestMetadata:
    """Test metadata endpoint."""

    def test_metadata_returns_dict(self, mock_provider):
        meta = mock_provider.metadata()
        assert isinstance(meta, dict)

    def test_metadata_has_required_keys(self, mock_provider):
        meta = mock_provider.metadata()
        required = ["model_id", "architecture", "total_params", "vocab_size",
                     "max_seq_len", "device", "quantized"]
        for key in required:
            assert key in meta, f"Missing key: {key}"

    def test_metadata_model_id(self, mock_provider):
        meta = mock_provider.metadata()
        assert meta["model_id"] == "test-model"

    def test_metadata_architecture(self, mock_provider):
        meta = mock_provider.metadata()
        assert meta["architecture"] == "SloTransformer"


class TestTokenize:
    """Test tokenize/detokenize."""

    def test_tokenize_returns_list(self, mock_provider):
        tokens = mock_provider.tokenize("hello")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_detokenize_returns_string(self, mock_provider):
        text = mock_provider.detokenize([1, 2, 3])
        assert isinstance(text, str)

    def test_count_tokens(self, mock_provider):
        count = mock_provider.count_tokens("hello world")
        assert isinstance(count, int)
        assert count > 0


class TestSeedControl:
    """Test seed-based reproducibility."""

    def test_seed_produces_same_output(self, mock_provider):
        # Reset counter to make it deterministic
        mock_provider._model._counter = 0
        r1 = mock_provider.generate_with_stop("hi", max_tokens=3, seed=42)
        mock_provider._model._counter = 0
        r2 = mock_provider.generate_with_stop("hi", max_tokens=3, seed=42)
        assert r1 == r2

    def test_seed_does_not_crash(self, mock_provider):
        result = mock_provider.generate_with_stop("hi", max_tokens=3, seed=123)
        assert result is not None


class TestGenerateWithLogprobs:
    """Test logprobs generation."""

    def test_returns_tuple(self, mock_provider):
        result = mock_provider.generate_with_logprobs("hello", max_tokens=3)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_logprobs_structure(self, mock_provider):
        text, logprobs = mock_provider.generate_with_logprobs("hello", max_tokens=3)
        assert isinstance(text, str)
        assert isinstance(logprobs, list)
        for entry in logprobs:
            assert "token_id" in entry
            assert "token" in entry
            assert "logprob" in entry
            assert "position" in entry

    def test_logprobs_positions_sequential(self, mock_provider):
        _, logprobs = mock_provider.generate_with_logprobs("hello", max_tokens=5)
        for i, entry in enumerate(logprobs):
            assert entry["position"] == i
