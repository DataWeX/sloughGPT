"""
Tests for ONNX backend integration with ModelServer.

Tests:
1. ONNXBackend.generate() with mock session
2. ONNXBackend.generate_stream() with mock session
3. ONNXBackend.alive property
4. ModelServer selects ONNXBackend when available
5. ModelServer falls back to LocalBackend when ONNX unavailable
"""

import numpy as np
import pytest
from domains.infrastructure.model_server import ONNXBackend, ModelServer, LocalBackend


class MockTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, prompt, return_tensors="np", **kwargs):
        return {"input_ids": np.array([[1, 2, 3]]), "attention_mask": np.array([[1, 1, 1]])}

    def decode(self, tokens, skip_special_tokens=True):
        return "hello world"

    def encode(self, text):
        return [1, 2, 3]


class MockModel:
    device = "cpu"
    def generate(self, **kwargs):
        return [[1, 2, 3]]


class MockONNXSession:
    def run(self, output_names, input_feed):
        logits = np.random.randn(1, 1, 100).astype(np.float32)
        return [logits]


class MockONNXEngine:
    model_name = "gpt2"
    session = MockONNXSession()
    tokenizer = None

    def __init__(self):
        self.tokenizer = type("Tok", (), {
            "eos_token_id": 0,
            "pad_token_id": 0,
            "decode": lambda self, tokens, skip_special_tokens=True: "hello",
        })()

    def generate(self, prompt, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9, do_sample=True):
        return "hello world"

    def generate_stream(self, prompt, max_new_tokens=100, **kwargs):
        yield "hello"
        yield " world"


def test_onnx_backend_generate():
    """ONNXBackend.generate() returns text."""
    engine = MockONNXEngine()
    backend = ONNXBackend(engine, MockTokenizer())
    result = backend.generate("hello", max_new_tokens=5, temperature=0.8, top_p=0.9, top_k=50, repetition_penalty=1.0)
    assert isinstance(result, dict)
    assert "text" in result
    assert isinstance(result["text"], str)
    assert len(result["text"]) > 0


def test_onnx_backend_generate_stream():
    """ONNXBackend.generate_stream() yields tokens."""
    engine = MockONNXEngine()
    backend = ONNXBackend(engine, MockTokenizer())
    tokens = list(backend.generate_stream("hello", max_new_tokens=5, temperature=0.8, top_p=0.9, top_k=50, repetition_penalty=1.0))
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_onnx_backend_alive():
    """ONNXBackend.alive is True when engine is set."""
    engine = MockONNXEngine()
    backend = ONNXBackend(engine, MockTokenizer())
    assert backend.alive is True


def test_onnx_backend_dead_when_no_engine():
    """ONNXBackend.alive is False when engine is None."""
    backend = ONNXBackend(None, None)
    assert backend.alive is False


def test_model_server_selects_onnx_backend():
    """ModelServer selects ONNXBackend when onnx_engine is provided."""
    engine = MockONNXEngine()
    server = ModelServer(
        model=MockModel(),
        tokenizer=MockTokenizer(),
        model_id="test",
        onnx_engine=engine,
    )
    backend = server._select_backend()
    assert isinstance(backend, ONNXBackend)


def test_model_server_falls_back_without_onnx():
    """ModelServer falls back to LocalBackend when no onnx_engine."""
    server = ModelServer(
        model=MockModel(),
        tokenizer=MockTokenizer(),
        model_id="test",
    )
    backend = server._select_backend()
    from domains.infrastructure.model_server import LocalBackend
    assert isinstance(backend, LocalBackend)
