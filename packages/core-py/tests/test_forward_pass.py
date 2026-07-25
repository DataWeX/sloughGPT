"""Tests for the unified forward pass interface."""

import pytest
import numpy as np

from domains.inference.forward_pass import (
    ForwardPassable,
    ForwardPassResult,
    timed_forward,
)


# ── ForwardPassResult ──────────────────────────────────────────────────────

class TestForwardPassResult:
    def test_basic_construction(self):
        logits = np.random.randn(1, 10, 256).astype(np.float32)
        result = ForwardPassResult(logits=logits)
        assert result.shape == [1, 10, 256]
        assert result.engine == "unknown"
        assert result.forward_time_ms == 0.0

    def test_shape_property(self):
        logits = np.random.randn(2, 5, 128).astype(np.float32)
        result = ForwardPassResult(logits=logits)
        assert result.shape == [2, 5, 128]

    def test_engine_tracking(self):
        logits = np.zeros((1, 1, 100), dtype=np.float32)
        result = ForwardPassResult(logits=logits, engine="numpy")
        assert result.engine == "numpy"
        result2 = ForwardPassResult(logits=logits, engine="c")
        assert result2.engine == "c"

    def test_model_name_tracking(self):
        logits = np.zeros((1, 1, 100), dtype=np.float32)
        result = ForwardPassResult(logits=logits, model_name="qwen-0.5b")
        assert result.model_name == "qwen-0.5b"


# ── Protocol compliance ────────────────────────────────────────────────────

class MockNumpyEngine:
    """Satisfies ForwardPassable via numpy."""
    def __init__(self, vocab=256):
        self.vocab = vocab

    def forward_pass(self, input_ids: np.ndarray) -> ForwardPassResult:
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        logits = np.random.randn(input_ids.shape[0], input_ids.shape[1],
                                 self.vocab).astype(np.float32)
        return ForwardPassResult(logits=logits, engine="numpy")


class MockCEngine:
    """Satisfies ForwardPassable via C-style processing."""
    def __init__(self, vocab=50257):
        self.vocab = vocab

    def forward_pass(self, input_ids: np.ndarray) -> ForwardPassResult:
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        logits = np.random.randn(1, 1, self.vocab).astype(np.float32)
        return ForwardPassResult(logits=logits, engine="c")


class NotForwardPassable:
    """Does NOT satisfy the protocol."""
    def compute(self, x):
        return x


class TestForwardPassableProtocol:
    def test_numpy_engine_satisfies_protocol(self):
        engine = MockNumpyEngine()
        assert isinstance(engine, ForwardPassable)

    def test_c_engine_satisfies_protocol(self):
        engine = MockCEngine()
        assert isinstance(engine, ForwardPassable)

    def test_non_conforming_class_rejected(self):
        obj = NotForwardPassable()
        assert not isinstance(obj, ForwardPassable)

    def test_dict_rejected(self):
        assert not isinstance({}, ForwardPassable)


# ── timed_forward ──────────────────────────────────────────────────────────

class TestTimedForward:
    def test_measures_time(self):
        engine = MockNumpyEngine(vocab=100)
        ids = np.array([[1, 2, 3]], dtype=np.int64)
        result = timed_forward(engine, ids, model_name="test-model")
        assert result.forward_time_ms >= 0
        assert result.model_name == "test-model"
        assert result.shape == [1, 3, 100]

    def test_preserves_engine_field(self):
        engine = MockCEngine(vocab=200)
        ids = np.array([[5]], dtype=np.int64)
        result = timed_forward(engine, ids)
        assert result.engine == "c"
        assert result.shape == [1, 1, 200]


# ── SloTransformer.forward_pass ───────────────────────────────────────────

class TestSloTransformerForwardPass:
    """Test that SloTransformer.forward_pass() returns ForwardPassResult."""

    def test_returns_forward_pass_result(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(vocab_size=64, n_embed=32, n_layer=2,
                               n_head=4, block_size=16, max_seq_len=32)
        ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
        result = model.forward_pass(ids)
        assert isinstance(result, ForwardPassResult)
        assert result.engine == "numpy"

    def test_output_shape(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(vocab_size=64, n_embed=32, n_layer=2,
                               n_head=4, block_size=16, max_seq_len=32)
        ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
        result = model.forward_pass(ids)
        assert result.logits.shape == (1, 4, 64)

    def test_1d_input_reshaped(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(vocab_size=64, n_embed=32, n_layer=2,
                               n_head=4, block_size=16, max_seq_len=32)
        ids = np.array([1, 2, 3], dtype=np.int64)
        result = model.forward_pass(ids)
        assert result.logits.shape == (1, 3, 64)

    def test_logits_are_finite(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(vocab_size=64, n_embed=32, n_layer=2,
                               n_head=4, block_size=16, max_seq_len=32)
        ids = np.array([[1, 2]], dtype=np.int64)
        result = model.forward_pass(ids)
        assert np.all(np.isfinite(result.logits))

    def test_satisfies_forward_passable(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(vocab_size=64, n_embed=32, n_layer=2,
                               n_head=4, block_size=16, max_seq_len=32)
        assert isinstance(model, ForwardPassable)


# ── NPU uses forward_pass ──────────────────────────────────────────────────

class TestNPUUsesForwardPass:
    """Verify NPUDevice.forward() calls forward_pass(), not forward_numpy()."""

    def _make_npu_with_mock(self):
        """Create an NPU with a MockProvider directly registered."""
        import time as _time
        from tests.test_npu import MockProvider
        from domains.shell.kernel_npu import NPUDevice, NPUModel

        npu = NPUDevice()
        npu.open()
        provider = MockProvider(vocab_size=64, n_embed=32)
        model = NPUModel(
            name="test",
            provider=provider,
            config=provider.metadata(),
            loaded_at=_time.time(),
        )
        npu._models["test"] = model
        npu._default_model = "test"
        return npu, provider

    def test_npu_forward_returns_engine_field(self):
        npu, _ = self._make_npu_with_mock()
        result = npu.forward("test", [1, 2, 3, 4])
        assert result.success
        data = result.value
        assert "engine" in data
        assert data["engine"] == "numpy"
        assert "logits" in data
        assert data["shape"] == [1, 4, 64]

    def test_npu_forward_uses_unified_interface(self):
        """Verify the NPU calls forward_pass, not forward_numpy."""
        from domains.shell.kernel_npu import NPUDevice
        from unittest.mock import patch

        npu, provider = self._make_npu_with_mock()
        model = provider._model

        with patch.object(type(model), 'forward_pass') as mock_fp:
            mock_fp.return_value = ForwardPassResult(
                logits=np.zeros((1, 4, 64), dtype=np.float32),
                engine="numpy",
            )
            result = npu.forward("test", [1, 2, 3, 4])
            mock_fp.assert_called_once()
            assert result.success
