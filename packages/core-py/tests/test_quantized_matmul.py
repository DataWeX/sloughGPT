"""
Tests for quantized matmul kernels (int8 GEMM).

Tests:
  - quantize_activation: float32 → int8
  - int8_matmul: int8 × int8 → float32
  - quantized_linear: full quantized linear layer
  - Numerical accuracy vs float32 matmul
  - Throughput comparison
"""

import time

import numpy as np
import pytest

from domains.infrastructure.quantization import (
    QuantEngine,
    TensorInfo,
    quantize_activation,
    int8_matmul,
    quantized_linear,
)


class TestQuantizeActivation:
    """Test float32 → int8 activation quantization."""

    def test_basic(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        q = quantize_activation(x, scale=1.0)
        np.testing.assert_array_equal(q, [-1, 0, 1])

    def test_with_scale(self):
        x = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        q = quantize_activation(x, scale=2.0 / 127)
        # Should map to approximately [-127, 0, 127]
        assert q[0] == -127
        assert q[1] == 0
        assert q[2] == 127

    def test_clipping(self):
        x = np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
        q = quantize_activation(x, scale=1.0)
        assert q[0] == -128  # clipped
        assert q[2] == 127   # clipped

    def test_preserves_shape(self):
        x = np.random.randn(3, 4, 5).astype(np.float32)
        q = quantize_activation(x, scale=0.1)
        assert q.shape == x.shape
        assert q.dtype == np.int8


class TestInt8Matmul:
    """Test int8 × int8 → float32 matmul."""

    def test_identity(self):
        """Multiply by identity should preserve values."""
        a = np.array([[1, 2, 3]], dtype=np.int8)
        b = np.eye(3, dtype=np.int8)  # identity matrix

        result = int8_matmul(a, b, a_scale=1.0, b_scale=1.0)
        np.testing.assert_array_equal(result, [[1, 2, 3]])

    def test_symmetric_basic(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int8)
        b = np.array([[5, 7], [6, 8]], dtype=np.int8)  # (N, K) format

        result = int8_matmul(a, b, a_scale=1.0, b_scale=1.0)

        # a @ b.T: b.T = [[5,6],[7,8]]
        # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_with_scales(self):
        a = np.array([[1, 2]], dtype=np.int8)
        b = np.array([[3, 4]], dtype=np.int8)

        result = int8_matmul(a, b, a_scale=0.1, b_scale=0.2)

        # a @ b.T = [[1*3+2*4]] = [[11]]
        # scaled: 11 * 0.1 * 0.2 = 0.22
        expected = np.array([[0.22]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_matches_float32(self):
        """int8 matmul should closely match float32 matmul for small quantization error."""
        rng = np.random.RandomState(42)
        a_fp = rng.randn(8, 16).astype(np.float32) * 0.5
        b_fp = rng.randn(4, 16).astype(np.float32) * 0.5

        # Quantize
        a_scale = np.max(np.abs(a_fp)) / 127
        b_scale = np.max(np.abs(b_fp)) / 127
        a_int = np.clip(np.round(a_fp / a_scale), -128, 127).astype(np.int8)
        b_int = np.clip(np.round(b_fp / b_scale), -128, 127).astype(np.int8)

        # Float32 reference
        y_fp32 = a_fp @ b_fp.T

        # Int8 matmul
        y_int8 = int8_matmul(a_int, b_int, a_scale=a_scale, b_scale=b_scale)

        # Should be very close
        cosine = np.dot(y_fp32.flatten(), y_int8.flatten()) / (
            np.linalg.norm(y_fp32) * np.linalg.norm(y_int8)
        )
        assert cosine > 0.99, f"Cosine similarity too low: {cosine}"


class TestQuantizedLinear:
    """Test full quantized linear layer."""

    def test_basic(self):
        x = np.random.randn(1, 16).astype(np.float32)
        w = np.random.randn(8, 16).astype(np.float32) * 0.02

        # Quantize weight
        engine = QuantEngine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)

        # Float32 reference
        y_fp32 = x @ w.T

        # Quantized linear
        y_quant = quantized_linear(x, info.array, info.meta.scale)

        cosine = np.dot(y_fp32.flatten(), y_quant.flatten()) / (
            np.linalg.norm(y_fp32) * np.linalg.norm(y_quant)
        )
        assert cosine > 0.99

    def test_with_bias(self):
        x = np.random.randn(1, 16).astype(np.float32)
        w = np.random.randn(8, 16).astype(np.float32) * 0.02
        bias = np.random.randn(8).astype(np.float32) * 0.01

        engine = QuantEngine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)

        y_fp32 = x @ w.T + bias
        y_quant = quantized_linear(x, info.array, info.meta.scale, bias=bias)

        cosine = np.dot(y_fp32.flatten(), y_quant.flatten()) / (
            np.linalg.norm(y_fp32) * np.linalg.norm(y_quant)
        )
        assert cosine > 0.99

    def test_batch(self):
        """Should handle batch dimensions."""
        x = np.random.randn(4, 128, 768).astype(np.float32)
        w = np.random.randn(768, 768).astype(np.float32) * 0.02

        engine = QuantEngine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)

        y_fp32 = x @ w.T
        y_quant = quantized_linear(x, info.array, info.meta.scale)

        assert y_quant.shape == y_fp32.shape
        cosine = np.dot(y_fp32.flatten(), y_quant.flatten()) / (
            np.linalg.norm(y_fp32) * np.linalg.norm(y_quant)
        )
        assert cosine > 0.99

    def test_correctness_at_scale(self):
        """Quantized linear produces correct results at realistic dimensions."""
        x = np.random.randn(1, 128, 768).astype(np.float32)
        w = np.random.randn(768, 768).astype(np.float32) * 0.02

        engine = QuantEngine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)

        y_fp32 = x @ w.T
        y_quant = quantized_linear(x, info.array, info.meta.scale)

        cosine = np.dot(y_fp32.flatten(), y_quant.flatten()) / (
            np.linalg.norm(y_fp32) * np.linalg.norm(y_quant)
        )
        assert cosine > 0.95, f"Cosine similarity too low: {cosine}"
        assert y_quant.shape == y_fp32.shape


class TestSloLinearQuantized:
    """Test that SloLinear uses int8 GEMM when quantized weight is set."""

    def test_quantized_forward_matches_float(self):
        """Quantized SloLinear should produce output close to float32."""
        from domains.training.slonet import SloLinear, Tensor

        layer = SloLinear(16, 8, bias=True)
        x = Tensor(np.random.randn(1, 16).astype(np.float32))

        # Float32 forward
        y_float = layer.forward_numpy(x.data)

        # Quantize weight
        engine = QuantEngine(bits=8, mode="symmetric")
        info = engine.quantize(layer.name, layer.weight.data.copy())
        assert info.is_quantized
        layer.set_quantized_weight(info)

        # Quantized forward
        y_quant = layer.forward_numpy(x.data)

        # Should be close
        cosine = np.dot(y_float.flatten(), y_quant.flatten()) / (
            np.linalg.norm(y_float) * np.linalg.norm(y_quant)
        )
        assert cosine > 0.95, f"Cosine similarity too low: {cosine}"

    def test_quantized_forward_uses_int8_matmul(self):
        """Verify the quantized path produces output consistent with int8 matmul."""
        from domains.training.slonet import SloLinear, Tensor

        layer = SloLinear(16, 8, bias=True)
        x = Tensor(np.random.randn(1, 16).astype(np.float32))

        # Quantize
        engine = QuantEngine(bits=8, mode="symmetric")
        info = engine.quantize(layer.name, layer.weight.data.copy())
        assert info.is_quantized
        layer.set_quantized_weight(info)

        # Direct quantized_linear call
        from domains.infrastructure.quantization import quantized_linear
        bias_arr = layer.bias.data if layer.use_bias else None
        y_direct = quantized_linear(x.data, info.array, info.meta.scale,
                                     info.meta.zero_point, bias_arr)

        # SloLinear quantized forward
        y_via_layer = layer.forward_numpy(x.data)

        # Should match
        np.testing.assert_allclose(y_direct, y_via_layer, atol=1e-5)
        assert y_via_layer.shape == y_direct.shape

    def test_autograd_tensor_forward_quantized(self):
        """Quantized SloLinear.forward() returns a Tensor."""
        from domains.training.slonet import SloLinear, Tensor

        layer = SloLinear(16, 8, bias=True)
        x = Tensor(np.random.randn(2, 16).astype(np.float32))

        engine = QuantEngine(bits=8, mode="symmetric")
        info = engine.quantize(layer.name, layer.weight.data.copy())
        layer.set_quantized_weight(info)

        y = layer.forward(x)
        assert isinstance(y, Tensor)
        assert y.data.shape == (2, 8)

    def test_no_quantize_uses_float(self):
        """Without quantized weight, forward uses float32 matmul."""
        from domains.training.slonet import SloLinear, Tensor

        layer = SloLinear(16, 8, bias=True)
        x = Tensor(np.random.randn(1, 16).astype(np.float32))

        # Forward without quantization
        y = layer.forward(x)
        assert isinstance(y, Tensor)
        assert y.data.shape == (1, 8)

        # forward_numpy should match
        y_np = layer.forward_numpy(x.data)
        np.testing.assert_allclose(y.data, y_np, atol=1e-5)
