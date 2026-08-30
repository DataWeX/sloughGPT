"""
Tests for GPU accelerator precision selection.

Tests the base Accelerator class and module-level set_accelerator_precision()
function. Metal/CUDA backend tests require actual hardware.
"""

import numpy as np
import pytest

from domains.slolib.gpu import _Accelerator, get_accelerator, set_accelerator_precision, reset_accelerator


class TestBaseAcceleratorPrecision:
    """Tests for _Accelerator.set_precision() (base class, CPU path)."""

    def test_set_precision_fp32(self):
        acc = _Accelerator()
        result = acc.set_precision("fp32")
        assert result == "fp32"
        assert not acc._fp16_mode

    def test_set_precision_fp16_no_hardware(self):
        acc = _Accelerator()
        acc._fp16_available = False
        result = acc.set_precision("fp16")
        assert result == "fp32"
        assert not acc._fp16_mode

    def test_set_precision_auto_no_hardware(self):
        acc = _Accelerator()
        result = acc.set_precision("auto")
        assert result == "fp32"
        assert not acc._fp16_mode

    def test_set_precision_fp16_with_hardware(self):
        acc = _Accelerator()
        acc._fp16_available = True
        result = acc.set_precision("fp16")
        assert result == "fp16"
        assert acc._fp16_mode

    def test_set_precision_back_to_fp32(self):
        acc = _Accelerator()
        acc._fp16_available = True
        acc.set_precision("fp16")
        assert acc._fp16_mode
        result = acc.set_precision("fp32")
        assert result == "fp32"
        assert not acc._fp16_mode

    def test_precision_benchmark_no_hardware(self):
        acc = _Accelerator()
        result = acc._prec_benchmark()
        assert result == "fp32"

    def test_set_precision_unknown_mode(self):
        acc = _Accelerator()
        result = acc.set_precision("unknown")
        assert result == "fp32"
        assert not acc._fp16_mode

    def test_set_precision_empty_string(self):
        acc = _Accelerator()
        result = acc.set_precision("")
        assert result == "fp32"

    def test_fp16_mode_toggle(self):
        acc = _Accelerator()
        acc._fp16_available = True
        acc.set_precision("fp16")
        assert acc._fp16_mode is True
        acc.set_precision("fp32")
        assert acc._fp16_mode is False

    def test_set_precision_returns_string(self):
        acc = _Accelerator()
        result = acc.set_precision("fp32")
        assert isinstance(result, str)


class TestAcceleratorBaseOps:
    """Tests for _Accelerator base math and data operations."""

    def test_matmul(self):
        acc = _Accelerator()
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((3, 4), dtype=np.float32)
        result = acc.matmul(a, b)
        assert result.shape == (2, 4)
        assert np.allclose(result, 3.0)

    def test_add(self):
        acc = _Accelerator()
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = acc.add(a, b)
        assert np.allclose(result, [4.0, 6.0])

    def test_sub(self):
        acc = _Accelerator()
        a = np.array([5.0, 3.0])
        b = np.array([1.0, 2.0])
        result = acc.sub(a, b)
        assert np.allclose(result, [4.0, 1.0])

    def test_mul(self):
        acc = _Accelerator()
        a = np.array([2.0, 3.0])
        b = np.array([4.0, 5.0])
        result = acc.mul(a, b)
        assert np.allclose(result, [8.0, 15.0])

    def test_div(self):
        acc = _Accelerator()
        a = np.array([10.0, 8.0])
        b = np.array([2.0, 4.0])
        result = acc.div(a, b)
        assert np.allclose(result, [5.0, 2.0])

    def test_pow(self):
        acc = _Accelerator()
        a = np.array([2.0, 3.0])
        result = acc.pow(a, 2)
        assert np.allclose(result, [4.0, 9.0])

    def test_sqrt(self):
        acc = _Accelerator()
        a = np.array([4.0, 9.0])
        result = acc.sqrt(a)
        assert np.allclose(result, [2.0, 3.0])

    def test_exp(self):
        acc = _Accelerator()
        a = np.array([0.0, 1.0])
        result = acc.exp(a)
        assert np.allclose(result, [1.0, np.e], rtol=1e-5)

    def test_log(self):
        acc = _Accelerator()
        a = np.array([1.0, np.e])
        result = acc.log(a)
        assert np.allclose(result, [0.0, 1.0], rtol=1e-5)

    def test_sum(self):
        acc = _Accelerator()
        a = np.array([1.0, 2.0, 3.0])
        assert acc.sum(a) == 6.0

    def test_sum_axis(self):
        acc = _Accelerator()
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = acc.sum(a, axis=0)
        assert np.allclose(result, [4.0, 6.0])

    def test_mean(self):
        acc = _Accelerator()
        a = np.array([2.0, 4.0])
        assert acc.mean(a) == 3.0

    def test_mean_axis(self):
        acc = _Accelerator()
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = acc.mean(a, axis=1)
        assert np.allclose(result, [1.5, 3.5])

    def test_max(self):
        acc = _Accelerator()
        a = np.array([1.0, 5.0, 3.0])
        assert acc.max(a) == 5.0

    def test_min(self):
        acc = _Accelerator()
        a = np.array([5.0, 1.0, 3.0])
        assert acc.min(a) == 1.0

    def test_abs(self):
        acc = _Accelerator()
        a = np.array([-3.0, 4.0])
        result = acc.abs(a)
        assert np.allclose(result, [3.0, 4.0])

    def test_neg(self):
        acc = _Accelerator()
        a = np.array([1.0, -2.0])
        result = acc.neg(a)
        assert np.allclose(result, [-1.0, 2.0])

    def test_clamp(self):
        acc = _Accelerator()
        a = np.array([-1.0, 0.5, 2.0])
        result = acc.clamp(a, 0.0, 1.0)
        assert np.allclose(result, [0.0, 0.5, 1.0])

    def test_where(self):
        acc = _Accelerator()
        cond = np.array([True, False])
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = acc.where(cond, a, b)
        assert np.allclose(result, [1.0, 4.0])

    def test_gather(self):
        acc = _Accelerator()
        a = np.array([[10.0, 20.0, 30.0]])
        idx = np.array([[0, 2]])
        result = acc.gather(a, dim=1, index=idx)
        assert np.allclose(result, [[10.0, 30.0]])

    def test_scatter(self):
        acc = _Accelerator()
        a = np.zeros((1, 5))
        idx = np.array([[1, 3]])
        src = np.array([[10.0, 20.0]])
        result = acc.scatter(a, dim=1, index=idx, src=src)
        assert result[0, 1] == 10.0
        assert result[0, 3] == 20.0

    def test_pad(self):
        acc = _Accelerator()
        a = np.array([1.0, 2.0])
        result = acc.pad(a, [(1, 1)], mode="constant", constant_values=0.0)
        assert np.allclose(result, [0.0, 1.0, 2.0, 0.0])

    def test_softmax(self):
        acc = _Accelerator()
        a = np.array([[1.0, 2.0, 3.0]])
        result = acc.softmax(a)
        assert np.allclose(result.sum(axis=-1), 1.0)

    def test_log_softmax(self):
        acc = _Accelerator()
        a = np.array([[1.0, 2.0, 3.0]])
        result = acc.log_softmax(a)
        assert result.shape == (1, 3)

    def test_layer_norm(self):
        acc = _Accelerator()
        x = np.ones((1, 4))
        w = np.ones(4)
        b = np.zeros(4)
        result = acc.layer_norm(x, w, b)
        assert np.allclose(result, 0.0, atol=1e-5)

    def test_rms_norm(self):
        acc = _Accelerator()
        x = np.ones((1, 4))
        w = np.ones(4)
        result = acc.rms_norm(x, w)
        assert result.shape == (1, 4)

    def test_gelu(self):
        acc = _Accelerator()
        a = np.array([0.0, 1.0, -1.0])
        result = acc.gelu(a)
        assert result.shape == a.shape
        assert result[1] > result[0]

    def test_silu(self):
        acc = _Accelerator()
        a = np.array([0.0, 1.0, -1.0])
        result = acc.silu(a)
        assert result.shape == a.shape

    def test_relu(self):
        acc = _Accelerator()
        a = np.array([-1.0, 0.0, 1.0])
        result = acc.relu(a)
        assert np.allclose(result, [0.0, 0.0, 1.0])

    def test_sigmoid(self):
        acc = _Accelerator()
        a = np.array([0.0])
        result = acc.sigmoid(a)
        assert np.allclose(result, [0.5], atol=1e-5)

    def test_tanh(self):
        acc = _Accelerator()
        a = np.array([0.0])
        result = acc.tanh(a)
        assert np.allclose(result, [0.0], atol=1e-5)

    def test_fused_add_mul(self):
        acc = _Accelerator()
        a = np.array([1.0])
        b = np.array([2.0])
        c = np.array([3.0])
        result = acc.fused_add_mul(a, b, c)
        assert np.allclose(result, [7.0])

    def test_fused_mul_add(self):
        acc = _Accelerator()
        a = np.array([2.0])
        b = np.array([3.0])
        c = np.array([4.0])
        result = acc.fused_mul_add(a, b, c)
        assert np.allclose(result, [10.0])


class TestAcceleratorAttention:
    """Tests for scaled_dot_attention and multi_head_attention."""

    def test_scaled_dot_attention_basic(self):
        acc = _Accelerator()
        q = np.random.randn(1, 2, 4, 8).astype(np.float32)
        k = np.random.randn(1, 2, 4, 8).astype(np.float32)
        v = np.random.randn(1, 2, 4, 8).astype(np.float32)
        result = acc.scaled_dot_attention(q, k, v)
        assert result.shape == (1, 2, 4, 8)

    def test_scaled_dot_attention_with_mask(self):
        acc = _Accelerator()
        q = np.random.randn(1, 1, 4, 8).astype(np.float32)
        k = np.random.randn(1, 1, 4, 8).astype(np.float32)
        v = np.random.randn(1, 1, 4, 8).astype(np.float32)
        mask = np.zeros((1, 1, 4, 4), dtype=np.float32)
        result = acc.scaled_dot_attention(q, k, v, mask=mask)
        assert result.shape == (1, 1, 4, 8)

    def test_scaled_dot_attention_causal(self):
        acc = _Accelerator()
        q = np.random.randn(1, 1, 4, 8).astype(np.float32)
        k = np.random.randn(1, 1, 4, 8).astype(np.float32)
        v = np.random.randn(1, 1, 4, 8).astype(np.float32)
        result = acc.scaled_dot_attention(q, k, v, causal=True)
        assert result.shape == (1, 1, 4, 8)

    def test_scaled_dot_attention_custom_scale(self):
        acc = _Accelerator()
        q = np.random.randn(1, 1, 4, 8).astype(np.float32)
        k = np.random.randn(1, 1, 4, 8).astype(np.float32)
        v = np.random.randn(1, 1, 4, 8).astype(np.float32)
        result = acc.scaled_dot_attention(q, k, v, scale=0.1)
        assert result.shape == (1, 1, 4, 8)

    def test_multi_head_attention(self):
        acc = _Accelerator()
        q = np.random.randn(1, 4, 64).astype(np.float32)
        k = np.random.randn(1, 4, 64).astype(np.float32)
        v = np.random.randn(1, 4, 64).astype(np.float32)
        out, attn_weights = acc.multi_head_attention(q, k, v, num_heads=4)
        assert out.shape == (1, 4, 64)

    def test_multi_head_attention_causal(self):
        acc = _Accelerator()
        q = np.random.randn(1, 4, 64).astype(np.float32)
        k = np.random.randn(1, 4, 64).astype(np.float32)
        v = np.random.randn(1, 4, 64).astype(np.float32)
        out, _ = acc.multi_head_attention(q, k, v, num_heads=4, causal=True)
        assert out.shape == (1, 4, 64)


class TestAcceleratorDataOps:
    """Tests for reshape, transpose, topk, etc."""

    def test_reshape(self):
        acc = _Accelerator()
        a = np.arange(6.0)
        result = acc.reshape(a, (2, 3))
        assert result.shape == (2, 3)

    def test_transpose(self):
        acc = _Accelerator()
        a = np.array([[1.0, 2.0, 3.0]])
        result = acc.transpose(a)
        assert result.shape == (3, 1)

    def test_transpose_axes(self):
        acc = _Accelerator()
        a = np.ones((2, 3, 4))
        result = acc.transpose(a, (2, 0, 1))
        assert result.shape == (4, 2, 3)

    def test_concat(self):
        acc = _Accelerator()
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = acc.concat([a, b])
        assert np.allclose(result, [1.0, 2.0, 3.0, 4.0])

    def test_stack(self):
        acc = _Accelerator()
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = acc.stack([a, b])
        assert result.shape == (2, 2)

    def test_permute(self):
        acc = _Accelerator()
        a = np.ones((2, 3, 4))
        result = acc.permute(a, (1, 2, 0))
        assert result.shape == (3, 4, 2)

    def test_topk(self):
        acc = _Accelerator()
        a = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        values, indices = acc.topk(a, k=2)
        assert len(values) == 2
        assert values[0] == 5.0

    def test_topk_smallest(self):
        acc = _Accelerator()
        a = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        values, indices = acc.topk(a, k=2, largest=False)
        assert values[0] == 1.0

    def test_multinomial(self):
        acc = _Accelerator()
        probs = np.array([0.5, 0.3, 0.2])
        result = acc.multinomial(probs, num_samples=10, replacement=True)
        assert result.shape[1] == 10

    def test_dropout_training(self):
        acc = _Accelerator()
        a = np.ones((2, 4))
        result = acc.dropout(a, p=0.5, training=True)
        assert result.shape == a.shape

    def test_dropout_eval(self):
        acc = _Accelerator()
        a = np.ones((2, 4))
        result = acc.dropout(a, p=0.5, training=False)
        assert np.allclose(result, a)

    def test_one_hot(self):
        acc = _Accelerator()
        idx = np.array([0, 2])
        result = acc.one_hot(idx, num_classes=3)
        assert result.shape == (2, 3)
        assert result[0, 0] == 1.0
        assert result[1, 2] == 1.0

    def test_embedding_lookup(self):
        acc = _Accelerator()
        weight = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        idx = np.array([0, 2])
        result = acc.embedding_lookup(idx, weight)
        assert np.allclose(result, [[1.0, 2.0], [5.0, 6.0]])

    def test_cross_entropy(self):
        acc = _Accelerator()
        logits = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([2])
        loss = acc.cross_entropy(logits, targets)
        assert loss >= 0.0
        assert loss < 10.0

    def test_to_device(self):
        acc = _Accelerator()
        a = np.array([1.0, 2.0])
        result = acc.to_device(a)
        assert isinstance(result, np.ndarray)

    def test_from_device(self):
        acc = _Accelerator()
        a = np.array([1.0, 2.0])
        result = acc.from_device(a)
        assert isinstance(result, np.ndarray)

    def test_sync(self):
        acc = _Accelerator()
        acc.sync()


class TestModuleLevelFunctions:
    """Tests for module-level get_accelerator() / set_accelerator_precision()."""

    def setup_method(self):
        reset_accelerator()

    def test_get_accelerator_returns_instance(self):
        acc = get_accelerator()
        assert isinstance(acc, _Accelerator)

    def test_set_accelerator_precision_fp32(self):
        reset_accelerator()
        result = set_accelerator_precision("fp32")
        assert result == "fp32"

    def test_set_accelerator_precision_auto(self):
        reset_accelerator()
        result = set_accelerator_precision("auto")
        assert result == "fp32"

    def test_get_accelerator_singleton(self):
        acc1 = get_accelerator()
        acc2 = get_accelerator()
        assert acc1 is acc2

    def test_reset_accelerator(self):
        acc1 = get_accelerator()
        reset_accelerator()
        acc2 = get_accelerator()
        assert acc1 is not acc2

    def test_set_accelerator_precision_fp16(self):
        reset_accelerator()
        result = set_accelerator_precision("fp16")
        assert result == "fp32"

    def test_set_accelerator_precision_unknown(self):
        reset_accelerator()
        result = set_accelerator_precision("invalid")
        assert result == "fp32"

    def test_memory_hint(self):
        acc = get_accelerator()
        hint = acc.memory_hint()
        assert "tier" in hint

    def test_is_available(self):
        acc = get_accelerator()
        assert acc.is_available()

    def test_vram(self):
        acc = get_accelerator()
        vram = acc.vram_gb()
        assert isinstance(vram, float)
        assert vram >= 0.0
