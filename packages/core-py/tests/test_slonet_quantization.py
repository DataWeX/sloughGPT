"""Tests for training.slonet — quantization-related functions."""

from __future__ import annotations

import pytest
import numpy as np
from domains.training.slonet import (
    Tensor,
    SloLinear,
    _fuse_quant_weights,
    _fuse_quant_weights_int4,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


class MockQuantInfo:
    """Mock quantization info for testing."""
    def __init__(self, bits=8, scale=1.0, zero_point=0, original_shape=None):
        self.is_quantized = True
        self.meta = MockQuantMeta(bits=bits, scale=scale, zero_point=zero_point, original_shape=original_shape)
        self.array = None


class MockQuantMeta:
    def __init__(self, bits=8, scale=1.0, zero_point=0, original_shape=None):
        self.bits = bits
        self.scale = scale
        self.zero_point = zero_point
        self.original_shape = original_shape or (10, 10)


# ── _fuse_quant_weights ─────────────────────────────────────────────────────


class TestFuseQuantWeights:

    def test_returns_none_no_quantization(self):
        lin = SloLinear(10, 5)
        result = _fuse_quant_weights([lin])
        assert result is None

    def test_returns_none_mismatched_dims(self):
        lin1 = SloLinear(10, 5)
        lin2 = SloLinear(10, 3)
        lin1._quant_info = MockQuantInfo(original_shape=(5, 10))
        lin1._quant_info.array = np.zeros((5, 10), dtype=np.int8)
        lin2._quant_info = MockQuantInfo(original_shape=(3, 10))
        lin2._quant_info.array = np.zeros((3, 10), dtype=np.int8)
        # Different K dims should fail
        lin1._quant_info.array = np.zeros((5, 10), dtype=np.int8)
        lin2._quant_info.array = np.zeros((3, 8), dtype=np.int8)
        result = _fuse_quant_weights([lin1, lin2])
        assert result is None

    def test_returns_none_nonzero_zp(self):
        lin = SloLinear(10, 5)
        lin._quant_info = MockQuantInfo(zero_point=5, original_shape=(5, 10))
        lin._quant_info.array = np.zeros((5, 10), dtype=np.int8)
        result = _fuse_quant_weights([lin])
        assert result is None

    def test_fuse_success(self):
        lin1 = SloLinear(10, 5)
        lin2 = SloLinear(10, 3)
        lin1._quant_info = MockQuantInfo(scale=1.0, zero_point=0, original_shape=(5, 10))
        lin1._quant_info.array = np.ones((5, 10), dtype=np.int8)
        lin2._quant_info = MockQuantInfo(scale=1.0, zero_point=0, original_shape=(3, 10))
        lin2._quant_info.array = np.ones((3, 10), dtype=np.int8)
        result = _fuse_quant_weights([lin1, lin2])
        assert result is not None
        W, S, bias = result
        assert W.shape == (8, 10)
        assert S.shape == (8,)

    def test_fuse_with_bias(self):
        lin1 = SloLinear(10, 5, bias=True)
        lin2 = SloLinear(10, 3, bias=True)
        lin1._quant_info = MockQuantInfo(scale=1.0, zero_point=0, original_shape=(5, 10))
        lin1._quant_info.array = np.ones((5, 10), dtype=np.int8)
        lin2._quant_info = MockQuantInfo(scale=1.0, zero_point=0, original_shape=(3, 10))
        lin2._quant_info.array = np.ones((3, 10), dtype=np.int8)
        result = _fuse_quant_weights([lin1, lin2])
        assert result is not None
        W, S, bias = result
        assert bias is not None
        assert bias.shape == (8,)

    def test_fuse_mixed_bias(self):
        lin1 = SloLinear(10, 5, bias=True)
        lin2 = SloLinear(10, 3, bias=False)
        lin1._quant_info = MockQuantInfo(scale=1.0, zero_point=0, original_shape=(5, 10))
        lin1._quant_info.array = np.ones((5, 10), dtype=np.int8)
        lin2._quant_info = MockQuantInfo(scale=1.0, zero_point=0, original_shape=(3, 10))
        lin2._quant_info.array = np.ones((3, 10), dtype=np.int8)
        result = _fuse_quant_weights([lin1, lin2])
        assert result is None


# ── _fuse_quant_weights_int4 ────────────────────────────────────────────────


class TestFuseQuantWeightsInt4:

    def test_returns_none_no_quantization(self):
        lin = SloLinear(10, 5)
        result = _fuse_quant_weights_int4([lin])
        assert result is None

    def test_returns_none_not_int4(self):
        lin = SloLinear(10, 5)
        lin._quant_info = MockQuantInfo(bits=8, original_shape=(5, 10))
        lin._quant_info.array = np.zeros((5, 10), dtype=np.int8)
        result = _fuse_quant_weights_int4([lin])
        assert result is None

    def test_returns_none_mismatched_zp(self):
        lin1 = SloLinear(10, 5)
        lin2 = SloLinear(10, 3)
        lin1._quant_info = MockQuantInfo(bits=4, zero_point=0, original_shape=(5, 10))
        lin1._quant_info.array = np.zeros((5, 5), dtype=np.int8)
        lin2._quant_info = MockQuantInfo(bits=4, zero_point=1, original_shape=(3, 10))
        lin2._quant_info.array = np.zeros((3, 5), dtype=np.int8)
        result = _fuse_quant_weights_int4([lin1, lin2])
        assert result is None


# ── SloLinear.set_quantized_weight ──────────────────────────────────────────


class TestSetQuantizedWeight:

    def test_set_quantized_weight(self):
        lin = SloLinear(10, 5)
        info = MockQuantInfo(original_shape=(5, 10))
        lin.set_quantized_weight(info)
        assert lin._quant_info is info
        assert lin._quant_unpacked is None
        assert lin._weight_T_contig is None

    def test_get_quant_array_none(self):
        lin = SloLinear(10, 5)
        assert lin._get_quant_array() is None

    def test_free_quantized_originals_no_quant(self):
        lin = SloLinear(10, 5)
        assert lin.free_quantized_originals() is False

    def test_free_quantized_originals(self):
        lin = SloLinear(10, 5)
        info = MockQuantInfo(original_shape=(5, 10))
        lin.set_quantized_weight(info)
        assert lin.free_quantized_originals() is True
        assert lin._freed_shape == (5, 10)

    def test_free_quantized_idempotent(self):
        lin = SloLinear(10, 5)
        info = MockQuantInfo(original_shape=(5, 10))
        lin.set_quantized_weight(info)
        lin.free_quantized_originals()
        assert lin.free_quantized_originals() is True
