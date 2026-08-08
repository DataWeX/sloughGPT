"""
Supplementary coverage tests for the quantization engine.

Targets the remaining untested branches in quantization.py: the numpy
fallback GEMM kernels, the quant_core import-failure path, TensorInfo
unquantized accessors, quantize_with_scale, per-channel/asymmetric
dequantization, the asymmetric matmul paths, and the QuantizedLinear
torch/numpy dispatch (using a fake torch module).
"""

import importlib
import sys

import numpy as np
import pytest

from domains.infrastructure import quantization as q
from domains.infrastructure.quantization import (
    Quantine,
    QuantMeta,
    QuantMode,
    QuantizedLinear,
    TensorInfo,
    _dequantize,
    _ensure_2d_packed,
    _int4_numpy_fallback,
    _numpy_fallback,
    _pack_int4,
    _unpack_int4,
    int4_matmul,
    int4_quantized_linear,
    int8_matmul,
    quantize_activation,
    quantized_linear,
    walk_hf_linears,
)


class Linear:
    def __init__(self, has_weight=True):
        if has_weight:
            self.weight = np.ones((4, 4), dtype=np.float32)


class Embedding:
    def __init__(self):
        self.weight = np.ones((4, 4), dtype=np.float32)


class _FakeHfModel:
    def __init__(self):
        self.modules = [
            ("lm_head", Linear(True)),
            ("embed", Embedding()),
            ("no_weight", Linear(False)),
        ]

    def named_modules(self):
        yield from self.modules


# ══════════════════════════════════════════════════════════════════════════════
# walk_hf_linears
# ══════════════════════════════════════════════════════════════════════════════


def test_walk_hf_linears_collects_only_linear_with_weight():
    model = _FakeHfModel()
    layers = walk_hf_linears(model)
    assert set(layers.keys()) == {"lm_head"}
    assert isinstance(layers["lm_head"], Linear)


# ══════════════════════════════════════════════════════════════════════════════
# numpy fallback GEMM kernels
# ══════════════════════════════════════════════════════════════════════════════


def test_numpy_fallback_matches_int32_gemm():
    a = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int8)
    b = np.array([[1, 0, -1], [2, -2, 1]], dtype=np.int8)
    expected = a.astype(np.int32) @ b.astype(np.int32).T
    assert np.array_equal(_numpy_fallback(a, b), expected)


def test_int4_numpy_fallback_matches_unpacked_gemm():
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int8)
    B = np.array([[1, -2, 3, 4], [0, 1, -1, 2], [7, -8, 5, 6]], dtype=np.int8)
    packed = _pack_int4(B.flatten()).reshape(B.shape[0], B.shape[1] // 2)

    out = _int4_numpy_fallback(A, packed, B.shape[1])

    B_ref = np.stack(
        [_unpack_int4(packed[j], B.shape[1], signed=True) for j in range(B.shape[0])]
    )
    expected = A.astype(np.int32) @ B_ref.astype(np.int32).T
    assert np.array_equal(out, expected)


def test_quant_core_import_failure_falls_back_to_numpy(monkeypatch):
    import domains.infrastructure.quant_core.wrapper as wrapper

    class _Broken:
        def __getattr__(self, name):
            raise ImportError(f"cannot import name {name!r} from fake wrapper")

    monkeypatch.setitem(sys.modules, "domains.infrastructure.quant_core.wrapper", _Broken())
    importlib.reload(q)
    try:
        assert q._c_matmul is q._numpy_fallback
        assert q._c_matmul_int4 is q._int4_numpy_fallback
    finally:
        monkeypatch.setitem(sys.modules, "domains.infrastructure.quant_core.wrapper", wrapper)
        importlib.reload(q)


# ══════════════════════════════════════════════════════════════════════════════
# TensorInfo accessors
# ══════════════════════════════════════════════════════════════════════════════


def _make_meta(original_shape=(4,), original_dtype="float32"):
    return QuantMeta(
        scale=0.1,
        zero_point=0,
        bits=8,
        mode="symmetric",
        dtype_code=5,
        original_shape=original_shape,
        original_dtype=original_dtype,
    )


def test_tensor_info_meta_dtype_and_shape():
    info = TensorInfo(name="w", array=np.zeros(4, dtype=np.int8), meta=_make_meta())
    assert info.dtype == np.dtype("float32")
    assert info.shape == (4,)


def test_tensor_info_plain_nbytes():
    info = TensorInfo(name="w", array=np.ones(4, dtype=np.float32))
    assert info.nbytes == 16


def test_tensor_info_as_float_casts_non_fp32():
    info = TensorInfo(name="w", array=np.array([1, 2, 3], dtype=np.int32))
    out = info.as_float()
    assert out.dtype == np.float32
    assert np.array_equal(out, np.array([1.0, 2.0, 3.0]))


def test_tensor_info_plain_quantized_bytes_and_ratio():
    info = TensorInfo(name="w", array=np.ones(4, dtype=np.float32))
    assert info.quantized_bytes() == 16
    assert info.compression_ratio() == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Quantine branches
# ══════════════════════════════════════════════════════════════════════════════


def test_is_sensitive_true_for_norm_prefixes():
    engine = Quantine(bits=8)
    assert engine.is_sensitive("attn_norm.weight")
    assert engine.is_sensitive("ff_norm.weight")
    assert not engine.is_sensitive("q_proj.weight")


def test_quantize_skips_when_relative_mse_above_threshold():
    engine = Quantine(bits=8, mode="symmetric")
    engine._error_threshold = 1e-12
    arr = (np.random.RandomState(0).randn(32) * 5).astype(np.float32)
    info = engine.quantize("test.w", arr)
    assert info.meta is None
    assert np.array_equal(info.array, arr)


def test_quantize_per_channel_skips_when_above_threshold():
    engine = Quantine(bits=8, mode="symmetric")
    engine._error_threshold = 1e-12
    arr = (np.random.RandomState(1).randn(8, 16) * 3).astype(np.float32)
    info = engine.quantize("test.mat", arr)
    assert info.meta is None
    assert np.array_equal(info.array, arr)


def test_dequantize_to_float_matches_as_float():
    engine = Quantine(bits=8, mode="symmetric")
    arr = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    info = engine.quantize("test.w", arr)
    out = engine.dequantize_to_float(info)
    assert out.dtype == np.float32
    assert np.allclose(out, info.as_float())


def test_summary_empty_engine():
    engine = Quantine(bits=8)
    assert engine.summary() == {"tensors": 0}


def test_compute_asymmetric_constant_array():
    engine = Quantine(bits=8, mode="asymmetric")
    scale, zero_point = engine._compute_asymmetric(np.full(8, 2.5, dtype=np.float32))
    assert scale == 1.0
    assert zero_point == 0


def test_quantize_with_scale_skip_prefix():
    engine = Quantine(bits=8, mode="symmetric")
    info = engine.quantize_with_scale("norm.final.weight", np.ones(4), 0.01)
    assert info.meta is None


def test_quantize_with_scale_scalar_path():
    engine = Quantine(bits=8, mode="symmetric")
    arr = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    info = engine.quantize_with_scale("test.w", arr, scale=0.1)
    assert info.is_quantized
    assert info.meta.scale == 0.1
    assert np.allclose(info.as_float(), arr, atol=0.15)


def test_quantize_with_scale_per_channel_symmetric():
    engine = Quantine(bits=8, mode="symmetric")
    mat = np.array([[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]], dtype=np.float32)
    scale = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    info = engine.quantize_with_scale("test.mat", mat, scale=scale)
    assert info.is_quantized
    assert info.meta.is_per_channel
    assert info.as_float().shape == (3, 2)


def test_dequantize_per_channel_with_zero_point():
    grid = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int8)
    scale = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = _dequantize(grid, scale, zero_point=3, bits=8, original_shape=(3, 2))
    assert out.shape == (3, 2)
    assert np.allclose(out, (grid - 3) * scale.reshape(-1, 1))


# ══════════════════════════════════════════════════════════════════════════════
# module-level kernels — remaining branches
# ══════════════════════════════════════════════════════════════════════════════


def test_quantize_activation_with_zero_point():
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    out = quantize_activation(x, 0.1, zero_point=3)
    expected = np.clip(np.round(x / 0.1) + 3, -128, 127).astype(np.int8)
    assert np.array_equal(out, expected)


def test_ensure_2d_packed_passthrough():
    b = np.zeros((3, 2), dtype=np.uint8)
    assert _ensure_2d_packed(b, 4) is b


def test_int4_matmul_asymmetric_zero_point():
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int8)
    B = np.array([[1, -2, 3, 4], [0, 1, -1, 2]], dtype=np.int8)
    packed = _pack_int4(B.flatten()).reshape(2, 2)

    res = int4_matmul(a, packed, a_scale=0.1, b_scale=0.2, orig_k=4, b_zero_point=1)
    assert res.shape == (2, 2)

    B_ref = np.stack([_unpack_int4(packed[j], 4, signed=True) for j in range(2)])
    expected = int8_matmul(a, B_ref, 0.1, 0.2, a_zero_point=0, b_zero_point=1)
    assert np.allclose(res, expected)


def test_int8_matmul_asymmetric_zero_point():
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    b = np.array([[1, 0, -1], [2, -2, 1]], dtype=np.int8)
    res = int8_matmul(a, b, a_scale=0.1, b_scale=0.2, a_zero_point=1, b_zero_point=0)
    assert res.shape == (2, 2)
    assert np.all(np.isfinite(res))


def test_quantized_linear_with_explicit_x_scale():
    x = (np.random.RandomState(0).randn(3, 16) * 0.1).astype(np.float32)
    w = (np.random.RandomState(1).randn(8, 16) * 0.1).astype(np.float32)
    wq = np.clip(np.round(w / 0.01), -128, 127).astype(np.int8)
    res = quantized_linear(x, wq, weight_scale=0.01, x_scale=0.05)
    assert res.shape == (3, 8)


def test_quantized_linear_weight_only_scale_zero_row_and_bias():
    # Weight-only path (x_scale=None): per-token activation scale with a zero
    # row (np.where branches) plus the bias add. A non-zero weight_zero_point
    # bypasses the fused W8A8 C path so the numpy per-token scaling executes.
    x = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 2.0, -2.0]], dtype=np.float32)
    w = (np.random.RandomState(1).randn(8, 4) * 0.1).astype(np.float32)
    wq = np.clip(np.round(w / 0.01), -128, 127).astype(np.int8)
    bias = np.ones(8, dtype=np.float32)
    res = quantized_linear(x, wq, weight_scale=0.01, bias=bias, weight_zero_point=1)
    assert res.shape == (2, 8)
    assert np.all(np.isfinite(res))


def test_int4_quantized_linear_with_explicit_x_scale_and_zero_point():
    K = 8
    B = np.random.RandomState(2).randint(-8, 8, size=(5, K)).astype(np.int8)
    packed = _pack_int4(B.flatten()).reshape(5, K // 2)
    x = (np.random.RandomState(3).randn(2, K) * 0.1).astype(np.float32)
    res = int4_quantized_linear(
        x, packed, weight_scale=0.1, weight_zero_point=2, orig_k=K, x_scale=0.05
    )
    assert res.shape == (2, 5)
    assert np.all(np.isfinite(res))


# ══════════════════════════════════════════════════════════════════════════════
# QuantizedLinear
# ══════════════════════════════════════════════════════════════════════════════


class _FakeData:
    def cpu(self):
        return self

    def numpy(self):
        return np.array([0.1, -0.2, 0.3], dtype=np.float32)


class _FakeBias:
    data = _FakeData()


class _FakeLinearModule:
    bias = _FakeBias()


def test_from_linear_extracts_bias():
    info = TensorInfo(
        name="w",
        array=np.zeros((3, 8), dtype=np.int8),
        meta=_make_meta(original_shape=(3, 8)),
    )
    ql = QuantizedLinear.from_linear(_FakeLinearModule(), info)
    assert ql.bias is not None
    assert ql.bias.shape == (3,)


def _make_ql():
    w8 = np.clip(np.round(np.random.RandomState(0).randn(2, 16) * 100), -128, 127).astype(np.int8)
    return QuantizedLinear(
        weight_int8=w8,
        scale=0.01,
        zero_point=0,
        bias=None,
        bits=8,
        original_shape=(2, 16),
        mode="symmetric",
    )


def test_dequantize_cached():
    ql = _make_ql()
    w1 = ql.dequantize()
    w2 = ql.dequantize()
    assert w1 is w2
    assert w1.shape == (2, 16)


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeTorch:
    Tensor = _FakeTensor

    @staticmethod
    def from_numpy(arr):
        return _FakeTensor(arr)


def test_make_torch_forward_invoked(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    ql = _make_ql()
    fwd = ql.make_torch_forward()
    out = fwd(_FakeTensor(np.ones((2, 16))))
    assert isinstance(out, _FakeTensor)
    assert out.numpy().shape == (2, 2)


def test_call_torch_tensor(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    ql = _make_ql()
    out = ql(_FakeTensor(np.ones((2, 16))))
    assert isinstance(out, _FakeTensor)


def test_call_numpy_forward_without_torch():
    ql = _make_ql()
    out = ql(np.ones((2, 16), dtype=np.float32))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)


# ══════════════════════════════════════════════════════════════════════════════
# suggest_format edge paths
# ══════════════════════════════════════════════════════════════════════════════


def test_suggest_format_1d_weight_reshaped():
    w = np.random.RandomState(0).randn(128).astype(np.float32)
    res = Quantine.suggest_format(sample_weight=w, quality_threshold=0.0, min_speed_ratio=0.0)
    assert "format" in res
    assert res["bits"] in (32, 8, 4)


def test_suggest_format_without_avx2(monkeypatch):
    class _Broken:
        def __getattr__(self, name):
            raise ImportError(f"cannot import name {name!r}")

    monkeypatch.setitem(sys.modules, "domains.infrastructure.quant_core.wrapper", _Broken())
    res = Quantine.suggest_format(
        sample_weight=np.ones((64, 64), dtype=np.float32),
        quality_threshold=0.0,
        min_speed_ratio=0.0,
    )
    assert "format" in res
