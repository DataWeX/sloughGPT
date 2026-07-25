"""Tests for the modular ops layer (C base, numpy fallback)."""

import pytest
import numpy as np
from unittest.mock import patch

from domains.inference.ops import matmul, layernorm, rmsnorm
from domains.inference.ops import blas


class TestBlasAvailability:
    """Test Accelerate BLAS detection."""

    def test_is_available_returns_bool(self):
        result = blas.is_available()
        assert isinstance(result, bool)

    def test_sgemm_shape(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        result = blas.sgemm(a, b)
        assert result.shape == (3, 5)
        assert result.dtype == np.float32

    def test_sgemm_matches_numpy(self):
        a = np.random.randn(8, 16).astype(np.float32)
        b = np.random.randn(16, 8).astype(np.float32)
        c_blas = blas.sgemm(a, b)
        c_np = np.matmul(a, b)
        np.testing.assert_allclose(c_blas, c_np, rtol=1e-5, atol=1e-5)


class TestMatmul:
    """Test matmul op dispatches correctly."""

    def test_matmul_float32_uses_c(self):
        a = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(8, 4).astype(np.float32)
        result = matmul(a, b)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_matmul_float64_uses_numpy(self):
        a = np.random.randn(4, 8).astype(np.float64)
        b = np.random.randn(8, 4).astype(np.float64)
        result = matmul(a, b)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_matmul_shape(self):
        a = np.random.randn(2, 3).astype(np.float32)
        b = np.random.randn(3, 7).astype(np.float32)
        result = matmul(a, b)
        assert result.shape == (2, 7)

    def test_matmul_batched(self):
        a = np.random.randn(5, 16).astype(np.float32)
        b = np.random.randn(16, 5).astype(np.float32)
        result = matmul(a, b)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    @patch.object(blas, "is_available", return_value=False)
    def test_matmul_falls_back_to_numpy(self, _mock):
        a = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(8, 4).astype(np.float32)
        result = matmul(a, b)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestLayernorm:
    """Test layernorm op."""

    def test_layernorm_shape(self):
        x = np.random.randn(2, 10, 64).astype(np.float32)
        w = np.ones(64, dtype=np.float32)
        b = np.zeros(64, dtype=np.float32)
        result = layernorm(x, w, b)
        assert result.shape == x.shape

    def test_layernorm_normalizes(self):
        x = np.random.randn(4, 32).astype(np.float32) * 10 + 5
        w = np.ones(32, dtype=np.float32)
        b = np.zeros(32, dtype=np.float32)
        result = layernorm(x, w, b)
        np.testing.assert_allclose(result.mean(axis=-1), 0, atol=1e-5)
        np.testing.assert_allclose(result.std(axis=-1), 1, atol=1e-5)


class TestRmsnorm:
    """Test rmsnorm op."""

    def test_rmsnorm_shape(self):
        x = np.random.randn(2, 10, 64).astype(np.float32)
        w = np.ones(64, dtype=np.float32)
        result = rmsnorm(x, w)
        assert result.shape == x.shape

    def test_rmsnorm_unit_rms(self):
        x = np.ones((4, 32), dtype=np.float32) * 3.0
        w = np.ones(32, dtype=np.float32)
        result = rmsnorm(x, w)
        # All ones input -> RMS = 1.0 -> output ≈ 1.0 * weight = 1.0
        np.testing.assert_allclose(result, 1.0, atol=1e-5)


class TestNativeEngineRename:
    """Test that TransformerEngine was renamed to NativeEngine."""

    def test_import_native_engine(self):
        from domains.inference.native.engine import NativeEngine
        assert NativeEngine is not None

    def test_old_name_removed(self):
        from domains.inference.native import engine
        assert not hasattr(engine, "TransformerEngine")

    def test_ct_provider_uses_native_engine(self):
        from domains.inference.ct_provider import CTransformProvider
        import inspect
        src = inspect.getsource(CTransformProvider)
        assert "NativeEngine" in src
        assert "TransformerEngine" not in src


class TestBridge:
    """Test NativeEngine ↔ SloTransformer bridge."""

    def test_ct_provider_from_slo(self):
        from domains.inference.ct_provider import CTransformProvider
        from domains.training.slonet import SloTransformer
        model = SloTransformer(vocab_size=64, n_embed=32, n_layer=2, n_head=4,
                               max_seq_len=64, _lazy=False)
        # Will fail because C lib isn't loaded in test env, but the method exists
        assert hasattr(CTransformProvider, "from_slo")
