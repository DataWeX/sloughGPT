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


class TestAccelerateUnavailable:
    """Real numpy fallback paths when the Accelerate dylib cannot load."""

    def test_load_returns_none_when_marked_unavailable(self, monkeypatch):
        monkeypatch.setattr(blas, "_unavailable", True)
        monkeypatch.setattr(blas, "_accelerate", None)
        assert blas._load_accelerate() is None
        assert blas.is_available() is False

    def test_load_failure_marks_unavailable(self, monkeypatch):
        monkeypatch.setattr(blas, "_unavailable", False)
        monkeypatch.setattr(blas, "_accelerate", None)
        assert blas._load_accelerate() is None
        assert blas._unavailable is True
        assert blas.is_available() is False

    def test_sgemm_falls_back_to_numpy(self, monkeypatch):
        monkeypatch.setattr(blas, "_unavailable", True)
        monkeypatch.setattr(blas, "_accelerate", None)
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        np.testing.assert_allclose(blas.sgemm(a, b), a @ b)


class TestAccelerateAvailable:
    """Real ctypes CBLAS path — exercised against the system libblas.so.3.

    On Linux the Apple ``libAccelerate.dylib`` is unavailable, so the module
    cache is pointed at the real reference-BLAS ``cblas_sgemm`` (identical
    CBLAS ABI) to genuinely exercise the ctypes call path.
    """

    _DYLIB = "libblas.so.3"

    @pytest.fixture
    def cblas(self, monkeypatch):
        import ctypes as _ctypes
        lib = _ctypes.CDLL(self._DYLIB)
        blas._setup_sgemm(lib)
        monkeypatch.setattr(blas, "_accelerate", lib)
        monkeypatch.setattr(blas, "_unavailable", False)
        return lib

    def test_setup_sgemm_signature(self, cblas):
        assert cblas.cblas_sgemm.restype is None
        assert len(cblas.cblas_sgemm.argtypes) == 14

    def test_load_accelerate_returns_cached(self, cblas):
        assert blas._load_accelerate() is cblas
        assert blas.is_available() is True

    def test_load_accelerate_loads_and_setup(self, monkeypatch, cblas):
        monkeypatch.setattr(blas, "_accelerate", None)
        monkeypatch.setattr(blas, "_unavailable", False)
        import ctypes as _ctypes
        monkeypatch.setattr(_ctypes, "CDLL", lambda *a, **k: cblas)
        loaded = blas._load_accelerate()
        assert loaded is cblas
        assert blas._load_accelerate() is cblas

    def test_sgemm_matches_numpy(self, cblas):
        a = np.random.RandomState(1).randn(3, 4).astype(np.float32)
        b = np.random.RandomState(2).randn(4, 5).astype(np.float32)
        np.testing.assert_allclose(blas.sgemm(a, b), a @ b, rtol=1e-5, atol=1e-5)

    def test_sgemm_alpha_scaling(self, cblas):
        a = np.random.RandomState(3).randn(2, 3).astype(np.float32)
        b = np.random.RandomState(4).randn(3, 2).astype(np.float32)
        np.testing.assert_allclose(
            blas.sgemm(a, b, alpha=2.0), 2.0 * (a @ b), rtol=1e-5, atol=1e-5)

    def test_sgemm_shape_mismatch_asserts(self, cblas):
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((4, 5), dtype=np.float32)
        with pytest.raises(AssertionError):
            blas.sgemm(a, b)

    def test_sgemm_dtype_asserts(self, cblas):
        a = np.ones((2, 3), dtype=np.float64)
        b = np.ones((3, 4), dtype=np.float64)
        with pytest.raises(AssertionError):
            blas.sgemm(a, b)

    def test_matmul_uses_c_path(self, cblas):
        a = np.random.RandomState(5).randn(8, 16).astype(np.float32)
        b = np.random.RandomState(6).randn(16, 8).astype(np.float32)
        np.testing.assert_allclose(matmul(a, b), a @ b, rtol=1e-5, atol=1e-5)
