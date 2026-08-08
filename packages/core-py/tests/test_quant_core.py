"""Tests for quant_core.wrapper — AVX2 int8/int4 GEMM with numpy fallback."""

import ctypes
import importlib
import os
import subprocess
import sys
import time
import types

import numpy as np
import pytest

from domains.infrastructure import quant_core
from domains.infrastructure.quant_core import wrapper
from domains.infrastructure.quant_core.wrapper import (
    HAS_AVX2,
    _build_all,
    _build_one,
    _fallback,
    _fallback_int4,
    _load_lib,
    matmul_int4_c,
    matmul_int8_c,
    matmul_int8_f32_c,
)
from domains.infrastructure.quantization import _pack_int4


def _int8(M, K, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(-30, 30, size=(M, K)).astype(np.int8)


def _int4_rows(N, K, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(-8, 8, size=(N, K)).astype(np.int8)


def _packed_rows(B):
    return np.stack([_pack_int4(row) for row in B]).astype(np.uint8)


def _ref_int8(A, B):
    return np.matmul(A.astype(np.int32), B.astype(np.int32).T)


def _ref_int4(A, B):
    return np.matmul(A.astype(np.int32), B.astype(np.int32).T)


class TestFallbacks:
    def test_fallback_int8_matches_reference(self):
        A, B = _int8(3, 8), _int8(5, 8)
        np.testing.assert_array_equal(_fallback(A, B), _ref_int8(A, B))

    def test_fallback_int4_matches_reference(self):
        A, B = _int8(3, 8), _int4_rows(5, 8)
        packed = _packed_rows(B)
        np.testing.assert_array_equal(_fallback_int4(A, packed, 8), _ref_int4(A, B))

    def test_fallback_int4_odd_k_dimension(self):
        K = 6
        A, B = _int8(2, K), _int4_rows(4, K)
        packed = _packed_rows(B)
        assert packed.shape == (4, K // 2)
        np.testing.assert_array_equal(_fallback_int4(A, packed, K), _ref_int4(A, B))


class TestMatmulWithForcedFallback:
    def test_matmul_int8_c_uses_fallback(self, monkeypatch):
        monkeypatch.setattr(wrapper, "_load_lib", lambda: False)
        A, B = _int8(3, 8), _int8(5, 8)
        np.testing.assert_array_equal(matmul_int8_c(A, B), _ref_int8(A, B))

    def test_matmul_int4_c_uses_fallback(self, monkeypatch):
        monkeypatch.setattr(wrapper, "_load_lib", lambda: False)
        A, B = _int8(3, 8), _int4_rows(5, 8)
        packed = _packed_rows(B)
        np.testing.assert_array_equal(matmul_int4_c(A, packed, 8), _ref_int4(A, B))

    def test_matmul_int8_c_force_native_missing(self, monkeypatch):
        # Even when a lib object exists, a missing matmul_int4 symbol must fall back
        fake_lib = object()
        monkeypatch.setattr(wrapper, "_LIB", fake_lib)
        monkeypatch.setattr(wrapper, "_load_lib", lambda: True)
        A, B = _int8(2, 4), _int4_rows(3, 4)
        np.testing.assert_array_equal(matmul_int4_c(A, _packed_rows(B), 4), _ref_int4(A, B))


@pytest.mark.skipif(not HAS_AVX2, reason="AVX2 C library not compiled in this environment")
class TestNativePath:
    def test_native_int8_matches_reference(self):
        A, B = _int8(3, 8), _int8(5, 8)
        np.testing.assert_array_equal(matmul_int8_c(A, B), _ref_int8(A, B))

    def test_native_int8_matches_numpy_fallback(self):
        A, B = _int8(4, 8), _int8(6, 8)
        np.testing.assert_array_equal(matmul_int8_c(A, B), _fallback(A, B))

    def test_native_int8_shape_mismatch_raises(self):
        A, B = _int8(3, 8), _int8(5, 6)
        with pytest.raises(AssertionError):
            matmul_int8_c(A, B)

    def test_native_int4_matches_reference(self):
        A, B = _int8(3, 8), _int4_rows(5, 8)
        packed = _packed_rows(B)
        np.testing.assert_array_equal(matmul_int4_c(A, packed, 8), _ref_int4(A, B))

    def test_native_int4_shape_mismatch_raises(self):
        A, B = _int8(3, 8), _int4_rows(5, 6)
        packed = _packed_rows(B)
        with pytest.raises(AssertionError):
            matmul_int4_c(A, packed, 8)


@pytest.mark.skipif(not HAS_AVX2, reason="AVX2 C library not compiled in this environment")
class TestNativeBlocking:
    """Exercises j-blocking (N spanning multiple 256KB blocks) and both int4 paths."""

    def test_int8_large_n_multi_block(self):
        # K=768 → jblock=341 rows; N=2000 forces ~6 B-blocks.
        M, N, K = 3, 2000, 768
        A = _int8(M, K, seed=10)
        B = _int8(N, K, seed=11)
        np.testing.assert_array_equal(matmul_int8_c(A, B), _ref_int8(A, B))

    def test_int8_many_rows(self):
        # M=128 exercises row reuse across a j-block (the failing 2.15x case).
        M, N, K = 128, 768, 768
        A = _int8(M, K, seed=20)
        B = _int8(N, K, seed=21)
        np.testing.assert_array_equal(matmul_int8_c(A, B), _ref_int8(A, B))

    def test_int8_odd_dimensions_scalar_remainder(self):
        M, N, K = 3, 100, 63
        A = _int8(M, K, seed=25)
        B = _int8(N, K, seed=26)
        np.testing.assert_array_equal(matmul_int8_c(A, B), _ref_int8(A, B))

    def test_int4_m1_inline_path(self):
        # M == 1 uses the register-resident inline unpack path.
        M, N, K = 1, 4096, 896
        A = _int8(M, K, seed=30)
        B = _int4_rows(N, K, seed=31)
        np.testing.assert_array_equal(
            matmul_int4_c(A, _packed_rows(B), K), _ref_int4(A, B))

    def test_int4_many_rows_scratch_path(self):
        # M > 1 uses the unpack-once-per-block scratch path.
        M, N, K = 16, 1500, 768
        A = _int8(M, K, seed=40)
        B = _int4_rows(N, K, seed=41)
        np.testing.assert_array_equal(
            matmul_int4_c(A, _packed_rows(B), K), _ref_int4(A, B))

    def test_int4_large_n_multi_block(self):
        # K=768 → packed jblock=682 rows; N=4864 forces ~7 B-blocks.
        M, N, K = 5, 4864, 896
        A = _int8(M, K, seed=50)
        B = _int4_rows(N, K, seed=51)
        np.testing.assert_array_equal(
            matmul_int4_c(A, _packed_rows(B), K), _ref_int4(A, B))


@pytest.mark.skipif(not HAS_AVX2, reason="AVX2 C library not compiled in this environment")
class TestFusedInt8:
    """Fused per-token quantize + int8 GEMM + dequantize + bias.

    The fused kernel must reproduce the unfused Python path bit-for-bit so
    replacing one with the other is a pure speed change with no numerical
    impact on generated output.
    """

    def _unfused(self, monkeypatch, x, w, bscale, bias, **kw):
        import domains.infrastructure.quantization as Q

        monkeypatch.setattr(Q, "matmul_int8_f32_c", None)
        return Q.quantized_linear(x, w, bscale, 0, bias, **kw)

    def _fused(self, x, w, bscale, bias, **kw):
        import domains.infrastructure.quantization as Q

        return Q.quantized_linear(x, w, bscale, 0, bias, **kw)

    def _weights(self, M, N, K, seed=100):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((M, K)).astype(np.float32)
        w = rng.integers(-128, 128, size=(N, K)).astype(np.int8)
        bias = rng.standard_normal(N).astype(np.float32)
        return x, w, bias

    def test_fused_matches_unfused_per_tensor(self, monkeypatch):
        x, w, b = self._weights(1, 100, 63, seed=1)
        np.testing.assert_array_equal(
            self._fused(x, w, 0.0125, b),
            self._unfused(monkeypatch, x, w, 0.0125, b))

    def test_fused_matches_unfused_numpy_float_scale(self, monkeypatch):
        # np.float32 scalars must take the per-tensor path (regression: they
        # were misdetected as per-row scales and rejected).
        x, w, b = self._weights(1, 100, 63, seed=1)
        np.testing.assert_array_equal(
            self._fused(x, w, np.float32(0.0125), b),
            self._unfused(monkeypatch, x, w, np.float32(0.0125), b))

    def test_fused_matches_unfused_per_row(self, monkeypatch):
        x, w, b = self._weights(1, 2000, 768, seed=2)
        bscale = np.linspace(0.001, 0.05, 2000).astype(np.float32)
        np.testing.assert_array_equal(
            self._fused(x, w, bscale, b),
            self._unfused(monkeypatch, x, w, bscale, b))

    def test_fused_matches_unfused_multi_row(self, monkeypatch):
        # M > 1 exercises per-row activation scales (one scale per token).
        x, w, b = self._weights(5, 896, 4864, seed=3)
        bscale = np.linspace(0.001, 0.05, 896).astype(np.float32)
        np.testing.assert_array_equal(
            self._fused(x, w, bscale, b),
            self._unfused(monkeypatch, x, w, bscale, b))

    def test_fused_matches_unfused_large_n(self, monkeypatch):
        # lm_head shape (N=151936 spans ~520 j-blocks).
        x, w, b = self._weights(1, 151936, 896, seed=4)
        bscale = np.linspace(0.001, 0.05, 151936).astype(np.float32)
        np.testing.assert_array_equal(
            self._fused(x, w, bscale, b),
            self._unfused(monkeypatch, x, w, bscale, b))

    def test_fused_odd_dimensions(self, monkeypatch):
        x, w, b = self._weights(3, 100, 63, seed=5)
        np.testing.assert_array_equal(
            self._fused(x, w, 0.01, b),
            self._unfused(monkeypatch, x, w, 0.01, b))

    def test_fused_no_bias(self, monkeypatch):
        x, w, _ = self._weights(1, 100, 63, seed=6)
        np.testing.assert_array_equal(
            self._fused(x, w, 0.01, None),
            self._unfused(monkeypatch, x, w, 0.01, None))

    def test_fused_zero_activation_rows(self, monkeypatch):
        # An all-zero row must fall back to scale 1.0 → int8 zeros → output = bias.
        x = np.zeros((2, 63), dtype=np.float32)
        w = np.ones((100, 63), dtype=np.int8) * 7
        b = np.arange(100, dtype=np.float32)
        np.testing.assert_array_equal(
            self._fused(x, w, 0.01, b),
            self._unfused(monkeypatch, x, w, 0.01, b))

    def test_fused_3d_input(self, monkeypatch):
        # (..., K) leading dims preserved by the reshape/restore.
        x, w, b = self._weights(1, 100, 63, seed=7)
        x3 = np.stack([x, x * 2.0])
        np.testing.assert_array_equal(
            self._fused(x3, w, 0.01, b),
            self._unfused(monkeypatch, x3, w, 0.01, b))

    def test_asymmetric_input_skips_fused(self, monkeypatch):
        import domains.infrastructure.quantization as Q

        def _boom(*args, **kw):
            raise AssertionError("fused kernel must not run for asymmetric input")

        monkeypatch.setattr(Q, "matmul_int8_f32_c", _boom)
        x, w, b = self._weights(1, 100, 63, seed=8)
        out = Q.quantized_linear(x, w, 0.01, 0, b, x_zero_point=1)
        assert np.isfinite(out).all()
        assert out.shape == (1, 100)

    def test_fixed_x_scale_skips_fused(self, monkeypatch):
        import domains.infrastructure.quantization as Q

        def _boom(*args, **kw):
            raise AssertionError("fused kernel must not run when x_scale is given")

        monkeypatch.setattr(Q, "matmul_int8_f32_c", _boom)
        x, w, b = self._weights(1, 100, 63, seed=9)
        out = Q.quantized_linear(x, w, 0.01, 0, b, x_scale=0.02)
        assert np.isfinite(out).all()
        assert out.shape == (1, 100)


@pytest.mark.skipif(not HAS_AVX2, reason="AVX2 C library not compiled in this environment")
class TestThreading:
    """Threaded j-block slicing must stay bit-identical to serial output.

    Threads partition B into contiguous column slices (disjoint C columns);
    the per-column k-order is unchanged, so results must match exactly.
    The C kernel reads MAN_GEMM_THREADS from the process environment per call.
    """

    def _set_threads(self, n):
        if n <= 1:
            os.environ.pop("MAN_GEMM_THREADS", None)
        else:
            os.environ["MAN_GEMM_THREADS"] = str(n)

    def test_int8_threaded_matches_serial(self):
        M, N, K = 3, 8192, 896  # N*K = 7.3 MB > _THREAD_MIN_BYTES
        A = _int8(M, K, seed=70)
        B = _int8(N, K, seed=71)
        self._set_threads(1)
        serial = matmul_int8_c(A, B)
        self._set_threads(8)
        threaded = matmul_int8_c(A, B)
        np.testing.assert_array_equal(threaded, serial)

    def test_int8_many_rows_threaded_matches_serial(self):
        M, N, K = 128, 8192, 896
        A = _int8(M, K, seed=72)
        B = _int8(N, K, seed=73)
        self._set_threads(1)
        serial = matmul_int8_c(A, B)
        self._set_threads(8)
        threaded = matmul_int8_c(A, B)
        np.testing.assert_array_equal(threaded, serial)

    def test_fused_threaded_matches_serial(self):
        import domains.infrastructure.quantization as Q

        M, N, K = 5, 8192, 896  # lm_head-like fused shape above the threshold
        rng = np.random.default_rng(74)
        x = rng.standard_normal((M, K)).astype(np.float32)
        w = rng.integers(-128, 128, size=(N, K)).astype(np.int8)
        bscale = np.linspace(0.001, 0.05, N).astype(np.float32)
        bias = rng.standard_normal(N).astype(np.float32)
        self._set_threads(1)
        serial = Q.quantized_linear(x, w, bscale, 0, bias)
        self._set_threads(8)
        threaded = Q.quantized_linear(x, w, bscale, 0, bias)
        np.testing.assert_array_equal(threaded, serial)

    def test_fused_small_gemm_stays_serial_path(self):
        # Below the thread threshold the serial loop must still match reference.
        import domains.infrastructure.quantization as Q

        M, N, K = 1, 100, 63
        rng = np.random.default_rng(75)
        x = rng.standard_normal((M, K)).astype(np.float32)
        w = rng.integers(-128, 128, size=(N, K)).astype(np.int8)
        self._set_threads(8)
        out = Q.quantized_linear(x, w, 0.01, 0, None)
        self._set_threads(1)
        serial = Q.quantized_linear(x, w, 0.01, 0, None)
        np.testing.assert_array_equal(out, serial)


class TestBuild:
    def test_HAS_AVX2_is_bool(self):
        assert isinstance(HAS_AVX2, bool)

    def test_build_one_missing_source_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(wrapper, "_SRCS", {"matmul_int8": str(tmp_path / "nope.c")})
        assert _build_one("matmul_int8") is False

    def test_build_one_gcc_failure_returns_false(self, monkeypatch, tmp_path):
        src = tmp_path / "matmul_int8.c"
        src.write_text("int main(void){return 0;}")

        class _Result:
            returncode = 1
            stderr = "compile error"

        monkeypatch.setattr(wrapper, "_SRCS", {"matmul_int8": str(src)})
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
        assert _build_one("matmul_int8") is False

    def test_build_one_gcc_missing_returns_false(self, monkeypatch, tmp_path):
        src = tmp_path / "matmul_int8.c"
        src.write_text("int main(void){return 0;}")

        def _no_gcc(*a, **k):
            raise FileNotFoundError("gcc")

        monkeypatch.setattr(wrapper, "_SRCS", {"matmul_int8": str(src)})
        monkeypatch.setattr(subprocess, "run", _no_gcc)
        assert _build_one("matmul_int8") is False

    def test_build_one_success(self, monkeypatch, tmp_path):
        src = tmp_path / "matmul_int8.c"
        dylib = tmp_path / "matmul_int8.so"
        src.write_text("int main(void){return 0;}")

        class _Result:
            returncode = 0
            stderr = ""

        monkeypatch.setattr(wrapper, "_SRCS", {"matmul_int8": str(src)})
        monkeypatch.setattr(wrapper, "_DYLIBS", {"matmul_int8": str(dylib)})
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
        assert _build_one("matmul_int8") is True

    def test_build_one_generic_exception_returns_false(self, monkeypatch, tmp_path):
        src = tmp_path / "matmul_int8.c"
        src.write_text("int main(void){return 0;}")

        def _boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(wrapper, "_SRCS", {"matmul_int8": str(src)})
        monkeypatch.setattr(subprocess, "run", _boom)
        assert _build_one("matmul_int8") is False

    def test_build_all_oserror_on_mtime_returns_true(self, monkeypatch, tmp_path):
        dylib = tmp_path / "matmul_int8.so"
        src = tmp_path / "matmul_int8.c"
        dylib.touch()
        src.touch()

        def _boom(path):
            raise OSError("stat failed")

        monkeypatch.setattr(wrapper, "_DYLIBS", {"matmul_int8": str(dylib)})
        monkeypatch.setattr(wrapper, "_SRCS", {"matmul_int8": str(src)})
        monkeypatch.setattr(os.path, "getmtime", _boom)
        assert _build_all() is True

    def test_build_all_true_when_libs_present(self, monkeypatch, tmp_path):
        lib1 = tmp_path / "matmul_int8.so"
        lib2 = tmp_path / "matmul_int4.so"
        lib1.touch()
        lib2.touch()
        monkeypatch.setattr(wrapper, "_DYLIBS", {
            "matmul_int8": str(lib1),
            "matmul_int4": str(lib2),
        })
        assert _build_all() is True

    def test_build_all_rebuilds_when_source_newer(self, monkeypatch, tmp_path):
        dylib = tmp_path / "matmul_int8.so"
        src = tmp_path / "matmul_int8.c"
        dylib.touch()
        src.touch()
        os.utime(str(src), (time.time() + 60, time.time() + 60))  # newer than lib
        built = []
        monkeypatch.setattr(wrapper, "_DYLIBS", {"matmul_int8": str(dylib)})
        monkeypatch.setattr(wrapper, "_SRCS", {"matmul_int8": str(src)})
        monkeypatch.setattr(wrapper, "_build_one",
                            lambda name: built.append(name) or True)
        assert _build_all() is True
        assert built == ["matmul_int8"]

    def test_build_all_skips_when_source_not_newer(self, monkeypatch, tmp_path):
        dylib = tmp_path / "matmul_int8.so"
        src = tmp_path / "matmul_int8.c"
        dylib.touch()
        src.touch()
        os.utime(str(dylib), (time.time() + 60, time.time() + 60))  # newer than src
        built = []
        monkeypatch.setattr(wrapper, "_DYLIBS", {"matmul_int8": str(dylib)})
        monkeypatch.setattr(wrapper, "_SRCS", {"matmul_int8": str(src)})
        monkeypatch.setattr(wrapper, "_build_one",
                            lambda name: built.append(name) or True)
        assert _build_all() is True
        assert built == []

    def test_build_all_force_rebuilds(self, monkeypatch, tmp_path):
        dylib = tmp_path / "matmul_int8.so"
        src = tmp_path / "matmul_int8.c"
        dylib.touch()
        src.touch()
        os.utime(str(dylib), (time.time() + 60, time.time() + 60))
        built = []
        monkeypatch.setattr(wrapper, "_DYLIBS", {"matmul_int8": str(dylib)})
        monkeypatch.setattr(wrapper, "_SRCS", {"matmul_int8": str(src)})
        monkeypatch.setattr(wrapper, "_build_one",
                            lambda name: built.append(name) or True)
        assert _build_all(force=True) is True
        assert built == ["matmul_int8"]

    def test_build_all_false_without_libs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(wrapper, "_DYLIBS", {
            "matmul_int8": str(tmp_path / "missing1.so"),
            "matmul_int4": str(tmp_path / "missing2.so"),
        })
        monkeypatch.setattr(wrapper, "_SRCS", {
            "matmul_int8": str(tmp_path / "missing1.c"),
            "matmul_int4": str(tmp_path / "missing2.c"),
        })
        assert _build_all() is False

    def test_load_lib_idempotent(self, monkeypatch):
        monkeypatch.setattr(wrapper, "_LIB", object())
        assert _load_lib() is True

    def test_load_lib_no_library_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(wrapper, "_LIB", None)
        monkeypatch.setattr(wrapper, "_SRCS", {
            "matmul_int8": str(tmp_path / "missing1.c"),
            "matmul_int4": str(tmp_path / "missing4.c"),
        })
        monkeypatch.setattr(wrapper, "_DYLIBS", {
            "matmul_int8": str(tmp_path / "missing1.so"),
            "matmul_int4": str(tmp_path / "missing4.so"),
        })
        assert _load_lib() is False

    def test_matmul_int8_f32_c_missing_symbol_returns_none(self, monkeypatch):
        monkeypatch.setattr(wrapper, "_load_lib", lambda: True)
        monkeypatch.setattr(wrapper, "_LIB", object())
        A = np.ones((2, 4), dtype=np.float32)
        B = np.ones((3, 4), dtype=np.int8)
        assert matmul_int8_f32_c(A, B, 0.1) is None


class TestPlatformExtension:
    def test_darwin_uses_dylib_extension(self, monkeypatch):
        def _no_gcc(*a, **k):
            raise FileNotFoundError("gcc")

        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(subprocess, "run", _no_gcc)
        importlib.reload(wrapper)
        try:
            assert wrapper._EXT == ".dylib"
            assert wrapper.HAS_AVX2 is False
        finally:
            monkeypatch.undo()
            importlib.reload(wrapper)
        assert wrapper._EXT == ".so"
        assert wrapper.HAS_AVX2 is True

    def test_win32_uses_dll_extension(self, monkeypatch):
        def _no_gcc(*a, **k):
            raise FileNotFoundError("gcc")

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(subprocess, "run", _no_gcc)
        importlib.reload(wrapper)
        try:
            assert wrapper._EXT == ".dll"
            assert wrapper.HAS_AVX2 is False
        finally:
            monkeypatch.undo()
            importlib.reload(wrapper)
        assert wrapper._EXT == ".so"
        assert wrapper.HAS_AVX2 is True

    def test_int8_only_load_logs_int8_only(self, monkeypatch):
        fake = types.SimpleNamespace(
            matmul_int8=types.SimpleNamespace(argtypes=None, restype=None),
            matmul_int8_f32=types.SimpleNamespace(argtypes=None, restype=None),
        )
        calls = []

        def _fake_cdll(path):
            calls.append(path)
            if len(calls) == 1:
                return fake
            raise OSError("matmul_int4.so not available")

        monkeypatch.setattr(ctypes, "CDLL", _fake_cdll)
        importlib.reload(wrapper)
        try:
            assert wrapper.HAS_AVX2 is True
            assert not hasattr(wrapper._LIB, "matmul_int4")
        finally:
            monkeypatch.undo()
            importlib.reload(wrapper)
        assert wrapper.HAS_AVX2 is True
        assert hasattr(wrapper._LIB, "matmul_int4")
