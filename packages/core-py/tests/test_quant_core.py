"""Tests for quant_core.wrapper — AVX2 int8/int4 GEMM with numpy fallback."""

import os
import subprocess

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


class TestBuild:
    def test_HAS_AVX2_is_bool(self):
        assert isinstance(HAS_AVX2, bool)

    def test_build_one_missing_source_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(wrapper, "_SRCS", {"matmul_int8": str(tmp_path / "nope.c")})
        assert _build_one("matmul_int8") is False

    def test_build_one_gcc_failure_returns_false(self, monkeypatch):
        class _Result:
            returncode = 1
            stderr = "compile error"

        monkeypatch.setattr(wrapper, "_SRCS", {"matmul_int8": "/tmp/does-not-matter.c"})
        monkeypatch.setattr(wrapper, "subprocess", subprocess)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
        assert _build_one("matmul_int8") is False

    def test_build_one_gcc_missing_returns_false(self, monkeypatch):
        def _no_gcc(*a, **k):
            raise FileNotFoundError("gcc")

        monkeypatch.setattr(wrapper, "_SRCS", {"matmul_int8": "/tmp/does-not-matter.c"})
        monkeypatch.setattr(subprocess, "run", _no_gcc)
        assert _build_one("matmul_int8") is False

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
