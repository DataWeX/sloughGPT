"""Tests for the unified ML type system (ml_types.py)."""

import sys

import numpy as np
import pytest

from domains.infrastructure import ml_types as mt


class TestDtype:
    def test_name_resolution(self):
        assert mt.dtype("float32") == np.float32
        assert mt.dtype("fp32") == np.float32
        assert mt.dtype("half") == np.float16
        assert mt.dtype("bf16") == np.float32
        assert mt.dtype("double") == np.float64
        assert mt.dtype("int") == np.int32
        assert mt.dtype("bool") == np.bool_

    def test_case_insensitive(self):
        assert mt.dtype("Float32") == np.float32

    def test_torch_prefix_stripped(self):
        assert mt.dtype("torch.float32") == np.float32
        assert mt.dtype("torch.int64") == np.int64

    def test_numpy_dtype_passthrough(self):
        assert mt.dtype(np.float16) == np.float16
        assert mt.dtype(np.dtype(np.float64)) == np.float64

    def test_class_passthrough(self):
        assert mt.dtype(np.float32) == np.float32

    def test_object_with_numpy(self):
        class FakeTorchDtype:
            def numpy(self):
                return np.float16
        assert mt.dtype(FakeTorchDtype()) == np.float16

    def test_torch_dtype_numpy_raises_falls_back(self, monkeypatch):
        class _Raises:
            def numpy(self):
                raise RuntimeError("no torch")

        real_dtype = np.dtype

        class _DTypeProxy:
            def __new__(cls, v):
                return real_dtype(np.float32)

        monkeypatch.setattr(mt.np, "dtype", _DTypeProxy)
        assert mt.dtype(_Raises()) == np.float32

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError):
            mt.dtype("not_a_dtype")


class TestDevice:
    def test_cpu(self):
        d = mt.device("cpu")
        assert d.type == "cpu"
        assert d.index is None
        assert str(d) == "cpu"

    def test_cuda_with_index(self):
        d = mt.device("cuda:0")
        assert d.type == "cuda"
        assert d.index == 0
        assert str(d) == "cuda:0"

    def test_mps_index(self):
        d = mt.device("mps:1")
        assert str(d) == "mps:1"

    def test_non_numeric_index_none(self):
        d = mt.device("cuda:x")
        assert d.index is None

    def test_repr(self):
        assert repr(mt.device("cpu")) == "device('cpu')"

    def test_eq_string(self):
        assert mt.device("cpu") == "cpu"
        assert mt.device("CPU") == "cpu"

    def test_eq_device(self):
        assert mt.device("cuda:0") == mt.device("cuda:0")
        assert mt.device("cuda:0") != mt.device("cuda:1")

    def test_eq_other(self):
        assert (mt.device("cpu") == 42) is False

    def test_hashable(self):
        assert hash(mt.device("cpu")) == hash("cpu")

    def test_copy_constructor(self):
        d = mt.device(mt.device("cuda:2"))
        assert str(d) == "cuda:2"

    def test_case_normalized(self):
        assert mt.device("CUDA:0").type == "cuda"


class TestCreation:
    def test_tensor_from_list(self):
        arr = mt.tensor([1, 2, 3])
        assert arr.dtype == np.float32
        assert arr.tolist() == [1.0, 2.0, 3.0]

    def test_tensor_with_dtype(self):
        arr = mt.tensor([1, 2], dtype=np.int64)
        assert arr.dtype == np.int64

    def test_tensor_from_ndarray_no_copy(self):
        src = np.zeros(3, dtype=np.float32)
        arr = mt.tensor(src)
        assert arr is src

    def test_tensor_from_ndarray_casts(self):
        src = np.zeros(3, dtype=np.float64)
        arr = mt.tensor(src, dtype=np.float32)
        assert arr.dtype == np.float32

    def test_zeros(self):
        assert mt.zeros((2, 3)).shape == (2, 3)
        assert mt.zeros(2).dtype == np.float32

    def test_ones(self):
        arr = mt.ones((2, 2), dtype=np.int32)
        assert arr.dtype == np.int32
        assert arr.sum() == 4

    def test_full(self):
        arr = mt.full((3,), 7.0)
        assert arr.tolist() == [7.0, 7.0, 7.0]

    def test_full_like(self):
        src = np.ones((2, 3), dtype=np.float32)
        arr = mt.full_like(src, 5.0)
        assert arr.shape == (2, 3)
        assert arr.dtype == np.float32

    def test_full_like_explicit_dtype(self):
        arr = mt.full_like(np.zeros(2), 1.0, dtype=np.float16)
        assert arr.dtype == np.float16

    def test_empty_shape_dtype(self):
        arr = mt.empty((2,), dtype=np.int16)
        assert arr.shape == (2,)
        assert arr.dtype == np.int16

    def test_randn_varargs(self):
        arr = mt.randn(2, 3)
        assert arr.shape == (2, 3)
        assert arr.dtype == np.float32

    def test_randn_tuple(self):
        arr = mt.randn((2, 3))
        assert arr.shape == (2, 3)

    def test_arange(self):
        arr = mt.arange(0, 5)
        assert arr.tolist() == [0, 1, 2, 3, 4]
        assert arr.dtype == np.int64

    def test_from_numpy_identity(self):
        src = np.zeros(2)
        assert mt.from_numpy(src) is src


class TestInspection:
    def test_isnan(self):
        arr = np.array([1.0, np.nan])
        assert mt.isnan(arr).tolist() == [False, True]

    def test_isinf(self):
        arr = np.array([1.0, np.inf])
        assert mt.isinf(arr).tolist() == [False, True]

    def test_isfinite(self):
        arr = np.array([1.0, np.nan, np.inf])
        assert mt.isfinite(arr).tolist() == [True, False, False]

    def test_numel(self):
        assert mt.numel(np.zeros((2, 3))) == 6

    def test_allclose(self):
        assert mt.allclose(np.array([1.0]), np.array([1.0]))
        assert not mt.allclose(np.array([1.0]), np.array([2.0]))

    def test_allclose_custom_tolerances(self):
        assert mt.allclose(np.array([1.0]), np.array([1.1]), rtol=0.2)
        assert not mt.allclose(np.array([1.0]), np.array([1.1]), rtol=0.01)

    def test_item(self):
        assert mt.item(np.array(3.5)) == 3.5
        assert mt.item(np.array([42])) == 42.0


class TestMathOps:
    def test_cat(self):
        a = np.array([[1], [2]])
        b = np.array([[3], [4]])
        assert mt.cat([a, b], dim=1).shape == (2, 2)

    def test_stack(self):
        a = np.array([1, 2])
        b = np.array([3, 4])
        assert mt.stack([a, b], dim=1).shape == (2, 2)

    def test_where(self):
        cond = np.array([True, False])
        out = mt.where(cond, np.array([1, 2]), np.array([9, 8]))
        assert out.tolist() == [1, 8]

    def test_topk_values_descending(self):
        arr = np.array([[3.0, 1.0, 2.0]])
        values, indices = mt.topk(arr, k=2, dim=-1)
        assert values[0].tolist() == [3.0, 2.0]
        assert indices[0].tolist() == [0, 2]

    def test_topk_dim0(self):
        arr = np.array([[1.0], [3.0], [2.0]])
        values, indices = mt.topk(arr, k=2, dim=0)
        assert values[0].tolist() == [3.0]
        assert values[1].tolist() == [2.0]

    def test_sort_ascending(self):
        arr = np.array([3, 1, 2])
        values, indices = mt.sort(arr)
        assert values.tolist() == [1, 2, 3]
        assert indices.tolist() == [1, 2, 0]

    def test_sort_descending(self):
        arr = np.array([3, 1, 2])
        values, _ = mt.sort(arr, descending=True)
        assert values.tolist() == [3, 2, 1]

    def test_clamp_min_max(self):
        arr = np.array([-1.0, 0.0, 5.0])
        assert mt.clamp(arr, min=0.0, max=1.0).tolist() == [0.0, 0.0, 1.0]

    def test_clamp_legacy_names(self):
        arr = np.array([-1.0, 5.0])
        assert mt.clamp(arr, min_val=0.0, max_val=1.0).tolist() == [0.0, 1.0]

    def test_clamp_min_only(self):
        arr = np.array([-1.0, 5.0])
        assert mt.clamp(arr, min=0.0).tolist() == [0.0, 5.0]

    def test_multinomial_counts(self):
        probs = np.array([0.5, 0.5])
        samples = mt.multinomial(probs, num_samples=1000)
        assert samples.shape == (1000,)
        assert set(np.unique(samples)) <= {0, 1}

    def test_multinomial_normalizes(self):
        probs = np.array([1.0, 3.0])
        samples = mt.multinomial(probs, num_samples=100)
        assert set(np.unique(samples)) <= {0, 1}

    def test_softmax_sums_to_one(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = mt.softmax(arr)
        assert np.isclose(out.sum(), 1.0)

    def test_softmax_preserves_input_dtype(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert mt.softmax(arr).dtype == np.float32

    def test_softmax_axis(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = mt.softmax(arr, dim=0)
        assert np.allclose(out.sum(axis=0), 1.0)

    def test_matmul(self):
        a = np.array([[1.0, 2.0]])
        b = np.array([[3.0], [4.0]])
        assert mt.matmul(a, b).tolist() == [[11.0]]

    def test_cosine_similarity_parallel(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[2.0, 0.0]])
        assert mt.cosine_similarity(a, b).tolist() == pytest.approx([1.0], abs=1e-5)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        assert mt.cosine_similarity(a, b).tolist() == pytest.approx([0.0], abs=1e-5)


class TestNoGrad:
    def test_runs_and_yields(self):
        ran = []
        with mt.no_grad():
            ran.append(True)
        assert ran == [True]

    def test_values_unchanged(self):
        x = np.array([1.0])
        with mt.no_grad():
            y = x * 2
        assert y.tolist() == [2.0]


class TestPlatform:
    def test_mps_not_on_linux(self, monkeypatch):
        monkeypatch.setattr(mt.sys, "platform", "linux")
        assert mt._mps_available() is False

    def test_mps_blocked_on_intel(self, monkeypatch):
        monkeypatch.setattr(mt.sys, "platform", "darwin")
        monkeypatch.setattr(mt.platform, "machine", lambda: "x86_64")
        assert mt._mps_available() is False

    def test_mps_true_with_fake_torch(self, monkeypatch):
        monkeypatch.setattr(mt.sys, "platform", "darwin")
        monkeypatch.setattr(mt.platform, "machine", lambda: "arm64")
        fake = type("M", (), {})()
        fake.backends = type("B", (), {"mps": type("M2", (), {"is_available": staticmethod(lambda: True)})()})()
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert mt._mps_available() is True

    def test_mps_false_when_torch_says_no(self, monkeypatch):
        monkeypatch.setattr(mt.sys, "platform", "darwin")
        monkeypatch.setattr(mt.platform, "machine", lambda: "arm64")
        fake = type("M", (), {})()
        fake.backends = type("B", (), {"mps": type("M2", (), {"is_available": staticmethod(lambda: False)})()})()
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert mt._mps_available() is False

    def test_mps_fallback_arm64(self, monkeypatch):
        monkeypatch.setattr(mt.sys, "platform", "darwin")
        monkeypatch.setattr(mt.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(mt.platform, "system", lambda: "Darwin")
        monkeypatch.setitem(sys.modules, "torch", None)
        assert mt._mps_available() is True

    def test_mps_fallback_non_arm64(self, monkeypatch):
        monkeypatch.setattr(mt.sys, "platform", "darwin")
        monkeypatch.setattr(mt.platform, "machine", lambda: "ppc64")
        monkeypatch.setattr(mt.platform, "system", lambda: "Darwin")
        monkeypatch.setitem(sys.modules, "torch", None)
        assert mt._mps_available() is False

    def test_mps_stub_methods(self, monkeypatch):
        monkeypatch.setattr(mt, "_mps_available", lambda: False)
        assert mt.mps.is_available() is False
        assert mt.mps.empty_cache() is None

    def test_cuda_stub_methods(self, monkeypatch):
        monkeypatch.setattr(mt, "_cuda_available", lambda: False)
        assert mt.cuda.is_available() is False
        assert mt.cuda.empty_cache() is None

    def test_cuda_unavailable_without_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        assert mt._cuda_available() is False

    def test_cuda_available_with_fake_torch(self, monkeypatch):
        fake = type("M", (), {})()
        fake.cuda = type("C", (), {"is_available": staticmethod(lambda: True)})()
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert mt._cuda_available() is True

    def test_auto_device_cpu(self, monkeypatch):
        monkeypatch.setattr(mt, "_mps_available", lambda: False)
        monkeypatch.setattr(mt, "_cuda_available", lambda: False)
        assert mt.auto_device() == "cpu"

    def test_auto_device_mps_priority(self, monkeypatch):
        monkeypatch.setattr(mt, "_mps_available", lambda: True)
        monkeypatch.setattr(mt, "_cuda_available", lambda: True)
        assert mt.auto_device() == "mps"

    def test_auto_device_cuda(self, monkeypatch):
        monkeypatch.setattr(mt, "_mps_available", lambda: False)
        monkeypatch.setattr(mt, "_cuda_available", lambda: True)
        assert mt.auto_device() == "cuda"
