"""Tests for domains.infrastructure.quantization — QuantMode, QuantDtype, QuantMeta, TensorInfo."""

import numpy as np
from domains.infrastructure.quantization import QuantMode, QuantDtype, QuantMeta, TensorInfo


class TestQuantMode:
    def test_all_members(self):
        assert len(QuantMode) == 2
    def test_values(self):
        assert QuantMode.SYMMETRIC.value == "symmetric"
        assert QuantMode.ASYMMETRIC.value == "asymmetric"


class TestQuantDtype:
    def test_all_members(self):
        assert len(QuantDtype) == 3
    def test_values(self):
        assert QuantDtype.INT8.value == "int8"
        assert QuantDtype.UINT8.value == "uint8"
        assert QuantDtype.INT4.value == "int4"


class TestQuantMeta:
    def test_fields(self):
        qm = QuantMeta(scale=0.5, zero_point=0, bits=8, mode="symmetric",
                        dtype_code=np.dtype("int8").num, original_shape=(3, 4),
                        original_dtype="float32")
        assert qm.scale == 0.5
        assert qm.bits == 8
        assert qm.cosine_sim == 1.0

    def test_is_per_channel(self):
        qm = QuantMeta(scale=np.array([0.1, 0.2]), zero_point=0, bits=8,
                        mode="symmetric", dtype_code=np.dtype("int8").num,
                        original_shape=(2, 3), original_dtype="float32")
        assert qm.is_per_channel is True

    def test_not_per_channel(self):
        qm = QuantMeta(scale=0.5, zero_point=0, bits=8, mode="symmetric",
                        dtype_code=np.dtype("int8").num, original_shape=(3,),
                        original_dtype="float32")
        assert qm.is_per_channel is False

    def test_to_dict(self):
        qm = QuantMeta(scale=0.5, zero_point=0, bits=8, mode="symmetric",
                        dtype_code=np.dtype("int8").num, original_shape=(3,),
                        original_dtype="float32")
        d = qm.to_dict()
        assert isinstance(d, dict)
        assert d["scale"] == 0.5
        assert d["bits"] == 8

    def test_to_dict_per_channel(self):
        qm = QuantMeta(scale=np.array([0.1, 0.2]), zero_point=0, bits=8,
                        mode="symmetric", dtype_code=np.dtype("int8").num,
                        original_shape=(2, 3), original_dtype="float32")
        d = qm.to_dict()
        assert isinstance(d["scale"], list)

    def test_from_dict(self):
        d = {"scale": 0.5, "zero_point": 0, "bits": 8, "mode": "symmetric",
             "dtype_code": 1, "original_shape": [3], "original_dtype": "float32"}
        qm = QuantMeta.from_dict(d)
        assert qm.scale == 0.5
        assert qm.bits == 8

    def test_from_dict_list_scale(self):
        d = {"scale": [0.1, 0.2], "zero_point": 0, "bits": 8, "mode": "symmetric",
             "dtype_code": 1, "original_shape": [2, 3], "original_dtype": "float32"}
        qm = QuantMeta.from_dict(d)
        assert qm.is_per_channel is True


class TestTensorInfo:
    def test_not_quantized(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ti = TensorInfo(name="w", array=arr)
        assert ti.is_quantized is False
        assert ti.shape == (3,)
        assert ti.dtype == np.float32
        assert ti.nbytes == arr.nbytes

    def test_as_float_not_quantized(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        ti = TensorInfo(name="w", array=arr)
        result = ti.as_float()
        np.testing.assert_array_equal(result, arr)

    def test_compression_ratio_not_quantized(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        ti = TensorInfo(name="w", array=arr)
        assert ti.compression_ratio() == 1.0

    def test_quantized(self):
        qm = QuantMeta(scale=0.5, zero_point=0, bits=8, mode="symmetric",
                        dtype_code=np.dtype("int8").num, original_shape=(3,),
                        original_dtype="float32")
        arr = np.array([1, 2, 3], dtype=np.int8)
        ti = TensorInfo(name="w", array=arr, meta=qm)
        assert ti.is_quantized is True
        assert ti.shape == (3,)
