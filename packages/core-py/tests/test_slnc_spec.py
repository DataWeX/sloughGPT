"""Tests for slnc/spec.py — .slnc binary format specification, dtype codes, layout helpers."""

import numpy as np
import pytest

from domains.infrastructure.slnc import spec


class TestConstants:
    def test_magic_and_version(self):
        assert spec.MAGIC == b"SLNC"
        assert spec.VERSION == 1
        assert spec.FLAGS_DEFAULT == 0

    def test_field_sizes(self):
        assert spec.MAGIC_SIZE == 4
        assert spec.VERSION_SIZE == 4
        assert spec.FLAGS_SIZE == 4
        assert spec.MODEL_META_SIZE == 64
        assert spec.JSON_LEN_SIZE == 4
        assert spec.ALIGNMENT == 64

    def test_dtype_codes(self):
        assert spec.DTYPE_FLOAT32 == 0
        assert spec.DTYPE_FLOAT16 == 1
        assert spec.DTYPE_BFLOAT16 == 2
        assert spec.DTYPE_INT32 == 3
        assert spec.DTYPE_INT64 == 4
        assert spec.DTYPE_UINT8 == 5

    def test_dtype_map_is_complete(self):
        assert spec.DTYPE_MAP == {
            0: "float32",
            1: "float16",
            2: "bfloat16",
            3: "int32",
            4: "int64",
            5: "uint8",
        }


class TestAlign:
    def test_aligned_value_unchanged(self):
        assert spec._align(64) == 64

    def test_unaligned_value_rounds_up(self):
        assert spec._align(65) == 128
        assert spec._align(100) == 128
        assert spec._align(0) == 0

    def test_boundary(self):
        assert spec._align(127) == 128
        assert spec._align(192) == 192


class TestComputeHeaderSize:
    def test_exact_small_payload(self):
        size = spec.compute_header_size(b"{}")
        assert size == 128  # 82 unaligned -> aligned to 64
        assert size % spec.ALIGNMENT == 0
    def test_zero_length_payload(self):
        size = spec.compute_header_size(b"")
        assert size % spec.ALIGNMENT == 0
        assert size >= spec.MAGIC_SIZE + spec.MODEL_META_SIZE

    def test_always_64_aligned(self):
        for n in (0, 1, 10, 60, 63, 64, 1000):
            assert spec.compute_header_size(b"x" * n) % spec.ALIGNMENT == 0

    def test_grows_with_payload(self):
        small = spec.compute_header_size(b"a")
        large = spec.compute_header_size(b"a" * 1000)
        assert large > small


class TestComputeTensorEntrySize:
    def test_known_values(self):
        assert spec.compute_tensor_entry_size(2, 4) == 32 + 8 + 4
        assert spec.compute_tensor_entry_size(1, 0) == 36
        assert spec.compute_tensor_entry_size(0, 10) == 42

    def test_scales_with_ndim_and_name(self):
        assert spec.compute_tensor_entry_size(3, 5) > spec.compute_tensor_entry_size(2, 5)
        assert spec.compute_tensor_entry_size(2, 8) > spec.compute_tensor_entry_size(2, 2)


class TestComputeTensorTableSize:
    def test_empty_table(self):
        assert spec.compute_tensor_table_size([]) == 0

    def test_string_names(self):
        entries = [("wte", 0, b"", 2, 0, 1)]
        assert spec.compute_tensor_table_size(entries) == spec.compute_tensor_entry_size(2, 3)

    def test_bytes_names(self):
        entries = [(b"wte", 0, b"", 2, 0, 1)]
        assert spec.compute_tensor_table_size(entries) == spec.compute_tensor_entry_size(2, 3)

    def test_multi_entry_sum(self):
        entries = [
            ("a", 0, b"", 1, 0, 1),
            ("bb", 0, b"", 2, 0, 1),
            (b"ccc", 0, b"", 3, 0, 1),
        ]
        expected = sum(
            spec.compute_tensor_entry_size(ndim, name_len)
            for (name, _, _, ndim, _, _), name_len in zip(entries, [1, 2, 3])
        )
        assert spec.compute_tensor_table_size(entries) == expected


class TestDtypeToCode:
    @pytest.mark.parametrize(
        "dtype,code",
        [
            (np.float32, spec.DTYPE_FLOAT32),
            (np.float16, spec.DTYPE_FLOAT16),
            (np.int32, spec.DTYPE_INT32),
            (np.int64, spec.DTYPE_INT64),
            (np.uint8, spec.DTYPE_UINT8),
        ],
    )
    def test_supported_dtypes(self, dtype, code):
        assert spec.dtype_to_code(dtype) == code

    def test_accepts_dtype_instances(self):
        assert spec.dtype_to_code(np.dtype("float32")) == spec.DTYPE_FLOAT32
        assert spec.dtype_to_code(np.dtype("int64")) == spec.DTYPE_INT64

    @pytest.mark.parametrize("dtype", [np.float64, np.bool_, np.complex64, "not-a-dtype", 3])
    def test_unsupported_dtype_raises(self, dtype):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            spec.dtype_to_code(dtype)


class TestCodeToDtype:
    @pytest.mark.parametrize(
        "code,name",
        [
            (spec.DTYPE_FLOAT32, "float32"),
            (spec.DTYPE_FLOAT16, "float16"),
            (spec.DTYPE_INT32, "int32"),
            (spec.DTYPE_INT64, "int64"),
            (spec.DTYPE_UINT8, "uint8"),
        ],
    )
    def test_known_codes(self, code, name):
        assert spec.code_to_dtype(code) == getattr(np, name)

    @pytest.mark.parametrize("code", [-1, 6, 99, 2, None])
    def test_unknown_codes_raise(self, code):
        with pytest.raises(ValueError, match="Unknown dtype code"):
            spec.code_to_dtype(code)


class TestDtypeRoundTrip:
    def test_dtype_to_code_inverse_of_code_to_dtype(self):
        for code, name in spec.DTYPE_MAP.items():
            if not hasattr(np, name):
                continue
            np_dtype = spec.code_to_dtype(code)
            assert spec.dtype_to_code(np_dtype) == code
