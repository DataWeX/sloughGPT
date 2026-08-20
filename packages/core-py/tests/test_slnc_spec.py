"""Tests for domains.infrastructure.slnc.spec — format constants, layout helpers, dtype mapping."""

import numpy as np
import pytest


class TestConstants:
    def test_magic(self):
        from domains.infrastructure.slnc.spec import MAGIC
        assert MAGIC == b"SLNC"

    def test_version(self):
        from domains.infrastructure.slnc.spec import VERSION
        assert VERSION == 1

    def test_alignment(self):
        from domains.infrastructure.slnc.spec import ALIGNMENT
        assert ALIGNMENT == 64

    def test_dtype_codes(self):
        from domains.infrastructure.slnc.spec import (
            DTYPE_FLOAT32, DTYPE_FLOAT16, DTYPE_INT32, DTYPE_INT64, DTYPE_UINT8,
        )
        assert DTYPE_FLOAT32 == 0
        assert DTYPE_FLOAT16 == 1
        assert DTYPE_INT32 == 3
        assert DTYPE_INT64 == 4
        assert DTYPE_UINT8 == 5


class TestAlign:
    def test_align_already_aligned(self):
        from domains.infrastructure.slnc.spec import _align
        assert _align(64) == 64
        assert _align(128) == 128

    def test_align_one_over(self):
        from domains.infrastructure.slnc.spec import _align
        assert _align(65) == 128

    def test_align_small(self):
        from domains.infrastructure.slnc.spec import _align
        assert _align(1) == 64
        assert _align(63) == 64

    def test_align_zero(self):
        from domains.infrastructure.slnc.spec import _align
        assert _align(0) == 0


class TestComputeHeaderSize:
    def test_empty_json(self):
        from domains.infrastructure.slnc.spec import compute_header_size
        size = compute_header_size(b"")
        assert size >= 4 + 4 + 4 + 64 + 4  # magic + version + flags + meta + json_len
        assert size % 64 == 0

    def test_with_json(self):
        from domains.infrastructure.slnc.spec import compute_header_size
        json_data = b'{"model": "gpt2"}'
        size = compute_header_size(json_data)
        assert size % 64 == 0
        assert size >= compute_header_size(b"")  # must be >= empty case


class TestComputeTensorEntrySize:
    def test_1d(self):
        from domains.infrastructure.slnc.spec import compute_tensor_entry_size
        size = compute_tensor_entry_size(ndim=1, name_len=10)
        assert size == 32 + 1 * 4 + 10  # 46

    def test_2d(self):
        from domains.infrastructure.slnc.spec import compute_tensor_entry_size
        size = compute_tensor_entry_size(ndim=2, name_len=20)
        assert size == 32 + 2 * 4 + 20  # 60

    def test_4d(self):
        from domains.infrastructure.slnc.spec import compute_tensor_entry_size
        size = compute_tensor_entry_size(ndim=4, name_len=5)
        assert size == 32 + 4 * 4 + 5  # 53


class TestComputeTensorTableSize:
    def test_single_entry(self):
        from domains.infrastructure.slnc.spec import compute_tensor_table_size
        entries = [("ln_1.weight", 0, 256, 2, 0, 0)]
        total = compute_tensor_table_size(entries)
        assert total == 32 + 2 * 4 + len("ln_1.weight")

    def test_multiple_entries(self):
        from domains.infrastructure.slnc.spec import compute_tensor_table_size
        entries = [
            ("a", 0, 100, 1, 0, 0),
            ("bb", 100, 200, 2, 0, 0),
        ]
        total = compute_tensor_table_size(entries)
        e1 = 32 + 1 * 4 + 1
        e2 = 32 + 2 * 4 + 2
        assert total == e1 + e2


class TestDtypeConversion:
    def test_float32_roundtrip(self):
        from domains.infrastructure.slnc.spec import dtype_to_code, code_to_dtype, DTYPE_FLOAT32
        code = dtype_to_code(np.float32)
        assert code == DTYPE_FLOAT32
        assert code_to_dtype(code) == np.float32

    def test_float16_roundtrip(self):
        from domains.infrastructure.slnc.spec import dtype_to_code, code_to_dtype
        assert code_to_dtype(dtype_to_code(np.float16)) == np.float16

    def test_int32_roundtrip(self):
        from domains.infrastructure.slnc.spec import dtype_to_code, code_to_dtype
        assert code_to_dtype(dtype_to_code(np.int32)) == np.int32

    def test_int64_roundtrip(self):
        from domains.infrastructure.slnc.spec import dtype_to_code, code_to_dtype
        assert code_to_dtype(dtype_to_code(np.int64)) == np.int64

    def test_uint8_roundtrip(self):
        from domains.infrastructure.slnc.spec import dtype_to_code, code_to_dtype
        assert code_to_dtype(dtype_to_code(np.uint8)) == np.uint8

    def test_unsupported_dtype(self):
        from domains.infrastructure.slnc.spec import dtype_to_code
        with pytest.raises(ValueError, match="Unsupported dtype"):
            dtype_to_code(np.float64)

    def test_unknown_code(self):
        from domains.infrastructure.slnc.spec import code_to_dtype
        with pytest.raises(ValueError, match="Unknown dtype code"):
            code_to_dtype(999)
