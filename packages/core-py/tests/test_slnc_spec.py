"""Tests for slnc/spec.py — format constants, alignment, dtype conversion."""

import numpy as np
import pytest
from domains.infrastructure.slnc.spec import (
    MAGIC, VERSION, FLAGS_DEFAULT,
    MAGIC_SIZE, VERSION_SIZE, FLAGS_SIZE, MODEL_META_SIZE, JSON_LEN_SIZE,
    ALIGNMENT,
    DTYPE_FLOAT32, DTYPE_FLOAT16, DTYPE_BFLOAT16, DTYPE_INT32, DTYPE_INT64, DTYPE_UINT8,
    DTYPE_MAP,
    compute_header_size, compute_tensor_entry_size, compute_tensor_table_size,
    _align, dtype_to_code, code_to_dtype,
)


class TestConstants:
    def test_magic(self):
        assert MAGIC == b"SLNC"

    def test_version(self):
        assert VERSION == 1

    def test_flags_default(self):
        assert FLAGS_DEFAULT == 0

    def test_sizes_add_up(self):
        assert MAGIC_SIZE == 4
        assert VERSION_SIZE == 4
        assert FLAGS_SIZE == 4
        assert MODEL_META_SIZE == 64
        assert JSON_LEN_SIZE == 4

    def test_alignment(self):
        assert ALIGNMENT == 64

    def test_dtype_map_completeness(self):
        assert len(DTYPE_MAP) == 6
        assert DTYPE_FLOAT32 == 0
        assert DTYPE_UINT8 == 5


class TestAlign:
    def test_already_aligned(self):
        assert _align(64) == 64
        assert _align(128) == 128

    def test_needs_alignment(self):
        assert _align(1) == 64
        assert _align(65) == 128
        assert _align(100) == 128

    def test_large_values(self):
        assert _align(1000) == 1024  # 64 * 16 = 1024


class TestComputeHeaderSize:
    def test_minimal_json(self):
        size = compute_header_size(b"{}")
        assert size >= MAGIC_SIZE + VERSION_SIZE + FLAGS_SIZE + MODEL_META_SIZE + JSON_LEN_SIZE + 2
        assert size % ALIGNMENT == 0

    def test_empty_json(self):
        size = compute_header_size(b"")
        assert size % ALIGNMENT == 0
        assert size >= 76  # 4+4+4+64+4+0


class TestComputeTensorEntrySize:
    def test_1d_tensor(self):
        entry_size = compute_tensor_entry_size(ndim=1, name_len=10)
        # 32 (fixed) + 1*4 (shape) + 10 (name) = 46
        assert entry_size == 46

    def test_2d_tensor(self):
        entry_size = compute_tensor_entry_size(ndim=2, name_len=5)
        # 32 + 2*4 + 5 = 45
        assert entry_size == 45

    def test_3d_tensor(self):
        entry_size = compute_tensor_entry_size(ndim=3, name_len=20)
        # 32 + 3*4 + 20 = 64
        assert entry_size == 64


class TestComputeTensorTableSize:
    def test_single_entry(self):
        entries = [("weight", 0, 100, 2, DTYPE_FLOAT32, 0)]
        size = compute_tensor_table_size(entries)
        assert size == compute_tensor_entry_size(2, len(b"weight"))

    def test_multiple_entries(self):
        entries = [
            ("w", 0, 100, 1, DTYPE_FLOAT32, 0),
            ("bias", 100, 40, 1, DTYPE_FLOAT32, 0),
        ]
        size = compute_tensor_table_size(entries)
        assert size > 0

    def test_empty_entries(self):
        assert compute_tensor_table_size([]) == 0


class TestDtypeConversion:
    def test_float32_roundtrip(self):
        code = dtype_to_code(np.float32)
        assert code == DTYPE_FLOAT32
        assert code_to_dtype(code) == np.float32

    def test_float16_roundtrip(self):
        code = dtype_to_code(np.float16)
        assert code == DTYPE_FLOAT16
        assert code_to_dtype(code) == np.float16

    def test_int32_roundtrip(self):
        code = dtype_to_code(np.int32)
        assert code == DTYPE_INT32
        assert code_to_dtype(code) == np.int32

    def test_int64_roundtrip(self):
        code = dtype_to_code(np.int64)
        assert code == DTYPE_INT64
        assert code_to_dtype(code) == np.int64

    def test_uint8_roundtrip(self):
        code = dtype_to_code(np.uint8)
        assert code == DTYPE_UINT8
        assert code_to_dtype(code) == np.uint8

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            dtype_to_code(np.float64)

    def test_unknown_code_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype code"):
            code_to_dtype(99)
