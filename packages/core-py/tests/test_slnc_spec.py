"""Tests for domains.infrastructure.slnc.spec — SLNC config, layout helpers, dtype conversion."""

from __future__ import annotations

import numpy as np
import pytest

from domains.infrastructure.slnc.spec import (
    MAGIC,
    VERSION,
    FLAGS_DEFAULT,
    FLAG_HAS_HEADER_CRC,
    FLAG_ALIGNED_TENSORS,
    FLAG_HAS_FILE_HASH,
    ALIGNMENT,
    DTYPE_FLOAT32,
    DTYPE_FLOAT16,
    DTYPE_BFLOAT16,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_UINT8,
    DTYPE_MAP,
    SLNCConfig,
    compute_header_size,
    compute_tensor_entry_size,
    compute_tensor_table_size,
    _align,
    _align_offset,
    dtype_to_code,
    code_to_dtype,
)


# ── Constants ─────────────────────────────────────────────────────────────────

class TestConstants:
    def test_magic(self):
        assert MAGIC == b"SLNC"

    def test_version(self):
        assert VERSION == 1

    def test_alignment_power_of_two(self):
        assert ALIGNMENT > 0
        assert ALIGNMENT & (ALIGNMENT - 1) == 0

    def test_dtype_map_keys(self):
        assert set(DTYPE_MAP.keys()) == {
            DTYPE_FLOAT32, DTYPE_FLOAT16, DTYPE_BFLOAT16,
            DTYPE_INT32, DTYPE_INT64, DTYPE_UINT8,
        }


# ── SLNCConfig ───────────────────────────────────────────────────────────────

class TestSLNCConfig:
    def test_defaults(self):
        cfg = SLNCConfig()
        assert cfg.alignment == ALIGNMENT
        assert cfg.verify_checksums is False
        assert cfg.align_tensors is False
        assert cfg.write_header_crc is False

    def test_from_flags_default(self):
        cfg = SLNCConfig.from_flags(0)
        assert cfg.align_tensors is False
        assert cfg.write_header_crc is False

    def test_from_flags_aligned(self):
        cfg = SLNCConfig.from_flags(FLAG_ALIGNED_TENSORS)
        assert cfg.align_tensors is True
        assert cfg.write_header_crc is False

    def test_from_flags_crc(self):
        cfg = SLNCConfig.from_flags(FLAG_HAS_HEADER_CRC)
        assert cfg.align_tensors is False
        assert cfg.write_header_crc is True

    def test_from_flags_both(self):
        flags = FLAG_ALIGNED_TENSORS | FLAG_HAS_HEADER_CRC
        cfg = SLNCConfig.from_flags(flags)
        assert cfg.align_tensors is True
        assert cfg.write_header_crc is True

    def test_to_flags_default(self):
        cfg = SLNCConfig()
        assert cfg.to_flags() == 0

    def test_to_flags_aligned(self):
        cfg = SLNCConfig(align_tensors=True)
        assert cfg.to_flags() == FLAG_ALIGNED_TENSORS

    def test_to_flags_crc(self):
        cfg = SLNCConfig(write_header_crc=True)
        assert cfg.to_flags() == FLAG_HAS_HEADER_CRC

    def test_roundtrip_flags(self):
        cfg = SLNCConfig(align_tensors=True, write_header_crc=True)
        flags = cfg.to_flags()
        cfg2 = SLNCConfig.from_flags(flags)
        assert cfg2.align_tensors == cfg.align_tensors
        assert cfg2.write_header_crc == cfg.write_header_crc


# ── _align / _align_offset ───────────────────────────────────────────────────

class TestAlign:
    def test_already_aligned(self):
        assert _align(64) == 64
        assert _align(128) == 128

    def test_needs_alignment(self):
        assert _align(1) == 64
        assert _align(65) == 128

    def test_align_offset(self):
        assert _align_offset(0) == 0
        assert _align_offset(1) == 64


# ── compute_header_size ──────────────────────────────────────────────────────

class TestComputeHeaderSize:
    def test_empty_json(self):
        size = compute_header_size(b"")
        # MAGIC(4) + VERSION(4) + FLAGS(4) + MODEL_META(64) + JSON_LEN(4) = 80
        # Aligned to 64
        assert size == 128  # 80 aligned to 128

    def test_with_json(self):
        json_bytes = b'{"key": "value"}'
        size = compute_header_size(json_bytes)
        assert size >= 80 + len(json_bytes)
        assert size % ALIGNMENT == 0


# ── compute_tensor_entry_size ────────────────────────────────────────────────

class TestComputeTensorEntrySize:
    def test_1d_tensor(self):
        # name_len(4) + name_bytes[5] + offset(8) + size(4) + ndim(4) + shape[1](4) + dtype(4) + crc32(4)
        size = compute_tensor_entry_size(ndim=1, name_len=5)
        assert size == 4 + 5 + 8 + 4 + 4 + 1 * 4 + 4 + 4

    def test_2d_tensor(self):
        size = compute_tensor_entry_size(ndim=2, name_len=10)
        assert size == 4 + 10 + 8 + 4 + 4 + 2 * 4 + 4 + 4

    def test_0d_tensor(self):
        size = compute_tensor_entry_size(ndim=0, name_len=3)
        assert size == 4 + 3 + 8 + 4 + 4 + 0 + 4 + 4


# ── compute_tensor_table_size ────────────────────────────────────────────────

class TestComputeTensorTableSize:
    def test_empty(self):
        assert compute_tensor_table_size([]) == 0

    def test_single_entry(self):
        entries = [("weight", 0, 100, 2, DTYPE_FLOAT32, 0)]
        size = compute_tensor_table_size(entries)
        expected = compute_tensor_entry_size(ndim=2, name_len=6)
        assert size == expected

    def test_multiple_entries(self):
        entries = [
            ("w1", 0, 100, 2, DTYPE_FLOAT32, 0),
            ("w2", 100, 200, 1, DTYPE_FLOAT16, 0),
        ]
        size = compute_tensor_table_size(entries)
        s1 = compute_tensor_entry_size(ndim=2, name_len=2)
        s2 = compute_tensor_entry_size(ndim=1, name_len=2)
        assert size == s1 + s2


# ── dtype_to_code / code_to_dtype ────────────────────────────────────────────

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

    def test_bfloat16_code(self):
        # bfloat16 is stored as uint16 but has its own code
        code = code_to_dtype(DTYPE_BFLOAT16)
        assert code == np.uint16

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            dtype_to_code(np.complex128)

    def test_unknown_code_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype code"):
            code_to_dtype(999)
