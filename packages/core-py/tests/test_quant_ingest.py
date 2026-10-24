"""Tests for quantize-on-ingest (card 019464d8) — int8 at store time, opt-in.

Proves the honest contract:
  * a fp32 safetensors shrinks ~4x at cosine >= 0.99 (proven int8 stack),
  * ``load_quantized`` dequantizes back to float32 within tolerance,
  * already-compressed artifacts (GGUF/NF4) and non-float tensors are refused,
  * the store entry records int8 provenance, and only on explicit opt-in.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from domain.files import (
    FileStore,
    UnsupportedForQuantization,
    load_quantized,
    quantize_safetensors,
)


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Minimal safetensors writer (8-byte LE header len + JSON header + data)."""
    import json

    header: dict = {}
    blobs: list[bytes] = []
    off = 0
    for name in sorted(tensors):
        arr = tensors[name].astype("<f4")
        data = arr.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [off, off + len(data)],
        }
        blobs.append(data)
        off += len(data)
    hjson = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in blobs:
            f.write(b)


@pytest.fixture
def model_dir(tmp_path):
    """A small fp32 'model': 2D weight matrices + a 1D bias."""
    rng = np.random.default_rng(7)
    tensors = {
        "blocks.0.q_proj.weight": rng.normal(0, 0.05, (64, 128)).astype(np.float32),
        "blocks.0.v_proj.weight": rng.normal(0, 0.05, (64, 128)).astype(np.float32),
        "blocks.0.bias": rng.normal(0, 0.05, (128,)).astype(np.float32),
    }
    src = tmp_path / "model.safetensors"
    _write_safetensors(src, tensors)
    return src, tensors


class TestQuantizeRoundTrip:
    def test_shrinks_and_preserves_cosine(self, model_dir, tmp_path):
        src, tensors = model_dir
        dst = tmp_path / "q.safetensors"
        result = quantize_safetensors(src, dst)

        # int8 = 4x vs fp32 (per-tensor overhead eats a little on tiny tensors)
        assert result.ratio >= 3.0
        assert result.quantized >= 1
        assert result.worst_cosine >= 0.99

    def test_load_quantized_dequantizes(self, model_dir, tmp_path):
        src, tensors = model_dir
        dst = tmp_path / "q.safetensors"
        quantize_safetensors(src, dst)

        restored = load_quantized(dst)
        assert set(restored) == set(tensors)
        for name, orig in tensors.items():
            got = restored[name]
            assert got.shape == orig.shape
            # cosine of the reconstructed vs original is the real fidelity bar
            cos = float(
                np.dot(orig.ravel(), got.ravel())
                / (np.linalg.norm(orig) * np.linalg.norm(got) + 1e-12)
            )
            assert cos >= 0.99, f"{name} cosine {cos} below tolerance"

    def test_dtype_converted_to_float32(self, model_dir, tmp_path):
        src, _ = model_dir
        dst = tmp_path / "q.safetensors"
        quantize_safetensors(src, dst)
        for arr in load_quantized(dst).values():
            assert arr.dtype == np.float32


class TestRefusals:
    def test_refuses_non_safetensors(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        # GGUF magic `GGUF` (0x46554747 LE) — an already-compressed artifact
        gguf.write_bytes(b"GGUF" + b"\x03\x00\x00\x00" + b"\x00" * 64)
        with pytest.raises(UnsupportedForQuantization, match="GGUF"):
            quantize_safetensors(gguf, tmp_path / "out.safetensors")

    def test_refuses_garbage(self, tmp_path):
        junk = tmp_path / "junk.safetensors"
        junk.write_bytes(b"not a container at all" * 8)
        with pytest.raises(UnsupportedForQuantization, match="safetensors"):
            quantize_safetensors(junk, tmp_path / "out.safetensors")

    def test_refuses_non_float_tensors(self, tmp_path):
        import json

        # an int32 tensor is not a quantization candidate
        path = tmp_path / "int.safetensors"
        data = np.arange(4, dtype="<i4").tobytes()
        header = {"t": {"dtype": "I32", "shape": [4], "data_offsets": [0, len(data)]}}
        hjson = json.dumps(header).encode()
        with path.open("wb") as f:
            f.write(struct.pack("<Q", len(hjson)))
            f.write(hjson)
            f.write(data)
        with pytest.raises(UnsupportedForQuantization):
            quantize_safetensors(path, tmp_path / "out.safetensors")


class TestStorePutQuantized:
    def test_put_quantized_records_provenance(self, model_dir, tmp_path):
        src, tensors = model_dir
        store = FileStore(root=tmp_path / "uploads")
        meta = store.put_quantized(src, "model.safetensors")

        q = meta["quantization"]
        assert meta["codec"] == "raw"  # int8 is entropy-dense — stored raw
        assert q["bits"] == 8
        assert q["worst_cosine"] >= 0.99
        assert q["ratio"] >= 3.0
        # stored blob is smaller than a naive fp32 copy of the same tensors
        fp32_bytes = sum(a.nbytes for a in tensors.values())
        assert meta["stored_size"] < fp32_bytes

    def test_put_stream_unaffected(self, model_dir, tmp_path):
        """Opt-in only: the normal put path must not quantize."""
        src, _ = model_dir
        store = FileStore(root=tmp_path / "uploads")
        meta = store.put_stream(src.open("rb"), "model.safetensors")
        assert "quantization" not in meta
        # raw passthrough of the fp32 source
        assert meta["codec"] == "raw"
        assert meta["size_bytes"] == src.stat().st_size
