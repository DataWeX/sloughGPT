"""Opt-in quantize-on-ingest for unquantized float safetensors (card 019464d8).

Turns a large fp32/bf16 ``.safetensors`` (incompressible at the byte level —
gzip/zstd give ~1.0x, proven) into a smaller int8 blob by applying the model
stack's *proven* quantizer at store time. This is the lossy lever that byte
codecs cannot pull: int8 = 4x vs fp32 at cosine ~0.9996 (see the quantization
benchmarks).

Honest by construction:
* **Opt-in only** — the normal store path is untouched; a caller must pass
  ``quantize=True``. A file store that silently rewrites bytes is a bug factory.
* **Never lossy on already-compressed artifacts** — GGUF/NF4/GPTQ/AWQ already
  arrive quantization-compressed; re-quantizing is loss-on-loss, so they are
  refused. Non-float tensors are refused too.
* **Provenance recorded** — the store entry gains ``quantization``: original
  dtype + byte size, int8 bit width + mode, the worst tensor cosine, and the
  original sha256. Loaders read this to know the weights are int8, not fp32.
* **Dedup on the quantized bytes** — identity is the bytes actually stored, so
  two identical source files collapse to one blob; the original sha is kept as
  provenance, not identity.

Reuses (no duplication): ``Quantine`` / ``TensorInfo`` / ``QuantMeta`` from the
quantization stack. This module owns only the safetensors container (read/write)
and the store integration seam.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

logger = logging.getLogger("slo.files.quant")

# dtype name -> (safetensors dtype token, numpy dtype). Only these are eligible
# for quantize-on-ingest; everything else (int, bool, already-quantized codes,
# f8) is refused rather than guessed at.
_FLOAT_DTYPES: dict[str, str] = {
    "float32": "F32",
    "float16": "F16",
    "bfloat16": "BF16",
}
_SF_TO_NUMPY: dict[str, np.dtype] = {
    "F32": np.dtype("float32"),
    "F16": np.dtype("float16"),
}
_SF_TO_NAME: dict[str, str] = {"F32": "float32", "F16": "float16"}


class UnsupportedForQuantization(ValueError):
    """Raised when a safetensors file is not eligible for int8 ingest.

    The hub maps this to HTTP 400 — the caller opted in on a file that is
    already quantization-compressed (GGUF/NF4/GPTQ/AWQ), holds non-float
    tensors, or uses a layout we refuse to guess at.
    """


@dataclass
class QuantResult:
    """Summary of one quantize-on-ingest run (for logging / benchmarks).

    Ineligibility is signalled by ``quantize_safetensors`` raising
    ``UnsupportedForQuantization`` — there is no soft-fail result.
    """

    tensors: int
    quantized: int
    kept_float: int
    orig_bytes: int
    quant_bytes: int
    worst_cosine: float

    @property
    def ratio(self) -> float:
        return self.orig_bytes / self.quant_bytes if self.quant_bytes else 0.0


# ── safetensors container (read) ────────────────────────────────────


def _read_header(f: BinaryIO) -> tuple[dict, int]:
    """Read the safetensors header. Returns (header dict, start of data).

    Refuses (``UnsupportedForQuantization``) anything that is not a
    safetensors container: GGUF (distinct magic `GGUF`) and other garbage
    fail here cleanly instead of surfacing a JSON traceback.
    """
    f.seek(0)
    prefix = f.read(8)
    if len(prefix) < 8:
        raise UnsupportedForQuantization("file too short to be safetensors")
    if prefix[:4] == b"GGUF":
        raise UnsupportedForQuantization("GGUF artifact is already quantization-compressed")
    (hlen,) = struct.unpack("<Q", prefix)
    if hlen > 100 * 1024 * 1024:
        raise UnsupportedForQuantization("safetensors header implausibly large")
    body = f.read(hlen)  # cursor sits right after the 8-byte prefix
    try:
        header = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise UnsupportedForQuantization(f"not a safetensors container: {e}") from None
    data_start = 8 + hlen
    return header, data_start


def _tensor_nbytes(entry: dict, sf_to_np: dict[str, np.dtype]) -> int:
    dtype = entry["dtype"]
    if dtype not in sf_to_np:
        raise UnsupportedForQuantization(f"unsupported tensor dtype: {dtype}")
    n = 1
    for d in entry["shape"]:
        n *= d
    return n * sf_to_np[dtype].itemsize


def _load_tensors(path: Path) -> tuple[dict[str, np.ndarray], list[str], int]:
    """Load all float tensors as float32. Returns (tensors, non_float_names, total_bytes).

    Raises ``UnsupportedForQuantization`` for unknown dtype tokens.
    """
    with path.open("rb") as f:
        header, data_start = _read_header(f)
        tensors: dict[str, np.ndarray] = {}
        non_float: list[str] = []
        total = 0
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            dtype = entry["dtype"]
            if dtype not in _SF_TO_NUMPY:
                # BF16 has no direct numpy dtype; handled below via uint16 view.
                if dtype != "BF16":
                    non_float.append(name)
                    total += _tensor_nbytes(entry, _SF_TO_NUMPY) if dtype in _SF_TO_NUMPY else 0
                    continue
            nbytes = _tensor_nbytes(entry, _SF_TO_NUMPY) if dtype in _SF_TO_NUMPY else None
            if dtype == "BF16":
                shape = entry["shape"]
                n = 1
                for d in shape:
                    n *= d
                nbytes = n * 2
            total += nbytes or 0
            off = data_start + entry["data_offsets"][0]
            f.seek(off)
            raw = f.read(entry["data_offsets"][1] - entry["data_offsets"][0])
            if dtype == "F32":
                arr = np.frombuffer(raw, dtype="<f4")
            elif dtype == "F16":
                arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            elif dtype == "BF16":
                arr = _bf16_to_f32(np.frombuffer(raw, dtype="<u2"))
            else:
                continue
            tensors[name] = arr.reshape(entry["shape"]) if entry["shape"] else arr
        return tensors, non_float, total


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """Reinterpret bfloat16 (top 16 bits of fp32) as float32."""
    wide = (bits.astype(np.uint32) << 16).astype(np.int32)
    return wide.view(np.float32)


# ── quantize-on-ingest ──────────────────────────────────────────────


def quantize_safetensors(
    src_path: Path,
    dst_path: Path,
    *,
    bits: int = 8,
    mode: str = "symmetric",
    clip_percentile: float | None = None,
) -> QuantResult:
    """Quantize a float safetensors file into a smaller int8 safetensors blob.

    Refuses (``UnsupportedForQuantization``) when the file is not eligible.
    Otherwise writes a standard safetensors container at *dst_path* holding
    int8/uint8 tensors plus a ``__metadata__`` block carrying per-tensor
    dequant scales (so ``load_quantized`` round-trips to float32).
    """
    from domain.infrastructure._internal.quantization import Quantine

    tensors, non_float, orig_bytes = _load_tensors(src_path)

    if non_float:
        raise UnsupportedForQuantization(
            f"{len(non_float)} non-float tensor(s), e.g. {non_float[:3]}"
        )
    if not tensors:
        raise UnsupportedForQuantization("no float tensors found")

    engine = Quantine(bits=bits, mode=mode, clip_percentile=clip_percentile)
    out: dict[str, np.ndarray] = {}
    deq_meta: dict[str, dict] = {}
    quantized = kept = 0
    worst = 1.0

    for name, arr in tensors.items():
        info = engine.quantize(name, arr)
        if info.is_quantized:
            out[name] = info.array
            deq_meta[name] = info.meta.to_dict()
            quantized += 1
            worst = min(worst, info.meta.cosine_sim)
        else:
            # Skipped by the engine (too high error) — keep the float tensor
            # as-is rather than force a lossy squeeze.
            out[name] = arr.astype(np.float32)
            kept += 1

    _write_quantized(dst_path, out, deq_meta)
    quant_bytes = dst_path.stat().st_size
    result = QuantResult(
        tensors=len(tensors),
        quantized=quantized,
        kept_float=kept,
        orig_bytes=orig_bytes,
        quant_bytes=quant_bytes,
        worst_cosine=worst if quantized else 1.0,
    )
    logger.info(
        "quantize-on-ingest: %s -> %s  %d tensors (%d int8, %d kept)  %.2fx  worst_cos=%.5f",
        src_path.name,
        dst_path.name,
        result.tensors,
        quantized,
        kept,
        result.ratio,
        result.worst_cosine,
    )
    return result


def _write_quantized(path: Path, tensors: dict[str, np.ndarray], deq_meta: dict) -> None:
    """Write int8 + float tensors + per-tensor dequant metadata as safetensors.

    The ``__metadata__`` block is the round-trip contract: each entry holds the
    QuantMeta dict (scale/zp/original dtype+shape/bits/cosine) so
    ``load_quantized`` can dequantize back to float32 byte-faithfully.
    """
    header: dict = {"__metadata__": {"slo_quant": "int8", "tensors": json.dumps(deq_meta)}}
    blobs: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        arr = tensors[name]
        if arr.dtype == np.float32:
            sf_dtype = "F32"
            data = arr.astype("<f4").tobytes()
        else:
            # int8/uint8 quantized tensor, stored little-endian
            sf_dtype = "U8" if arr.dtype == np.uint8 else "I8"
            data = arr.astype(arr.dtype.newbyteorder("<")).tobytes()
        header[name] = {
            "dtype": sf_dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)

    hjson = json.dumps(header).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in blobs:
            f.write(b)


def load_quantized(path: Path) -> dict[str, np.ndarray]:
    """Read a quantized safetensors blob back, dequantizing int8 to float32.

    This is the loader-side contract: anything that consumed the original fp32
    weights gets float32 back, whether or not a given tensor was quantized.
    """
    from domain.infrastructure._internal.quantization import QuantMeta, TensorInfo

    with path.open("rb") as f:
        header, data_start = _read_header(f)
    meta_raw = header.get("__metadata__", {}).get("tensors", "{}")
    deq_meta = json.loads(meta_raw)

    out: dict[str, np.ndarray] = {}
    with path.open("rb") as f:
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            f.seek(data_start + entry["data_offsets"][0])
            raw = f.read(entry["data_offsets"][1] - entry["data_offsets"][0])
            dtype = entry["dtype"]
            if dtype in ("I8", "U8") and name in deq_meta:
                # Reconstruct the TensorInfo and dequantize through the stack's
                # own as_float() — handles per-channel (N,) scales and int4.
                q = np.frombuffer(raw, dtype=np.int8 if dtype == "I8" else np.uint8)
                info = TensorInfo(
                    name=name,
                    array=q.reshape(entry["shape"]),
                    meta=QuantMeta.from_dict(deq_meta[name]),
                )
                out[name] = info.as_float()
            elif dtype == "F32":
                out[name] = np.frombuffer(raw, dtype="<f4").reshape(entry["shape"])
            elif dtype == "BF16":
                out[name] = _bf16_to_f32(np.frombuffer(raw, dtype="<u2")).reshape(entry["shape"])
            elif dtype == "F16":
                out[name] = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(entry["shape"])
    return out
