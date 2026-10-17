"""_safetensors — read ``.safetensors`` files with stdlib + numpy only.

The format promises simplicity and we take it at face value:

    [ 8 bytes: little-endian u64 header length N ]
    [ N bytes: JSON header — {"__metadata__": ..., "name": {dtype,
      shape, data_offsets}}; offsets are byte ranges into the buffer ]
    [ the raw little-endian tensor buffer ]

~30 lines replace a dependency exactly one script in the repo used.
Ground-up rule applied: the protocol is small, stable, and hot for
checkpoint profiling — we own it.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

# safetensors dtypes are little-endian by spec; encode that explicitly.
_DTYPES: dict[str, str] = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U8": "u1",
    "BOOL": "?",
}


def _decode(dtype: str, raw: bytes, shape: list[int]) -> np.ndarray:
    if dtype == "BF16":
        # bfloat16 = the float32 top 16 bits; widen via a uint32 shift.
        widened = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        return widened.reshape(shape)
    np_dtype = _DTYPES.get(dtype)
    if np_dtype is None:
        raise ValueError(f"unsupported safetensors dtype: {dtype!r}")
    return np.frombuffer(raw, dtype=np.dtype(np_dtype)).reshape(shape)


def read_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Every tensor in the file, name -> ndarray (read-only views)."""
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"not a safetensors file (too short): {path}")
    (header_len,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + header_len])
    buffer = data[8 + header_len :]
    tensors: dict[str, Any] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        begin, end = spec["data_offsets"]
        tensors[name] = _decode(spec["dtype"], buffer[begin:end], spec["shape"])
    return tensors
