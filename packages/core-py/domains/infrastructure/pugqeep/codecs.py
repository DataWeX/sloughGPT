"""
Codec registry — pluggable per-function-type codecs for Point.

Before this module, each Point codec's logic was spread across FOUR coupled
if/elif dispatch chains in ``point.py`` (``generate``, ``nbytes``,
``to_bytes``, ``from_bytes``). Adding a type required editing all four in
lockstep, and ``block_q4``/``block_q8`` were left half-wired — they serialized
and sized but ``generate()`` raised ``Unknown function type``.

A ``PointCodec`` bundles the four responsibilities for one function type.
``Point`` now dispatches to the registered codec, so adding a type is a single
self-contained registration.

Backward compatibility: the on-disk binary layout produced by
``serialize_params``/``deserialize_params`` is byte-identical to the legacy
hand-rolled format. An optional metadata trailer (accuracy/dtype/shape) is
appended only when present; blobs without it still parse.
"""

from __future__ import annotations

import base64
import struct
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .point import Point


@runtime_checkable
class PointCodec(Protocol):
    """One codec per Point.function_type.

    Bundles the four per-type responsibilities that previously lived in
    coupled if/elif chains in ``Point``.
    """

    function_type: str

    def generate(self, point: Point, n: int) -> np.ndarray:
        """Reconstruct ``n`` values as float32."""
        ...

    def nbytes(self, point: Point) -> int:
        """Estimate in-memory size of stored params, in bytes."""
        ...

    def serialize_params(self, point: Point) -> bytes:
        """Encode params (no type code, no residual) to bytes."""
        ...

    def deserialize_params(self, params: bytes) -> tuple[dict, np.ndarray | None]:
        """Decode params bytes into (params dict, residual or None).

        The codec owns the full param+residual layout for its type.
        """
        ...


# ── shared residual helpers (layout unchanged from the legacy chains) ────────


def _pack_residual(residual: np.ndarray | None) -> bytes:
    """[has_res:u8] ([len:u32] [data:f32*n])? — legacy residual layout."""
    if residual is None:
        return struct.pack("<B", 0)
    res_bytes = residual.astype(np.float32).tobytes()
    return struct.pack("<B", 1) + struct.pack("<I", len(res_bytes)) + res_bytes


def _unpack_residual(buf: bytes, offset: int) -> tuple[np.ndarray | None, int]:
    """Read legacy residual block; returns (residual, new_offset)."""
    if offset >= len(buf):
        return None, offset
    has_res = struct.unpack("<B", buf[offset : offset + 1])[0]
    offset += 1
    if not has_res:
        return None, offset
    res_len = struct.unpack("<I", buf[offset : offset + 4])[0]
    offset += 4
    residual = np.frombuffer(buf[offset : offset + res_len], dtype=np.float32)
    return residual, offset + res_len


def _pack_block_header(params: dict) -> bytes:
    return struct.pack(
        "<III",
        int(params["n_elements"]),
        int(params["n_blocks"]),
        int(params["block_size"]),
    )


def _unpack_block_header(buf: bytes) -> tuple[int, int, int, int]:
    n_elements, n_blocks, block_size = struct.unpack_from("<III", buf, 0)
    return n_elements, n_blocks, block_size, 12


# ── function-based codecs ────────────────────────────────────────────────────


class PeriodicCodec:
    """values = a*cos(i) + b*sin(i) + w (+ residual)."""

    function_type = "periodic"

    def generate(self, point: Point, n: int) -> np.ndarray:
        i = np.arange(n, dtype=np.float32)
        a, b, w = point.params["a"], point.params["b"], point.params["w"]
        values = a * np.cos(i) + b * np.sin(i) + w
        if point.residual is not None:
            values = values + point.residual[:n]
        return values

    def nbytes(self, point: Point) -> int:
        return (
            4 + len(point.params) * 4 + (point.residual.nbytes if point.residual is not None else 0)
        )

    def serialize_params(self, point: Point) -> bytes:
        p = point.params
        return struct.pack("fff", p["a"], p["b"], p["w"]) + _pack_residual(point.residual)

    def deserialize_params(self, buf: bytes) -> tuple[dict, np.ndarray | None]:
        a, b, w = struct.unpack("fff", buf[:12])
        residual, _ = _unpack_residual(buf, 12)
        return {"a": a, "b": b, "w": w}, residual


class LinearCodec:
    """values = a*i + b (+ residual). Params serialized in insertion order."""

    function_type = "linear"

    def generate(self, point: Point, n: int) -> np.ndarray:
        i = np.arange(n, dtype=np.float32)
        a, b = point.params["a"], point.params["b"]
        values = a * i + b
        if point.residual is not None:
            values = values + point.residual[:n]
        return values

    def nbytes(self, point: Point) -> int:
        return (
            4 + len(point.params) * 4 + (point.residual.nbytes if point.residual is not None else 0)
        )

    def serialize_params(self, point: Point) -> bytes:
        body = struct.pack(f"{len(point.params)}f", *point.params.values())
        return body + _pack_residual(point.residual)

    def deserialize_params(self, buf: bytes) -> tuple[dict, np.ndarray | None]:
        a, b = struct.unpack("ff", buf[:8])
        residual, _ = _unpack_residual(buf, 8)
        return {"a": a, "b": b}, residual


class PolynomialCodec:
    """values = a*i^2 + b*i + c (+ residual)."""

    function_type = "polynomial"

    def generate(self, point: Point, n: int) -> np.ndarray:
        i = np.arange(n, dtype=np.float32)
        a, b, c = point.params["a"], point.params["b"], point.params["c"]
        values = a * i**2 + b * i + c
        if point.residual is not None:
            values = values + point.residual[:n]
        return values

    def nbytes(self, point: Point) -> int:
        return (
            4 + len(point.params) * 4 + (point.residual.nbytes if point.residual is not None else 0)
        )

    def serialize_params(self, point: Point) -> bytes:
        p = point.params
        return struct.pack("fff", p["a"], p["b"], p["c"]) + _pack_residual(point.residual)

    def deserialize_params(self, buf: bytes) -> tuple[dict, np.ndarray | None]:
        a, b, c = struct.unpack("fff", buf[:12])
        residual, _ = _unpack_residual(buf, 12)
        return {"a": a, "b": b, "c": c}, residual


class ClusterCodec:
    """Vector-quantized generation via centroids + assignments (+ residual).

    Supports Huffman-coded assignments and uint8-quantized centroids, matching
    the legacy layout exactly.
    """

    function_type = "cluster"

    def generate(self, point: Point, n: int) -> np.ndarray:
        centroids = point.params["centroids"]
        assignments = point.params["assignments"]
        huffman_data = point.params.get("huffman_data")
        if huffman_data is not None:
            from .compressor import HuffmanTree

            huffman_codes = point.params["huffman_codes"]
            total_bits = point.params["huffman_bits"]
            tree = HuffmanTree.from_dict(huffman_codes).tree
            assignments = HuffmanTree.decode(huffman_data, total_bits, tree, len(assignments))
        if point.params.get("centroid_quantized"):
            scale = point.params["centroid_scale"]
            zp = point.params["centroid_zero_point"]
            centroids = (centroids.astype(np.float32) - zp) * scale
        values = centroids[assignments[:n]]
        if point.residual is not None:
            values = values + point.residual[:n]
        return values

    def nbytes(self, point: Point) -> int:
        centroids = point.params.get("centroids")
        assignments = point.params.get("assignments")
        if centroids is None or assignments is None:
            return 0
        huffman_data = point.params.get("huffman_data")
        if huffman_data is not None:
            tree_size = sum(len(v) for v in point.params.get("huffman_codes", {}).values())
            return centroids.nbytes + len(huffman_data) + tree_size + 8
        if point.params.get("centroid_quantized"):
            return len(centroids) + 8 + assignments.nbytes
        return (
            centroids.nbytes
            + assignments.nbytes
            + (point.residual.nbytes if point.residual is not None else 0)
        )

    def serialize_params(self, point: Point) -> bytes:
        p = point.params
        centroids = p["centroids"]
        assignments = p["assignments"]
        is_quantized = p.get("centroid_quantized", False)
        out = struct.pack("<I", centroids.size)
        out += struct.pack("<B", 1 if is_quantized else 0)
        if is_quantized:
            out += centroids.tobytes()  # uint8
            out += struct.pack("f", p["centroid_scale"])
            out += struct.pack("f", p["centroid_zero_point"])
        else:
            out += centroids.astype(np.float32).tobytes()
        out += struct.pack("<I", len(assignments))
        out += assignments.tobytes()
        out += _pack_residual(point.residual)
        return out

    def deserialize_params(self, buf: bytes) -> tuple[dict, np.ndarray | None]:
        offset = 0
        n_centroids = struct.unpack("<I", buf[offset : offset + 4])[0]
        offset += 4
        is_quantized = struct.unpack("<B", buf[offset : offset + 1])[0]
        offset += 1
        params: dict = {}
        if is_quantized:
            centroids = np.frombuffer(buf[offset : offset + n_centroids], dtype=np.uint8)
            offset += n_centroids
            scale = struct.unpack("f", buf[offset : offset + 4])[0]
            offset += 4
            zp = struct.unpack("f", buf[offset : offset + 4])[0]
            offset += 4
            params = {
                "centroids": centroids,
                "centroid_quantized": True,
                "centroid_scale": scale,
                "centroid_zero_point": zp,
            }
        else:
            centroids = np.frombuffer(buf[offset : offset + n_centroids * 4], dtype=np.float32)
            offset += n_centroids * 4
            params = {"centroids": centroids}
        n_assignments = struct.unpack("<I", buf[offset : offset + 4])[0]
        offset += 4
        params["assignments"] = np.frombuffer(buf[offset : offset + n_assignments], dtype=np.uint8)
        offset += n_assignments
        residual, _ = _unpack_residual(buf, offset)
        return params, residual


class RawCodec:
    """Lossless: stored as-is (base64 in params, raw bytes on the wire)."""

    function_type = "raw"

    def generate(self, point: Point, n: int) -> np.ndarray:
        raw_bytes = base64.b64decode(point.params["data_b64"])
        return np.frombuffer(raw_bytes, dtype=point.params["dtype"])

    def nbytes(self, point: Point) -> int:
        return len(base64.b64decode(point.params.get("data_b64", "")))

    def serialize_params(self, point: Point) -> bytes:
        raw_data = base64.b64decode(point.params["data_b64"])
        return struct.pack("<I", len(raw_data)) + raw_data

    def deserialize_params(self, buf: bytes) -> tuple[dict, np.ndarray | None]:
        n_bytes = struct.unpack("<I", buf[:4])[0]
        raw_data = buf[4 : 4 + n_bytes]
        return (
            {
                "data_b64": base64.b64encode(raw_data).decode(),
                "shape": [],
                "dtype": "float32",
            },
            None,
        )


# ── block-quantization codecs (previously could not generate) ────────────────


class BlockQ4Codec:
    """Block-wise 4-bit quantization (Q4_K style).

    ``generate`` dequantizes packed nibbles with per-block mins/scales — this
    closes the round-trip hole where a deserialized block_q4 Point raised
    ``Unknown function type``. Mirrors ``compressor.decompress_block_q4``.
    """

    function_type = "block_q4"

    def generate(self, point: Point, n: int) -> np.ndarray:
        mins = point.params["mins"]
        scales = point.params["scales"]
        packed = point.params["packed"]
        n_elements = point.params["n_elements"]
        n_blocks = point.params["n_blocks"]
        bs = point.params["block_size"]
        unpacked = np.zeros(n_blocks * bs, dtype=np.uint8)
        unpacked[0::2] = packed & 0x0F
        unpacked[1::2] = (packed >> 4) & 0x0F
        q = unpacked.reshape(n_blocks, bs).astype(np.float32)
        return (q * scales[:, None] + mins[:, None]).ravel()[: min(n, n_elements)]

    def nbytes(self, point: Point) -> int:
        mins = point.params.get("mins")
        scales = point.params.get("scales")
        packed = point.params.get("packed")
        if mins is not None and scales is not None and packed is not None:
            return mins.nbytes + scales.nbytes + packed.nbytes
        return 0

    def serialize_params(self, point: Point) -> bytes:
        p = point.params
        mins = p["mins"].astype(np.float32)
        scales = p["scales"].astype(np.float32)
        packed = p["packed"]
        out = _pack_block_header(p)
        out += mins.tobytes()
        out += scales.tobytes()
        out += struct.pack("<I", len(packed))
        out += packed.tobytes()
        return out

    def deserialize_params(self, buf: bytes) -> tuple[dict, np.ndarray | None]:
        n_elements, n_blocks, block_size, offset = _unpack_block_header(buf)
        n_f32 = n_blocks * 2
        floats = np.frombuffer(buf[offset : offset + n_f32 * 4], dtype=np.float32)
        offset += n_f32 * 4
        mins = floats[:n_blocks]
        scales = floats[n_blocks:]
        packed_len = struct.unpack_from("<I", buf, offset)[0]
        offset += 4
        packed = np.frombuffer(buf[offset : offset + packed_len], dtype=np.uint8)
        return (
            {
                "mins": mins,
                "scales": scales,
                "packed": packed,
                "n_elements": n_elements,
                "n_blocks": n_blocks,
                "block_size": block_size,
            },
            None,
        )


class BlockQ8Codec:
    """Block-wise 8-bit quantization.

    ``generate`` dequantizes per-block values with mins/scales — closes the
    same round-trip hole as Q4. Mirrors ``compressor.decompress_block_q8``.
    """

    function_type = "block_q8"

    def generate(self, point: Point, n: int) -> np.ndarray:
        mins = point.params["mins"]
        scales = point.params["scales"]
        values = point.params["values"]
        n_elements = point.params["n_elements"]
        n_blocks = point.params["n_blocks"]
        bs = point.params["block_size"]
        q = values.reshape(n_blocks, bs).astype(np.float32)
        return (q * scales[:, None] + mins[:, None]).ravel()[: min(n, n_elements)]

    def nbytes(self, point: Point) -> int:
        mins = point.params.get("mins")
        scales = point.params.get("scales")
        values = point.params.get("values")
        if mins is not None and scales is not None and values is not None:
            return mins.nbytes + scales.nbytes + values.nbytes
        return 0

    def serialize_params(self, point: Point) -> bytes:
        p = point.params
        mins = p["mins"].astype(np.float32)
        scales = p["scales"].astype(np.float32)
        values = p["values"]
        out = _pack_block_header(p)
        out += mins.tobytes()
        out += scales.tobytes()
        values_flat = values.astype(np.uint8).ravel()
        out += struct.pack("<I", values_flat.size)
        out += values_flat.tobytes()
        return out

    def deserialize_params(self, buf: bytes) -> tuple[dict, np.ndarray | None]:
        n_elements, n_blocks, block_size, offset = _unpack_block_header(buf)
        n_f32 = n_blocks * 2
        floats = np.frombuffer(buf[offset : offset + n_f32 * 4], dtype=np.float32)
        offset += n_f32 * 4
        mins = floats[:n_blocks]
        scales = floats[n_blocks:]
        values_len = struct.unpack_from("<I", buf, offset)[0]
        offset += 4
        values = np.frombuffer(buf[offset : offset + values_len], dtype=np.uint8)
        values = values.reshape(n_blocks, block_size)
        return (
            {
                "mins": mins,
                "scales": scales,
                "values": values,
                "n_elements": n_elements,
                "n_blocks": n_blocks,
                "block_size": block_size,
            },
            None,
        )


# ── registry ─────────────────────────────────────────────────────────────────


class CodecRegistry:
    """Maps function_type -> codec. Register to add a new Point type."""

    def __init__(self) -> None:
        self._codecs: dict[str, PointCodec] = {}

    def register(self, codec: PointCodec) -> None:
        self._codecs[codec.function_type] = codec

    def get(self, function_type: str) -> PointCodec:
        codec = self._codecs.get(function_type)
        if codec is None:
            raise ValueError(f"Unknown function type: {function_type}")
        return codec

    def __contains__(self, function_type: str) -> bool:
        return function_type in self._codecs

    def types(self) -> list[str]:
        return sorted(self._codecs)


codec_registry = CodecRegistry()
for _codec in (
    PeriodicCodec(),
    LinearCodec(),
    PolynomialCodec(),
    ClusterCodec(),
    RawCodec(),
    BlockQ4Codec(),
    BlockQ8Codec(),
):
    codec_registry.register(_codec)
del _codec
