"""
Point — compressed data with meaning.

A Point is a compressed representation of a numpy array, storing a generator
function instead of raw values. The encoder is pluggable:
  - Cluster-based generation (vector quantization)
  - Function-based generation (periodic, linear, polynomial)
  - Raw storage for incompressible arrays

The contract is the ARRAY, not the domain: any component that reduces to a
numpy array (weights, embeddings, features, time series) compresses the same
way. A Point is lossy unless `accuracy == 1.0`; it carries `residual`,
`dtype`, and `shape` so reconstruction is faithful within the error budget.

Implements PointProtocol — the abstract interface for compressed data.
"""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from .codecs import codec_registry
from .point_interface import PointProtocol


@dataclass(eq=False)
class Point(PointProtocol):
    """Compressed data with meaning — stores a generator function.

    Implements PointProtocol for type-safe, swappable compression backends.
    """

    identity: str  # what this point represents
    function_type: str  # "periodic", "linear", "polynomial", "cluster", "raw"
    params: dict  # function parameters
    residual: np.ndarray | None = None  # difference from exact
    accuracy: float = 0.0  # how well it fits (0-1)
    dtype: str = "float32"  # original data dtype
    shape: tuple = ()  # original data shape

    def _codec(self):
        """Look up the codec for this point's function type."""
        return codec_registry.get(self.function_type)

    @classmethod
    def _codec_for(cls, function_type: str):
        """Class-level codec lookup (used by from_bytes)."""
        return codec_registry.get(function_type)

    def generate(self, n: int) -> np.ndarray:
        """Generate n values from this point's function."""
        return self._codec().generate(self, n)

    def nbytes(self) -> int:
        """Estimated memory usage of stored parameters."""
        return self._codec().nbytes(self)

    def _estimate_raw_bytes(self) -> int:
        """Estimate uncompressed size for compression ratio calculation."""
        if self.function_type == "cluster":
            assignments = self.params.get("assignments")
            if assignments is not None:
                return len(assignments) * 4  # float32 per element
            return 0
        elif self.function_type == "raw":
            return self.nbytes()
        else:
            # Use stored shape if available
            if self.shape:
                return int(np.prod(self.shape)) * 4  # float32 per element
            return 0

    # Type code mapping — fixed 4-byte headers for binary serialization
    _TYPE_CODES = {
        "periodic": b"PER ",
        "linear": b"LIN ",
        "polynomial": b"POLY",
        "cluster": b"CLUS",
        "raw": b"RAW ",
        "block_q4": b"BQ4 ",
        "block_q8": b"BQ8 ",
    }
    _TYPE_DECODE = {v: k for k, v in _TYPE_CODES.items()}
    # CLH1: cluster point whose params carry a Huffman bitstream (card
    # d9028c8f). Same codec as CLUS — only the payload layout differs; old
    # readers without this entry fail loudly on the unknown code instead of
    # misparsing the bitstream as dense assignments.
    _TYPE_DECODE[b"CLH1"] = "cluster"

    @staticmethod
    def _pack_block_header(params: dict) -> bytes:
        return struct.pack(
            "<III",
            int(params["n_elements"]),
            int(params["n_blocks"]),
            int(params["block_size"]),
        )

    @staticmethod
    def _unpack_block_header(param_bytes: bytes) -> tuple[int, int, int, int]:
        n_elements, n_blocks, block_size = struct.unpack_from("<III", param_bytes, 0)
        return n_elements, n_blocks, block_size, 12

    # Optional metadata trailer appended after params (accuracy/dtype/shape).
    # Legacy blobs have no trailer -> from_bytes falls back to defaults.
    _META_MAGIC = b"PGM1"

    def to_bytes(self) -> bytes:
        """Serialize point to bytes via its codec.

        Layout: [type_code:4] [codec params (+ residual per codec)] [meta?]
        where meta (only when non-default) is:
          b"PGM1" [accuracy:f32] [dtype_len:u8] [dtype] [ndim:u8] [shape:u32*ndim]
        Legacy blobs without the trailer still parse (defaults applied).
        """
        type_bytes = self._TYPE_CODES.get(self.function_type, b"\0\0\0\0")
        if self.function_type == "cluster" and self.params.get("huffman_data") is not None:
            # Huffman-carrying cluster params use the CLH1 layout (see
            # ClusterCodec.serialize_params); plain dense clusters keep CLUS.
            type_bytes = b"CLH1"
        if self.function_type not in codec_registry:
            # Legacy contract: unknown types serialize to the bare zero code.
            return type_bytes
        out = type_bytes + codec_registry.get(self.function_type).serialize_params(self)

        has_meta = self.accuracy != 0.0 or self.dtype != "float32" or self.shape != ()
        if has_meta:
            dtype_b = self.dtype.encode()
            meta = self._META_MAGIC + struct.pack("f", float(self.accuracy))
            meta += struct.pack("B", len(dtype_b)) + dtype_b
            meta += struct.pack("B", len(self.shape))
            for dim in self.shape:
                meta += struct.pack("<I", int(dim))
            out += meta
        return out

    @classmethod
    def from_bytes(cls, data: bytes, identity: str = "unknown") -> Point:
        """Deserialize point from bytes via the registered codec.

        Tolerates legacy blobs (no metadata trailer) — accuracy/dtype/shape
        then take their defaults, matching historical behavior.
        """
        type_bytes = data[:4]
        function_type = cls._TYPE_DECODE.get(type_bytes)
        if function_type is None:
            raise ValueError(f"Unknown type code: {type_bytes!r}")
        rest = data[4:]

        # Split off the optional metadata trailer.
        accuracy = 0.0
        dtype = "float32"
        shape: tuple = ()
        meta_at = rest.rfind(cls._META_MAGIC)
        param_bytes = rest
        if meta_at != -1 and len(rest) - meta_at >= 10:
            candidate = rest[meta_at:]
            try:
                off = len(cls._META_MAGIC)
                accuracy = struct.unpack("f", candidate[off : off + 4])[0]
                off += 4
                dt_len = struct.unpack("B", candidate[off : off + 1])[0]
                off += 1
                dtype = candidate[off : off + dt_len].decode()
                off += dt_len
                ndim = struct.unpack("B", candidate[off : off + 1])[0]
                off += 1
                shape = tuple(
                    struct.unpack_from("<I", candidate, off + 4 * i)[0] for i in range(ndim)
                )
                param_bytes = rest[:meta_at]
            except (struct.error, UnicodeDecodeError, IndexError):
                # Not a real trailer (magic collision in raw payload) — treat as params.
                accuracy, dtype, shape = 0.0, "float32", ()
                param_bytes = rest

        codec = cls._codec_for(function_type)
        params, residual = codec.deserialize_params(param_bytes)
        return cls(
            identity=identity,
            function_type=function_type,
            params=params,
            residual=residual,
            accuracy=float(accuracy),
            dtype=dtype,
            shape=shape,
        )

    def to_dict(self) -> dict:
        """Serialize point to JSON-compatible dict."""
        d: dict[str, Any] = {
            "identity": self.identity,
            "function_type": self.function_type,
            "accuracy": self.accuracy,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }
        if self.function_type == "cluster":
            centroids = self.params["centroids"]
            assignments = self.params["assignments"]
            is_quantized = self.params.get("centroid_quantized", False)
            d["params"] = {
                "centroids_b64": base64.b64encode(centroids.tobytes()).decode(),
                "centroids_shape": list(centroids.shape),
                "centroids_dtype": str(centroids.dtype),
                "assignments_b64": base64.b64encode(assignments.tobytes()).decode(),
                "assignments_shape": list(assignments.shape),
                "assignments_dtype": str(assignments.dtype),
                "centroid_quantized": is_quantized,
            }
            if is_quantized:
                d["params"]["centroid_scale"] = self.params["centroid_scale"]
                d["params"]["centroid_zero_point"] = self.params["centroid_zero_point"]
        elif self.function_type == "raw":
            d["params"] = dict(self.params)
        elif self.function_type in ("block_q4", "block_q8"):
            payload_key = "packed" if self.function_type == "block_q4" else "values"
            payload = self.params[payload_key]
            d["params"] = {
                "mins_b64": base64.b64encode(self.params["mins"].tobytes()).decode(),
                "scales_b64": base64.b64encode(self.params["scales"].tobytes()).decode(),
                f"{payload_key}_b64": base64.b64encode(payload.tobytes()).decode(),
                f"{payload_key}_shape": list(payload.shape),
                "n_elements": int(self.params["n_elements"]),
                "n_blocks": int(self.params["n_blocks"]),
                "block_size": int(self.params["block_size"]),
            }
        else:
            d["params"] = {k: float(v) for k, v in self.params.items()}

        if self.residual is not None:
            d["residual_b64"] = base64.b64encode(self.residual.tobytes()).decode()
            d["residual_shape"] = list(self.residual.shape)
            d["residual_dtype"] = str(self.residual.dtype)

        return d

    @classmethod
    def from_dict(cls, d: dict) -> Point:
        """Deserialize point from dict."""
        func_type = d["function_type"]

        if func_type == "cluster":
            pd = d["params"]
            centroids = np.frombuffer(
                base64.b64decode(pd["centroids_b64"]),
                dtype=pd["centroids_dtype"],
            ).reshape(pd["centroids_shape"])
            assignments = np.frombuffer(
                base64.b64decode(pd["assignments_b64"]),
                dtype=pd["assignments_dtype"],
            ).reshape(pd["assignments_shape"])
            params = {"centroids": centroids, "assignments": assignments}
            if pd.get("centroid_quantized"):
                params["centroid_quantized"] = True
                params["centroid_scale"] = pd["centroid_scale"]
                params["centroid_zero_point"] = pd["centroid_zero_point"]
        elif func_type == "raw":
            params = dict(d["params"])
        elif func_type in ("block_q4", "block_q8"):
            pd = d["params"]
            payload_key = "packed" if func_type == "block_q4" else "values"
            n_blocks = pd["n_blocks"]
            block_size = pd["block_size"]
            payload = np.frombuffer(
                base64.b64decode(pd[f"{payload_key}_b64"]), dtype=np.uint8
            ).reshape(pd[f"{payload_key}_shape"])
            params = {
                "mins": np.frombuffer(base64.b64decode(pd["mins_b64"]), dtype=np.float32),
                "scales": np.frombuffer(base64.b64decode(pd["scales_b64"]), dtype=np.float32),
                payload_key: payload,
                "n_elements": pd["n_elements"],
                "n_blocks": n_blocks,
                "block_size": block_size,
            }
        else:
            params = {k: float(v) for k, v in d["params"].items()}

        residual = None
        if "residual_b64" in d:
            residual = np.frombuffer(
                base64.b64decode(d["residual_b64"]),
                dtype=d["residual_dtype"],
            ).reshape(d["residual_shape"])

        return cls(
            identity=d["identity"],
            function_type=func_type,
            params=params,
            residual=residual,
            accuracy=d.get("accuracy", 0.0),
            dtype=d.get("dtype", "float32"),
            shape=tuple(d.get("shape", [])),
        )
