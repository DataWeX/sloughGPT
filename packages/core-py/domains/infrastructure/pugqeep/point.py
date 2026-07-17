"""
Point — compressed data with meaning.

A Point is a compressed representation of any structured data, storing
a generator function instead of raw values. This enables:
  - Function-based generation (periodic, linear, polynomial)
  - Cluster-based generation (vector quantization)
  - Raw storage for incompressible data

Works with any numpy array: weight tensors, feature vectors, embeddings,
time series, sensor data, or any structured numerical data.
"""

import base64
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Point:
    """Compressed data with meaning — stores a generator function."""
    identity: str           # what this point represents
    function_type: str      # "periodic", "linear", "polynomial", "cluster", "raw"
    params: dict            # function parameters
    residual: Optional[np.ndarray] = None  # difference from exact
    accuracy: float = 0.0   # how well it fits (0-1)
    dtype: str = "float32"  # original data dtype
    shape: tuple = ()       # original data shape

    def generate(self, n: int) -> np.ndarray:
        """Generate n values from this point's function."""
        if self.function_type == "raw":
            raw_bytes = base64.b64decode(self.params["data_b64"])
            return np.frombuffer(raw_bytes, dtype=self.params["dtype"])

        if self.function_type == "cluster":
            centroids = self.params["centroids"]
            assignments = self.params["assignments"]
            return centroids[assignments[:n]]

        i = np.arange(n, dtype=np.float32)

        if self.function_type == "periodic":
            a, b, w = self.params["a"], self.params["b"], self.params["w"]
            values = a * np.cos(i) + b * np.sin(i) + w
        elif self.function_type == "linear":
            a, b = self.params["a"], self.params["b"]
            values = a * i + b
        elif self.function_type == "polynomial":
            a, b, c = self.params["a"], self.params["b"], self.params["c"]
            values = a * i**2 + b * i + c
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")

        if self.residual is not None:
            values += self.residual[:n]
        return values

    def nbytes(self) -> int:
        """Estimated memory usage of stored parameters."""
        if self.function_type == "cluster":
            return (self.params["centroids"].nbytes +
                    self.params["assignments"].nbytes +
                    (self.residual.nbytes if self.residual is not None else 0))
        elif self.function_type == "raw":
            return len(base64.b64decode(self.params["data_b64"]))
        else:
            return 4 + len(self.params) * 4 + (
                self.residual.nbytes if self.residual is not None else 0)

    # Type code mapping — fixed 4-byte headers for binary serialization
    _TYPE_CODES = {
        "periodic": b"PER ",
        "linear": b"LIN ",
        "polynomial": b"POLY",
        "cluster": b"CLUS",
        "raw": b"RAW ",
    }
    _TYPE_DECODE = {v: k for k, v in _TYPE_CODES.items()}

    def to_bytes(self) -> bytes:
        """Serialize point to bytes.

        Binary layout per type:
          header:  4 bytes type code (fixed)
          cluster: [n_centroids:uint32] [centroids:f32*n] [n_assignments:uint32] [assignments:u8*n] [has_residual:u8] [residual_bytes?]
          periodic/linear/polynomial: [params:f32*k] [has_residual:u8] [residual_bytes?]
          raw:     [n_bytes:uint32] [raw_data]
        """
        type_bytes = self._TYPE_CODES.get(self.function_type, b'\0\0\0\0')

        if self.function_type == "cluster":
            centroids = self.params["centroids"]
            assignments = self.params["assignments"]
            param_bytes = struct.pack('<I', len(centroids))
            param_bytes += centroids.astype(np.float32).tobytes()
            param_bytes += struct.pack('<I', len(assignments))
            param_bytes += assignments.tobytes()
            has_res = 1 if self.residual is not None else 0
            param_bytes += struct.pack('<B', has_res)
            if self.residual is not None:
                res_bytes = self.residual.astype(np.float32).tobytes()
                param_bytes += struct.pack('<I', len(res_bytes))
                param_bytes += res_bytes
        elif self.function_type == "periodic":
            param_bytes = struct.pack('fff', self.params["a"], self.params["b"], self.params["w"])
            has_res = 1 if self.residual is not None else 0
            param_bytes += struct.pack('<B', has_res)
            if self.residual is not None:
                res_bytes = self.residual.astype(np.float32).tobytes()
                param_bytes += struct.pack('<I', len(res_bytes))
                param_bytes += res_bytes
        elif self.function_type in ("linear", "polynomial"):
            param_bytes = struct.pack(f'{len(self.params)}f', *self.params.values())
            has_res = 1 if self.residual is not None else 0
            param_bytes += struct.pack('<B', has_res)
            if self.residual is not None:
                res_bytes = self.residual.astype(np.float32).tobytes()
                param_bytes += struct.pack('<I', len(res_bytes))
                param_bytes += res_bytes
        elif self.function_type == "raw":
            raw_data = base64.b64decode(self.params["data_b64"])
            param_bytes = struct.pack('<I', len(raw_data))
            param_bytes += raw_data
        else:
            param_bytes = b''

        return type_bytes + param_bytes

    @classmethod
    def from_bytes(cls, data: bytes, identity: str = "unknown") -> "Point":
        """Deserialize point from bytes."""
        type_bytes = data[:4]
        function_type = cls._TYPE_DECODE.get(type_bytes)
        if function_type is None:
            raise ValueError(f"Unknown type code: {type_bytes!r}")
        param_bytes = data[4:]

        residual = None

        if function_type == "periodic":
            a, b, w = struct.unpack('fff', param_bytes[:12])
            params = {"a": a, "b": b, "w": w}
            offset = 12
            if len(param_bytes) > offset:
                has_res = struct.unpack('<B', param_bytes[offset:offset + 1])[0]
                offset += 1
                if has_res:
                    res_len = struct.unpack('<I', param_bytes[offset:offset + 4])[0]
                    offset += 4
                    residual = np.frombuffer(param_bytes[offset:offset + res_len], dtype=np.float32)
        elif function_type == "linear":
            a, b = struct.unpack('ff', param_bytes[:8])
            params = {"a": a, "b": b}
            offset = 8
            if len(param_bytes) > offset:
                has_res = struct.unpack('<B', param_bytes[offset:offset + 1])[0]
                offset += 1
                if has_res:
                    res_len = struct.unpack('<I', param_bytes[offset:offset + 4])[0]
                    offset += 4
                    residual = np.frombuffer(param_bytes[offset:offset + res_len], dtype=np.float32)
        elif function_type == "polynomial":
            a, b, c = struct.unpack('fff', param_bytes[:12])
            params = {"a": a, "b": b, "c": c}
            offset = 12
            if len(param_bytes) > offset:
                has_res = struct.unpack('<B', param_bytes[offset:offset + 1])[0]
                offset += 1
                if has_res:
                    res_len = struct.unpack('<I', param_bytes[offset:offset + 4])[0]
                    offset += 4
                    residual = np.frombuffer(param_bytes[offset:offset + res_len], dtype=np.float32)
        elif function_type == "cluster":
            offset = 0
            n_centroids = struct.unpack('<I', param_bytes[offset:offset + 4])[0]
            offset += 4
            centroids = np.frombuffer(param_bytes[offset:offset + n_centroids * 4], dtype=np.float32)
            offset += n_centroids * 4
            n_assignments = struct.unpack('<I', param_bytes[offset:offset + 4])[0]
            offset += 4
            assignments = np.frombuffer(param_bytes[offset:offset + n_assignments], dtype=np.uint8)
            offset += n_assignments
            has_res = struct.unpack('<B', param_bytes[offset:offset + 1])[0]
            offset += 1
            if has_res:
                res_len = struct.unpack('<I', param_bytes[offset:offset + 4])[0]
                offset += 4
                residual = np.frombuffer(param_bytes[offset:offset + res_len], dtype=np.float32)
            params = {"centroids": centroids, "assignments": assignments}
        elif function_type == "raw":
            n_bytes = struct.unpack('<I', param_bytes[:4])[0]
            raw_data = param_bytes[4:4 + n_bytes]
            params = {"data_b64": base64.b64encode(raw_data).decode(),
                      "shape": [], "dtype": "float32"}
        else:
            raise ValueError(f"Unknown function type: {function_type}")

        return cls(
            identity=identity,
            function_type=function_type,
            params=params,
            residual=residual,
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
            d["params"] = {
                "centroids_b64": base64.b64encode(centroids.tobytes()).decode(),
                "centroids_shape": list(centroids.shape),
                "centroids_dtype": str(centroids.dtype),
                "assignments_b64": base64.b64encode(assignments.tobytes()).decode(),
                "assignments_shape": list(assignments.shape),
                "assignments_dtype": str(assignments.dtype),
            }
        elif self.function_type == "raw":
            d["params"] = dict(self.params)
        else:
            d["params"] = {k: float(v) for k, v in self.params.items()}

        if self.residual is not None:
            d["residual_b64"] = base64.b64encode(self.residual.tobytes()).decode()
            d["residual_shape"] = list(self.residual.shape)
            d["residual_dtype"] = str(self.residual.dtype)

        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Point":
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
        elif func_type == "raw":
            params = dict(d["params"])
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
