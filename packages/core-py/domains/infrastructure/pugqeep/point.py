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

    def to_bytes(self) -> bytes:
        """Serialize point to bytes."""
        type_bytes = self.function_type[:4].encode().ljust(4, b'\0')

        if self.function_type == "cluster":
            centroids = self.params["centroids"]
            assignments = self.params["assignments"]
            param_bytes = struct.pack(f'{len(centroids)}f', *centroids)
            param_bytes += assignments.tobytes()
        elif self.function_type == "periodic":
            param_bytes = struct.pack('fff', self.params["a"], self.params["b"], self.params["w"])
        elif self.function_type in ("linear", "polynomial"):
            param_bytes = struct.pack(f'{len(self.params)}f', *self.params.values())
        else:
            param_bytes = b''

        return type_bytes + param_bytes

    @classmethod
    def from_bytes(cls, data: bytes, identity: str = "unknown") -> "Point":
        """Deserialize point from bytes."""
        type_bytes = data[:4]
        function_type = type_bytes.decode().rstrip('\0')
        param_bytes = data[4:]

        if function_type == "periodic":
            a, b, w = struct.unpack('fff', param_bytes[:12])
            params = {"a": a, "b": b, "w": w}
        elif function_type == "linear":
            a, b = struct.unpack('ff', param_bytes[:8])
            params = {"a": a, "b": b}
        elif function_type == "polynomial":
            a, b, c = struct.unpack('fff', param_bytes[:12])
            params = {"a": a, "b": b, "c": c}
        elif function_type == "cluster":
            raise NotImplementedError("Cluster deserialization needs metadata")
        else:
            raise ValueError(f"Unknown function type: {function_type}")

        return cls(identity=identity, function_type=function_type, params=params)

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
