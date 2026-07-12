"""
PointWeight — weight representation using Points (generator functions).

Instead of storing weights as raw numpy arrays, SloLinear can use
PointWeight which wraps a Point (cluster, periodic, linear, polynomial).
During forward(), the Point generates the weight tensor on-the-fly.

This gives SloNet:
  - Compression (VQ clusters, function fits)
  - On-the-fly generation (no storage for generated weights)
  - Smooth interpolation (Points can be interpolated)
  - Meaning (each weight has an identity/function type)
"""

import numpy as np

from domains.infrastructure.pugqeep.point import Point
from domains.infrastructure.pugqeep.compressor import PointCompressor


class PointWeight:
    """Weight backed by a Point (generator function) instead of raw numpy.

    Usage:
        pw = PointWeight.from_array(weights, method="cluster")
        data = pw.generate()  # returns numpy array of shape self.shape
    """

    def __init__(self, point: Point, shape: tuple, dtype: str = "float32"):
        self.point = point
        self.shape = shape
        self.dtype = dtype
        self._cached: np.ndarray | None = None

    def generate(self) -> np.ndarray:
        """Generate weight array from Point. Caches after first call."""
        if self._cached is None:
            n = int(np.prod(self.shape))
            flat = self.point.generate(n)
            self._cached = flat.astype(self.dtype).reshape(self.shape)
        return self._cached

    def invalidate_cache(self):
        """Clear cached generated weights (e.g. after compression update)."""
        self._cached = None

    @property
    def data(self) -> np.ndarray:
        """Lazy generate — returns numpy array."""
        return self.generate()

    @classmethod
    def from_array(cls, weights: np.ndarray, identity: str = "weight",
                   method: str = "auto", n_clusters: int = 16) -> "PointWeight":
        """Compress a numpy weight array into a PointWeight.

        Args:
            weights: raw weight array
            identity: name for this weight (e.g. "blocks.0.attn.q_proj.weight")
            method: "cluster", "function", or "auto" (try both, pick best)
            n_clusters: number of VQ centroids (for cluster method)
        """
        compressor = PointCompressor()

        if method == "auto":
            # Try cluster first (better for NN weights), fall back to function
            p_cluster = compressor.compress_cluster(weights, identity, n_clusters)
            p_func = compressor.compress_function(weights, identity)
            point = p_cluster if p_cluster.accuracy >= p_func.accuracy else p_func
        else:
            point = compressor.compress(weights, identity, method)

        return cls(point, shape=weights.shape, dtype=str(weights.dtype))

    @classmethod
    def from_point(cls, point: Point, shape: tuple) -> "PointWeight":
        """Create PointWeight from an existing Point."""
        return cls(point, shape=shape, dtype=point.dtype)

    def nbytes(self) -> int:
        """Memory usage of stored Point (not generated array)."""
        return self.point.nbytes()

    def accuracy(self) -> float:
        """How well the Point fits the original data (0-1)."""
        return self.point.accuracy

    def __repr__(self) -> str:
        return (f"PointWeight(type={self.point.function_type}, "
                f"shape={self.shape}, acc={self.point.accuracy:.3f})")


def compress_slonet_to_points(model, method: str = "auto",
                              n_clusters: int = 16) -> dict:
    """Compress all SloNet/SloTransformer weights to PointWeights.

    Returns dict mapping weight name → PointWeight.
    """
    from domains.training.slonet import SloTransformer, SloNet

    points = {}
    compressor = PointCompressor()

    def _compress(name: str, arr: np.ndarray):
        if arr.size < 16:
            # Too small to compress — store raw
            from domains.infrastructure.pugqeep.point import Point as RawPoint
            raw_point = RawPoint(
                identity=name,
                function_type="raw",
                params={"data_b64": __import__("base64").b64encode(arr.tobytes()).decode(),
                        "dtype": str(arr.dtype)},
                accuracy=1.0,
                dtype=str(arr.dtype),
                shape=arr.shape,
            )
            points[name] = PointWeight(raw_point, shape=arr.shape, dtype=str(arr.dtype))
            return

        pw = PointWeight.from_array(arr, identity=name, method=method,
                                    n_clusters=n_clusters)
        points[name] = pw

    if hasattr(model, "parameters"):
        for param in model.parameters():
            name = getattr(param, "name", f"param_{id(param)}")
            _compress(name, param.data)

    # Also try named modules (for SloTransformer blocks)
    if hasattr(model, "named_modules"):
        for mod_name, module in model.named_modules():
            if hasattr(module, "weight") and hasattr(module.weight, "data"):
                _compress(f"{mod_name}.weight", module.weight.data)
            if hasattr(module, "bias") and hasattr(module.bias, "data"):
                _compress(f"{mod_name}.bias", module.bias.data)

    return points
