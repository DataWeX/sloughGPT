"""
ModelTree — model instance using Points for inference.

This is the "Tree" in the Point-Graph-Queue architecture:
  - Selects points from a PointLibrary
  - Generates weights on demand from points
  - Maintains its own isolated state

Plus integration helpers: load_model_to_points, save_library, load_library.
"""

import base64
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .point import Point
from .compressor import PointCompressor
from .library import PointLibrary
from .config import TreeConfig, CompressorConfig

logger = logging.getLogger("slo.pugqeep")


class ModelTree:
    """Model instance that compresses weights into Points and runs inference.

    Args:
        name: Model identifier.
        library: Optional pre-existing PointLibrary.
        n_clusters: VQ cluster count.
        config: Optional TreeConfig (overrides n_clusters, method, skip_embeddings, skip_biases).
        compressor: Optional PointCompressor (overrides config's compressor settings).
    """

    def __init__(self, name: str, library: Optional[PointLibrary] = None,
                 n_clusters: int = 16, config: Optional[TreeConfig] = None,
                 compressor: Optional[PointCompressor] = None):
        self.name = name
        self.library = library or PointLibrary(name=f"{name}_points")
        if config is not None:
            self.n_clusters = config.n_clusters
            self._method = config.method
            self._skip_embeddings = config.skip_embeddings
            self._skip_biases = config.skip_biases
        else:
            self.n_clusters = n_clusters
            self._method = "cluster"
            self._skip_embeddings = True
            self._skip_biases = True
        self._compressor = compressor or PointCompressor(
            n_clusters=self.n_clusters)
        self._weight_shapes: Dict[str, Tuple[int, ...]] = {}
        self._weight_dtypes: Dict[str, np.dtype] = {}
        self._loaded = False

    def load_weights(self, weights: Dict[str, np.ndarray], method: Optional[str] = None) -> dict:
        """Compress all weight tensors into Points and store in library.

        Args:
            weights: Dict of name → numpy array.
            method: Compression method ("cluster" or "function"). Defaults to self._method.

        Returns:
            Dict with compression stats.
        """
        if method is None:
            method = self._method
        total_raw = 0
        total_compressed = 0

        for name, raw in weights.items():
            point_id = f"{self.name}.{name}"
            flat = raw.flatten()

            # Skip embeddings and biases if configured
            skip = False
            if self._skip_embeddings and ("embed" in name.lower() or "embedding" in name.lower()):
                skip = True
            if self._skip_biases and name.lower().endswith("bias"):
                skip = True

            if skip:
                # Store as raw (no compression for discrete/small tensors)
                point = Point(
                    identity=point_id,
                    function_type="raw",
                    params={"data_b64": base64.b64encode(raw.tobytes()).decode(),
                            "shape": list(raw.shape),
                            "dtype": str(raw.dtype)},
                    accuracy=1.0,
                )
            elif method == "cluster" and len(flat) < self.n_clusters * 2:
                point = Point(
                    identity=point_id,
                    function_type="raw",
                    params={"data_b64": base64.b64encode(raw.tobytes()).decode(),
                            "shape": list(raw.shape),
                            "dtype": str(raw.dtype)},
                    accuracy=1.0,
                )
            elif method == "cluster":
                point = self._compressor.compress_cluster(flat, point_id, self.n_clusters)
            else:
                point = self._compressor.compress_function(flat, point_id)

            self.library.add(point)
            self._weight_shapes[name] = raw.shape
            self._weight_dtypes[name] = raw.dtype
            total_raw += raw.nbytes

            if point.function_type == "cluster":
                centroids = point.params["centroids"]
                assignments = point.params["assignments"]
                total_compressed += centroids.nbytes + assignments.nbytes
                if point.residual is not None:
                    total_compressed += point.residual.nbytes
            elif point.function_type == "raw":
                total_compressed += raw.nbytes
            else:
                total_compressed += 4 + len(point.params) * 4

        self._loaded = True
        ratio = total_raw / max(total_compressed, 1)
        return {
            "model": self.name,
            "num_weights": len(weights),
            "total_raw_bytes": total_raw,
            "total_compressed_bytes": total_compressed,
            "ratio": ratio,
            "method": method,
        }

    def get_weight(self, name: str) -> Optional[np.ndarray]:
        point_id = f"{self.name}.{name}"
        point = self.library.get(point_id)
        if point is None:
            return None

        shape = self._weight_shapes.get(name)
        dtype = self._weight_dtypes.get(name, np.float32)

        if point.function_type == "raw":
            raw_bytes = base64.b64decode(point.params["data_b64"])
            arr = np.frombuffer(raw_bytes, dtype=point.params["dtype"])
            return arr.reshape(point.params["shape"])

        flat = point.generate(point.params.get("n", 0) if "n" in point.params
                              else self._estimate_size(name))
        if shape is not None:
            flat = flat.reshape(shape)
        return flat.astype(dtype)

    def _estimate_size(self, name: str) -> int:
        shape = self._weight_shapes.get(name)
        if shape:
            return int(np.prod(shape))
        return 1000

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def stats(self) -> dict:
        lib_stats = self.library.stats()
        return {
            "model": self.name,
            "loaded": self._loaded,
            "num_weights": len(self._weight_shapes),
            "library": lib_stats,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Integration helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_model_to_points(
    model_id: str,
    library: Optional[PointLibrary] = None,
    n_clusters: int = 16,
    method: str = "cluster",
    storage_dir: Optional[Path] = None,
) -> ModelTree:
    """Load a HuggingFace model and compress its weights into Points."""
    from domains.infrastructure.numpy_engine import _load_weights

    config, weights = _load_weights(model_id)

    if library is None:
        library = PointLibrary(
            name=model_id.replace("/", "_"),
            storage_dir=storage_dir,
        )

    tree = ModelTree(model_id, library, n_clusters=n_clusters)
    tree.load_weights(weights, method=method)
    return tree


def save_library(library: PointLibrary, path: Path) -> Path:
    """Save a PointLibrary to disk."""
    return library.save(path)


def load_library(path: Path) -> PointLibrary:
    """Load a PointLibrary from disk."""
    return PointLibrary.load(path)
