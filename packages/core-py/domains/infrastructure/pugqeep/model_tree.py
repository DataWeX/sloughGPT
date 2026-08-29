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
from typing import Callable, Dict, List, Optional, Tuple

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
        self.library = library if library is not None else PointLibrary(name=f"{name}_points")
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

    def load_weights(self, weights: Dict[str, np.ndarray], method: Optional[str] = None,
                     num_workers: int = 0,
                     on_progress: Optional[Callable[[int, int, str], None]] = None) -> dict:
        """Compress all weight tensors into Points and store in library.

        Args:
            weights: Dict of name → numpy array.
            method: Compression method ("cluster" or "function"). Defaults to self._method.
            num_workers: Parallel workers for compression. 0 = sequential, -1 = cpu_count.
            on_progress: Optional callback(completed, total, name) called per weight.

        Returns:
            Dict with compression stats.
        """
        if method is None:
            method = self._method

        # Small models or explicit sequential
        if num_workers == 0 or len(weights) <= 1:
            return self._load_weights_sequential(weights, method, on_progress)

        return self._load_weights_parallel(weights, method, num_workers, on_progress)

    def _load_weights_sequential(self, weights: Dict[str, np.ndarray], method: str,
                                 on_progress: Optional[Callable] = None) -> dict:
        """Sequential weight compression (original path)."""
        total_raw = 0
        total_compressed = 0

        for i, (name, raw) in enumerate(weights.items()):
            point, compressed_bytes = self._compress_weight(name, raw, method)
            self.library.add(point)
            self._weight_shapes[name] = raw.shape
            self._weight_dtypes[name] = raw.dtype
            total_raw += raw.nbytes
            total_compressed += compressed_bytes
            if on_progress is not None:
                on_progress(i + 1, len(weights), name)

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

    def _load_weights_parallel(self, weights: Dict[str, np.ndarray], method: str,
                               num_workers: int,
                               on_progress: Optional[Callable] = None) -> dict:
        """Parallel weight compression via ProducerConsumerQueue."""
        import threading
        from domains.infrastructure.producer_consumer import ProducerConsumerQueue

        if num_workers < 0:
            import os
            num_workers = os.cpu_count() or 4

        results: Dict[str, tuple] = {}
        lock = threading.Lock()
        total_done = [0]
        total_count = len(weights)

        def compress_one(item):
            name, raw = item
            point, compressed_bytes = self._compress_weight(name, raw, method)
            with lock:
                results[name] = (point, compressed_bytes)
                total_done[0] += 1
                done, total, n = total_done[0], total_count, name
            if on_progress is not None:
                on_progress(done, total, n)

        q = ProducerConsumerQueue(
            maxsize=num_workers * 4,
            num_consumers=num_workers,
            handler=compress_one,
            name=f"load-{self.name}",
        )
        q.start()
        try:
            for name, raw in weights.items():
                q.put((name, raw))
            import time
            while not q.empty:
                time.sleep(0.01)
        finally:
            q.stop(timeout=30)

        # Store results
        total_raw = 0
        total_compressed = 0
        for name, raw in weights.items():
            point, compressed_bytes = results[name]
            self.library.add(point)
            self._weight_shapes[name] = raw.shape
            self._weight_dtypes[name] = raw.dtype
            total_raw += raw.nbytes
            total_compressed += compressed_bytes

        self._loaded = True
        ratio = total_raw / max(total_compressed, 1)
        return {
            "model": self.name,
            "num_weights": len(weights),
            "total_raw_bytes": total_raw,
            "total_compressed_bytes": total_compressed,
            "ratio": ratio,
            "method": method,
            "workers": num_workers,
        }

    def _compress_weight(self, name: str, raw: np.ndarray, method: str) -> tuple:
        """Compress a single weight tensor. Returns (Point, compressed_bytes)."""
        point_id = f"{self.name}.{name}"
        flat = raw.flatten()

        # Skip embeddings and biases if configured
        skip = False
        if self._skip_embeddings and ("embed" in name.lower() or "embedding" in name.lower()):
            skip = True
        if self._skip_biases and name.lower().endswith("bias"):
            skip = True

        if skip:
            point = Point(
                identity=point_id,
                function_type="raw",
                params={"data_b64": base64.b64encode(raw.tobytes()).decode(),
                        "shape": list(raw.shape),
                        "dtype": str(raw.dtype)},
                accuracy=1.0,
            )
            compressed_bytes = raw.nbytes
        elif method == "cluster" and len(flat) < self.n_clusters * 2:
            point = Point(
                identity=point_id,
                function_type="raw",
                params={"data_b64": base64.b64encode(raw.tobytes()).decode(),
                        "shape": list(raw.shape),
                        "dtype": str(raw.dtype)},
                accuracy=1.0,
            )
            compressed_bytes = raw.nbytes
        elif method == "cluster":
            point = self._compressor.compress_cluster(flat, point_id, self.n_clusters)
            centroids = point.params["centroids"]
            assignments = point.params["assignments"]
            compressed_bytes = centroids.nbytes + assignments.nbytes
            if point.residual is not None:
                compressed_bytes += point.residual.nbytes
        else:
            point = self._compressor.compress_function(flat, point_id)
            compressed_bytes = 4 + len(point.params) * 4

        return point, compressed_bytes

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

    def get_weights(self, names: Optional[List[str]] = None) -> Dict[str, Optional[np.ndarray]]:
        """Retrieve multiple weight tensors by name.

        Args:
            names: List of weight names. None = return all loaded weights.

        Returns:
            Dict mapping weight names to numpy arrays (None if not found).
        """
        if names is None:
            names = list(self._weight_shapes.keys())
        return {name: self.get_weight(name) for name in names}

    def _estimate_size(self, name: str) -> int:
        shape = self._weight_shapes.get(name)
        if shape:
            return int(np.prod(shape))
        # Try to get from point metadata
        point = self.library.get(name)
        if point is not None and point.shape:
            return int(np.prod(point.shape))
        # For cluster points, estimate from assignments
        if point is not None and point.function_type == "cluster":
            return len(point.params.get("assignments", []))
        return 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @is_loaded.setter
    def is_loaded(self, value: bool) -> None:
        self._loaded = value

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
    """Load a HuggingFace model and compress its weights into Points.

    Requires ``numpy_engine._load_weights`` (loads config + safetensors from HF cache).
    """
    try:
        from domains.infrastructure.numpy_engine import _load_weights
    except ImportError:
        raise ImportError(
            "load_model_to_points requires numpy_engine._load_weights. "
            "Ensure safetensors and huggingface_hub are installed."
        )

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


def load_from_points(path: str) -> Tuple[ModelTree, dict]:
    """Load a ModelTree from a `.points.json` library file.

    Args:
        path: Base path without extension (e.g. ``/tmp/test_model``).
            The library is expected at ``<path>.points.json``.

    Returns:
        Tuple of (ModelTree, meta_dict).

    Raises:
        FileNotFoundError: If the library file does not exist.
    """
    p = Path(path)
    lib_path = p.with_suffix(".points.json")
    if not lib_path.exists():
        lib_path = p / "library.json"
        if not lib_path.exists():
            raise FileNotFoundError(f"Library not found: {p.with_suffix('.points.json')} or {p / 'library.json'}")

    library = PointLibrary.load(lib_path)
    model_name = p.stem

    meta_path = p.with_suffix(".meta.json") if p.suffix else p / "meta.json"
    if not meta_path.exists():
        meta_path = p.parent / f"{p.name}.meta.json"
    meta = {}
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())

    weight_shapes = meta.get("metadata", {}).get("weight_shapes", {})

    tree = ModelTree(model_name, library=library)
    tree._loaded = True

    for point in library.list_all():
        prefix = f"{model_name}."
        if point.identity.startswith(prefix):
            weight_name = point.identity[len(prefix):]
        else:
            weight_name = point.identity
        shape = tuple(weight_shapes.get(weight_name, weight_shapes.get(point.identity, [])))
        tree._weight_shapes[weight_name] = shape
        tree._weight_dtypes[weight_name] = np.dtype(point.dtype)

    return tree, meta


def decompress_tree(tree: ModelTree, num_workers: int = 0) -> Dict[str, np.ndarray]:
    """Decompress all weights from a ModelTree back to numpy arrays.

    Args:
        tree: A loaded ModelTree.
        num_workers: Parallel workers. 0 = sequential, -1 = cpu_count.

    Returns:
        Dict mapping weight names to their decompressed numpy arrays.
    """
    prefix = f"{tree.name}."

    items = []
    for point in tree.library.list_all():
        if point.identity.startswith(prefix):
            weight_name = point.identity[len(prefix):]
        else:
            weight_name = point.identity
        items.append((weight_name, point))

    if num_workers == 0 or len(items) <= 1:
        return _decompress_sequential(tree, items)

    return _decompress_parallel(tree, items, num_workers)


def _decompress_sequential(tree: ModelTree, items: list) -> Dict[str, np.ndarray]:
    """Sequential decompression."""
    weights = {}
    for weight_name, point in items:
        arr = _decompress_point(tree, weight_name, point)
        weights[weight_name] = arr
    return weights


def _decompress_parallel(tree: ModelTree, items: list, num_workers: int) -> Dict[str, np.ndarray]:
    """Parallel decompression via ProducerConsumerQueue."""
    import threading
    from domains.infrastructure.producer_consumer import ProducerConsumerQueue

    if num_workers < 0:
        import os
        num_workers = os.cpu_count() or 4

    results: Dict[str, np.ndarray] = {}
    lock = threading.Lock()

    def decompress_one(item):
        weight_name, point = item
        arr = _decompress_point(tree, weight_name, point)
        with lock:
            results[weight_name] = arr

    q = ProducerConsumerQueue(
        maxsize=num_workers * 4,
        num_consumers=num_workers,
        handler=decompress_one,
        name=f"decompress-{tree.name}",
    )
    q.start()
    try:
        for item in items:
            q.put(item)
        import time
        while not q.empty:
            time.sleep(0.01)
    finally:
        q.stop(timeout=30)

    return results


def _decompress_point(tree: ModelTree, weight_name: str, point: Point) -> np.ndarray:
    """Decompress a single point to numpy."""
    shape = tree._weight_shapes.get(weight_name, ())
    dtype = tree._weight_dtypes.get(weight_name, np.float32)

    if point.function_type == "raw":
        raw_bytes = base64.b64decode(point.params["data_b64"])
        arr = np.frombuffer(raw_bytes, dtype=point.params["dtype"])
        if point.params.get("shape"):
            arr = arr.reshape(point.params["shape"])
    else:
        n = point.params.get("n", 0) if "n" in point.params else (int(np.prod(shape)) if shape else 1000)
        arr = point.generate(n)
        if shape:
            arr = arr.reshape(shape)
    return arr.astype(dtype)
