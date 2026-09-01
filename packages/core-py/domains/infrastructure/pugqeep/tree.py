"""
Tree — generic tree instance using Points for storage and generation.

Full engine: compress, store, retrieve, decompress arrays as Points.
Works for model weights, knowledge graphs, any array data.
ModelTree extends this with ML-specific skip logic (embeddings, biases).
"""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from .point import Point
from .compressor import PointCompressor
from .library import PointLibrary
from .config import TreeConfig
from .strategies import CompressStrategy, ClusterStrategy, FunctionStrategy, RawStrategy, BlockQuantStrategy
from .executor import ParallelExecutor

logger = logging.getLogger("slo.pugqeep")


class Tree:
    """Generic tree that compresses arrays into Points and generates on demand.

    Args:
        name: Tree identifier.
        library: Optional pre-existing PointLibrary.
        n_clusters: VQ cluster count.
        config: Optional TreeConfig (overrides n_clusters, method).
        compressor: Optional PointCompressor (overrides config's compressor settings).
        strategy: Optional CompressStrategy (overrides auto-selection from method).
        executor: Optional ParallelExecutor (overrides default parallel config).
    """

    __slots__ = ('name', 'library', 'n_clusters', '_method', '_compressor',
                 '_strategy', '_executor', '_shapes', '_dtypes', '_loaded')

    # Strategy registry: method name → strategy class
    _STRATEGIES: Dict[str, Type[CompressStrategy]] = {
        "cluster": ClusterStrategy,
        "function": FunctionStrategy,
        "block_q4": lambda c: BlockQuantStrategy(c, bits=4),
        "block_q8": lambda c: BlockQuantStrategy(c, bits=8),
    }

    def __init__(self, name: str, library: Optional[PointLibrary] = None,
                 n_clusters: int = 16, config: Optional[TreeConfig] = None,
                 compressor: Optional[PointCompressor] = None,
                 strategy: Optional[CompressStrategy] = None,
                 executor: Optional[ParallelExecutor] = None):
        self.name = name
        self.library = library if library is not None else PointLibrary(name=f"{name}_points")
        if config is not None:
            self.n_clusters = config.n_clusters
            self._method = config.method
        else:
            self.n_clusters = n_clusters
            self._method = "cluster"
        self._compressor = compressor or PointCompressor(n_clusters=self.n_clusters)
        self._strategy = strategy or self._make_strategy(self._method)
        self._executor = executor or ParallelExecutor()
        self._shapes: Dict[str, Tuple[int, ...]] = {}
        self._dtypes: Dict[str, np.dtype] = {}
        self._loaded = False

    def _make_strategy(self, method: str) -> CompressStrategy:
        cls = self._STRATEGIES.get(method)
        if cls is None:
            return RawStrategy()
        if callable(cls) and not isinstance(cls, type):
            return cls(self._compressor)
        return cls(self._compressor)

    # Backward-compat aliases
    @property
    def _weight_shapes(self):
        return self._shapes

    @property
    def _weight_dtypes(self):
        return self._dtypes

    # ── Batch loading ──

    def load_data(self, data: Dict[str, np.ndarray], method: Optional[str] = None,
                  num_workers: int = 0,
                  on_progress: Optional[Callable[[int, int, str], None]] = None) -> dict:
        """Compress all arrays into Points and store in library.

        Args:
            data: Dict of name → numpy array.
            method: Compression method ("cluster" or "function"). Defaults to self._method.
            num_workers: Parallel workers. 0 = sequential, -1 = cpu_count.
            on_progress: Optional callback(completed, total, name).

        Returns:
            Dict with compression stats.
        """
        if method is None:
            method = self._method

        strategy = self._make_strategy(method) if method != self._method else self._strategy

        if num_workers == 0 or len(data) <= 1:
            return self._load_sequential(data, strategy, method, on_progress)

        return self._load_parallel(data, strategy, method, num_workers, on_progress)

    # Backward-compat alias
    load_weights = load_data

    def _load_sequential(self, data: Dict[str, np.ndarray], strategy: CompressStrategy,
                         method: str, on_progress: Optional[Callable] = None) -> dict:
        total_raw = 0
        total_compressed = 0

        for i, (name, raw) in enumerate(data.items()):
            point, compressed_bytes = self._compress_item(name, raw, strategy)
            self.library.add(point)
            self._shapes[name] = raw.shape
            self._dtypes[name] = raw.dtype
            total_raw += raw.nbytes
            total_compressed += compressed_bytes
            if on_progress is not None:
                on_progress(i + 1, len(data), name)

        self._loaded = True
        ratio = total_raw / max(total_compressed, 1)
        return {
            "tree": self.name,
            "num_items": len(data),
            "total_raw_bytes": total_raw,
            "total_compressed_bytes": total_compressed,
            "ratio": ratio,
            "method": method,
        }

    def _load_parallel(self, data: Dict[str, np.ndarray], strategy: CompressStrategy,
                       method: str, num_workers: int,
                       on_progress: Optional[Callable] = None) -> dict:
        executor = ParallelExecutor(num_workers)
        total_done = [0]
        total_count = len(data)
        results: Dict[str, tuple] = {}

        def compress_one(item):
            name, raw = item
            point, compressed_bytes = self._compress_item(name, raw, strategy)
            results[name] = (point, compressed_bytes, raw)
            total_done[0] += 1
            if on_progress is not None:
                on_progress(total_done[0], total_count, name)

        items = list(data.items())
        executor.run(items, compress_one, name=f"load-{self.name}")

        total_raw = 0
        total_compressed = 0
        for name, raw in data.items():
            point, compressed_bytes, _ = results[name]
            self.library.add(point)
            self._shapes[name] = raw.shape
            self._dtypes[name] = raw.dtype
            total_raw += raw.nbytes
            total_compressed += compressed_bytes

        self._loaded = True
        ratio = total_raw / max(total_compressed, 1)
        return {
            "tree": self.name,
            "num_items": len(data),
            "total_raw_bytes": total_raw,
            "total_compressed_bytes": total_compressed,
            "ratio": ratio,
            "method": method,
            "workers": executor.num_workers,
        }

    # ── Compression hook (overridable by subclasses) ──

    def _compress_item(self, name: str, raw: np.ndarray, strategy: CompressStrategy) -> tuple:
        """Compress a single array. Returns (Point, compressed_bytes).

        Subclasses can override to add skip logic or custom handling.
        Falls back to raw storage for arrays too small to cluster.
        """
        point_id = f"{self.name}.{name}"
        flat = raw.flatten()
        if isinstance(strategy, ClusterStrategy) and len(flat) <= self.n_clusters:
            return RawStrategy().compress(name, raw, point_id, self.n_clusters)
        return strategy.compress(name, raw, point_id, self.n_clusters)

    # ── Retrieval ──

    def get_data(self, name: str) -> Optional[np.ndarray]:
        """Retrieve a decompressed array by name."""
        point_id = f"{self.name}.{name}"
        point = self.library.get(point_id)
        if point is None:
            return None

        shape = self._shapes.get(name)
        dtype = self._dtypes.get(name, np.float32)

        if point.function_type == "raw":
            raw_bytes = base64.b64decode(point.params["data_b64"])
            arr = np.frombuffer(raw_bytes, dtype=point.params["dtype"])
            if point.params.get("shape"):
                arr = arr.reshape(point.params["shape"])
            return arr

        flat = point.generate(point.params.get("n", 0) if "n" in point.params
                              else self._estimate_size(name))
        if shape is not None:
            flat = flat.reshape(shape)
        return flat.astype(dtype)

    def get_data_batch(self, names: Optional[List[str]] = None) -> Dict[str, Optional[np.ndarray]]:
        """Retrieve multiple arrays by name."""
        if names is None:
            names = list(self._shapes.keys())
        return {name: self.get_data(name) for name in names}

    # Backward-compat aliases
    get_weight = get_data
    get_weights = get_data_batch

    def _estimate_size(self, name: str) -> int:
        shape = self._shapes.get(name)
        if shape:
            return int(np.prod(shape))
        point = self.library.get(name)
        if point is not None and point.shape:
            return int(np.prod(point.shape))
        if point is not None and point.function_type == "cluster":
            return len(point.params.get("assignments", []))
        return 0

    # ── Lifecycle ──

    def store(self, point: Point) -> Point:
        """Store a Point in the library."""
        self.library.add(point)
        return point

    def get(self, name: str) -> Optional[Point]:
        """Get a Point by identity."""
        return self.library.get(name)

    def has(self, name: str) -> bool:
        """Check if a Point exists."""
        return self.library.get(name) is not None

    def remove(self, name: str) -> bool:
        """Remove a Point by identity."""
        removed = self.library.remove(name)
        self._shapes.pop(name, None)
        self._dtypes.pop(name, None)
        return removed

    def list_items(self) -> List[str]:
        """List all item names."""
        prefix = f"{self.name}."
        names = []
        for point in self.library.list_all():
            if point.identity.startswith(prefix):
                names.append(point.identity[len(prefix):])
            else:
                names.append(point.identity)
        return names

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @is_loaded.setter
    def is_loaded(self, value: bool) -> None:
        self._loaded = value

    def stats(self) -> dict:
        lib_stats = self.library.stats()
        return {
            "tree": self.name,
            "loaded": self._loaded,
            "num_items": len(self._shapes),
            "num_weights": len(self._shapes),
            "library": lib_stats,
        }

    # ── Decompression (instance method) ──

    def decompress(self, num_workers: int = 0) -> Dict[str, np.ndarray]:
        """Decompress all data back to numpy arrays."""
        return decompress_tree(self, num_workers)

    # ── Persistence (class methods) ──

    @classmethod
    def from_points(cls, path: str) -> Tuple["Tree", dict]:
        """Load a Tree from a `.points.json` library file."""
        return load_from_points(path)

    def save(self, path: Path) -> Path:
        """Save this tree's library to disk."""
        return self.library.save(path)

    @classmethod
    def load_library(cls, path: Path) -> PointLibrary:
        """Load a PointLibrary from disk."""
        return PointLibrary.load(path)


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers (backward compat)
# ══════════════════════════════════════════════════════════════════════════════

def decompress_tree(tree: Tree, num_workers: int = 0) -> Dict[str, np.ndarray]:
    """Decompress all data from a Tree back to numpy arrays."""
    prefix = f"{tree.name}."

    items = []
    for point in tree.library.list_all():
        if point.identity.startswith(prefix):
            item_name = point.identity[len(prefix):]
        else:
            item_name = point.identity
        items.append((item_name, point))

    if num_workers == 0 or len(items) <= 1:
        return {name: _decompress_point(tree, name, pt) for name, pt in items}

    # Parallel: items already have stripped names from the prefix loop above.
    # executor.map() unwraps (key, value) so fn receives just the Point.
    # We need the stripped name, so process directly with executor.run().
    results: Dict[str, np.ndarray] = {}
    lock = __import__("threading").Lock()

    def _decomp(item):
        name, pt = item
        arr = _decompress_point(tree, name, pt)
        with lock:
            results[name] = arr

    executor = ParallelExecutor(num_workers)
    executor.run(items, _decomp, name=f"decompress-{tree.name}")
    return results


def _decompress_point(tree: Tree, item_name: str, point: Point) -> np.ndarray:
    shape = tree._shapes.get(item_name)
    if not shape:
        shape = point.shape if point.shape else ()
    dtype = tree._dtypes.get(item_name, np.float32)

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


def save_library(library: PointLibrary, path: Path) -> Path:
    """Save a PointLibrary to disk."""
    return library.save(path)


def load_library(path: Path) -> PointLibrary:
    """Load a PointLibrary from disk."""
    return PointLibrary.load(path)


def load_from_points(path: str) -> Tuple[Tree, dict]:
    """Load a Tree from a `.points.json` library file."""
    p = Path(path)
    lib_path = p.with_suffix(".points.json")
    if not lib_path.exists():
        lib_path = p / "library.json"
        if not lib_path.exists():
            raise FileNotFoundError(f"Library not found: {p.with_suffix('.points.json')} or {p / 'library.json'}")

    library = PointLibrary.load(lib_path)
    tree_name = p.stem

    meta_path = p.with_suffix(".meta.json") if p.suffix else p / "meta.json"
    if not meta_path.exists():
        meta_path = p.parent / f"{p.name}.meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    saved_shapes = meta.get("metadata", {}).get("weight_shapes", {})

    tree = Tree(tree_name, library=library)
    tree._loaded = True

    for point in library.list_all():
        prefix = f"{tree_name}."
        if point.identity.startswith(prefix):
            item_name = point.identity[len(prefix):]
        else:
            item_name = point.identity
        shape = tuple(saved_shapes.get(item_name, saved_shapes.get(point.identity, [])))
        tree._shapes[item_name] = shape
        tree._dtypes[item_name] = np.dtype(point.dtype)

    return tree, meta


def load_model_to_points(
    model_id: str,
    library: Optional[PointLibrary] = None,
    n_clusters: int = 16,
    method: str = "cluster",
    storage_dir: Optional[Path] = None,
) -> Tree:
    """Load a HuggingFace model and compress its weights into Points."""
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

    tree = Tree(model_id, library, n_clusters=n_clusters)
    tree.load_data(weights, method=method)
    return tree
