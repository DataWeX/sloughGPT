# pugqeep — Point-Graph-Queue

Processing queue for graphed files. Compress any structured data (model weights, knowledge graphs, time series) via vector quantization with Huffman encoding. Store, retrieve, and generate decompressed arrays on demand.

## Quick Start

```python
import numpy as np
from pugqeep import Tree

# Create a tree and load data
tree = Tree("my-model", n_clusters=64)
tree.load_data({
    "weights_1": np.random.randn(768, 3072),
    "weights_2": np.random.randn(3072, 768),
})

# Retrieve decompressed data
w1 = tree.get_data("weights_1")  # → np.ndarray, same shape as input

# Save / load from disk
tree.save("model.points.json")
tree2, meta = Tree.from_points("model.points.json")
```

## Core Types

### `Point`

Compressed representation of any numpy array. Stores a generator function instead of raw values.

```python
from pugqeep import Point

point.identity       # str — unique identifier ("model.layer1.weights")
point.function_type  # str — "cluster", "periodic", "linear", "polynomial", "raw"
point.params         # dict — function parameters (centroids, assignments, etc.)
point.accuracy       # float — 0.0 to 1.0
point.shape          # tuple — original array shape
point.dtype          # str — original dtype ("float32")

# Generate decompressed values
arr = point.generate(n)     # → np.ndarray of n values
arr = point.generate(1024)

# Size info
point.nbytes()              # compressed size in bytes
point.compression_ratio     # raw_bytes / compressed_bytes
point.is_lossless           # accuracy >= 1.0

# Serialization
d = point.to_dict()         # → JSON-compatible dict
point = Point.from_dict(d)

b = point.to_bytes()        # → compact binary bytes
point = Point.from_bytes(b, identity="layer1")
```

### `FunctionType`

Enum of supported compression methods:

```python
from pugqeep import FunctionType

FunctionType.CLUSTER      # Vector quantization (Lloyd's + Huffman)
FunctionType.LINEAR       # a*x + b
FunctionType.POLYNOMIAL   # a*x^2 + b*x + c
FunctionType.PERIODIC     # a*cos(x) + b*sin(x) + w
FunctionType.RAW          # Incompressible — stored as-is
```

### `PointView`

Lazy decompression wrapper. Holds metadata, generates numpy on demand.

```python
view = library.view("model.layer1", shape=(768, 3072), dtype="float32")

arr = view.generate()    # decompress (cached after first call)
arr = view[:]            # same as generate()
arr = view[0:100]        # partial decompression (fast path for cluster)
view.clear_cache()       # free memory
```

## `Tree`

Generic tree that compresses arrays into Points and generates on demand. Works for any numpy data — model weights, knowledge graphs, sensor data.

```python
from pugqeep import Tree

tree = Tree("my-tree", n_clusters=64)

# Load data (sequential)
stats = tree.load_data({
    "layer1.weights": np.random.randn(768, 3072),
    "layer1.bias": np.random.randn(768),
})
# → {"tree": "my-tree", "num_items": 2, "ratio": 5.6, "method": "cluster", ...}

# Load data (parallel)
stats = tree.load_data(data_dict, num_workers=4, on_progress=lambda done, total, name: print(f"{done}/{total} {name}"))

# Retrieve
arr = tree.get_data("layer1.weights")           # → np.ndarray
batch = tree.get_data_batch()                    # → dict of all arrays
batch = tree.get_data_batch(["layer1.weights"])  # → dict of named arrays

# Check / list / remove
tree.has("layer1.weights")      # → bool
tree.list_items()               # → ["layer1.weights", "layer1.bias"]
tree.remove("layer1.bias")      # → bool

# Persistence
tree.save("model.points.json")
tree, meta = Tree.from_points("model.points.json")

# Stats
tree.stats()
# → {"tree": "my-tree", "loaded": True, "num_items": 2, "library": {...}}
```

### Custom compression strategy

```python
from pugqeep import Tree, PointCompressor
from pugqeep.config import CompressorConfig

comp = PointCompressor(CompressorConfig(n_clusters=128))
tree = Tree("my-tree", compressor=comp, n_clusters=128)
```

## `ModelTree`

ML-specialized Tree. Skips VQ for embedding layers and bias tensors automatically.

```python
from pugqeep import ModelTree

tree = ModelTree("llama-7b", n_clusters=64)

# Same API as Tree
tree.load_weights(weights_dict)
w = tree.get_weight("model.embed_tokens.weight")
```

Skip logic (configurable via `TreeConfig`):
- `skip_embeddings=True` — layers with "embed" in name → raw storage
- `skip_biases=True` — layers ending in "bias" → raw storage

## `PointLibrary`

Thread-safe store for Points. The "Graph" in Point-Graph-Queue.

```python
from pugqeep import PointLibrary

lib = PointLibrary("my-lib")

# CRUD
lib.add(point)                    # → True (new) / False (replaced)
lib.get("model.layer1")           # → Point or None
lib.has("model.layer1")           # → bool
lib.remove("model.layer1")        # → bool

# Batch ops
lib.add_many([point1, point2])    # → count added
lib.get_many(["id1", "id2"])      # → {"id1": Point, "id2": None}

# Listing
lib.list_all()                    # → [Point, Point, ...]
lib.list_by_type("cluster")       # → [Point, ...]
lib.list_identities()             # → ["id1", "id2", ...]
lib.list_types()                  # → {"cluster": 5, "raw": 2}

# Search
lib.search("layer1")              # → [Point, ...] (case-insensitive substring)
lib.best_points(10)               # → top 10 by accuracy
lib.worst_points(10)              # → bottom 10 by accuracy

# Lazy views
view = lib.view("model.layer1", shape=(768, 3072))
arr = view.generate()

# Compress & store directly
point = lib.compress_and_store(weights, "layer1", method="cluster", n_clusters=64)

# Persistence
lib.save("library.json")
lib = PointLibrary.load("library.json")

# Stats
lib.stats()
# → {"total_points": 10, "total_raw_bytes": 18_000_000, "total_compressed_bytes": 3_200_000, "ratio": 5.6, ...}
```

## `PointCompressor`

Low-level compression engine. Lloyd's algorithm with quantile initialization + Huffman encoding.

```python
from pugqeep import PointCompressor
from pugqeep.config import CompressorConfig

comp = PointCompressor(CompressorConfig(n_clusters=64))

# Vector quantization (recommended)
point = comp.compress_cluster(flat_array, "layer1", n_clusters=64)
# → Point with centroids, Huffman-encoded assignments

# Function fitting
point = comp.compress_function(flat_array, "layer1")
# → Point with fitted function (linear, polynomial, or periodic)

# Block quantization
point = comp.compress_block_q4(flat_array, "layer1")  # 4-bit, ~5.3:1 ratio
point = comp.compress_block_q8(flat_array, "layer1")  # 8-bit, ~3.2:1 ratio

# Decompress block-quantized points
arr = comp.decompress_block_q4(point)
arr = comp.decompress_block_q8(point)
```

### Compression benchmarks (768×3072 + 3072×768 weights, ~18.9MB raw)

| Method | Accuracy | Ratio | Time |
|--------|----------|-------|------|
| Lloyd's + Huffman k=64 | 99.9% | 5.6:1 | ~10s |
| Lloyd's + Huffman k=128 | 99.98% | 4.8:1 | ~19s |
| Q4 block | 99.4% | 5.3:1 | ~0.08s |
| Q8 block | 99.998% | 3.2:1 | ~0.08s |

## `PGQ`

Top-level facade. Manages data across tiers (Disk → Hot → Memory) with compression, caching, and task queuing.

```python
from pugqeep import PGQ
import numpy as np

# Create system
pgq = PGQ("my-system", n_clusters=64)

# Store data (compressed)
pgq.put("weights", np.random.randn(768, 3072))
pgq.put("biases", np.random.randn(768), compress=False)  # raw cache

# Retrieve data
arr = pgq.get("weights")  # → np.ndarray

# Batch operations
pgq.put_many({"w1": arr1, "w2": arr2})
batch = pgq.get_many(["w1", "w2"])

# Task management
from pugqeep import Task, TaskPriority
task = Task(id="job-1", name="train", priority=TaskPriority.HIGH)
pgq.submit_task(task)
pgq.list_tasks()
pgq.complete_task("job-1", result={"loss": 0.5})

# Spawn processes
pgq.spawn(my_function, arg1, arg2, name="worker-1")
pgq.run()

# Persistence
pgq.save("system.points.json")
pgq = PGQ.load("system.points.json")

# Factory methods
pgq = PGQ.from_model("meta-llama/Llama-2-7b", n_clusters=64)
pgq = PGQ.from_file("system.points.json")
```

## `ModelQueue`

Manages multiple Trees (one per loaded model). Supports deduplication across trees.

```python
from pugqeep import ModelQueue
from pugqeep.config import QueueConfig

queue = ModelQueue(QueueConfig(max_trees=10, dedup=True))

# Add trees
queue.add_tree("model-a")
queue.add_tree("model-b")

# Load HuggingFace models directly
queue.load_model("meta-llama/Llama-2-7b", n_clusters=64)

# List / get / remove
queue.list_trees()           # → ["model-a", "model-b"]
queue.get_tree("model-a")    # → ModelTree
queue.remove_tree("model-b") # → bool

# Dedup identical points across trees
queue.deduplicate()          # → {"merged": 5, "bytes_saved": 1024, "groups": 3}

# Stats
queue.stats()
# → {"num_trees": 2, "total_points": 50, "ratio": 4.8, ...}

# Persistence
queue.save_all("models/")
queue.load_all("models/")
```

## Config

All config classes are `@dataclass(slots=True)`:

```python
from pugqeep.config import (
    PointConfig,        # function_type, n_clusters, residual_threshold
    CompressorConfig,   # n_clusters, lloyd_iterations, gap_fill_*, method
    LibraryConfig,      # name, storage_dir, auto_save
    TreeConfig,         # name, n_clusters, method, skip_embeddings, skip_biases
    QueueConfig,        # max_trees, default_n_clusters, storage_dir, dedup
    EngineConfig,       # name, max_trees, tree_workers, max_stems, queue_size
    SubprocessConfig,   # enabled, python_exe, max_workers, memory_limit_mb
    RestartPolicy,      # max_restarts, restart_delay, backoff, max_backoff
    MonitorConfig,      # enabled, poll_interval, stall_timeout, on_stall
)
```

## Architecture

```
PGQ (facade — tiered cache + task queue + engine)
  └── Tree / ModelTree (compresses arrays into Points)
        └── PointLibrary (thread-safe store, persistence)
              └── Point (compressed data + generator function)
```

## Thread Safety

- `PointLibrary`: thread-safe (RLock on all CRUD)
- `PointCompressor`: NOT thread-safe (create one per thread)
- `Tree`: NOT thread-safe (wraps PointLibrary, which is)
- `PGQ`: NOT thread-safe (use from single thread or wrap calls)

## Performance Notes

- **Lloyd's + Huffman** (default): Best accuracy-to-ratio. ~10s for 19MB on CPU. One-time cost.
- **Q4 block**: Fastest compression (~0.08s). Good for live streams.
- **Q8 block**: Near lossless. Best for archival.
- **Quantile init**: Default. 4x faster than k-means++ with identical accuracy.
- **Huffman encoding**: Lossless compression on VQ assignments. Adds ~1.6:1 on top of VQ.
