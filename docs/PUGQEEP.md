# pugqeep — Point-Graph-Queue System

**Processing queue on graphed files** — not a messaging queue.

Loads any model AND files of the same MIME type (behavior trees, Android OS configs,
knowledge graphs, weight files). Everything compresses to Points via VQ or function
fitting, regardless of origin.

**Location:** `packages/core-py/domains/infrastructure/pugqeep/`

## Architecture

```
PGQ (facade)
  ├── Engine — process dispatch, Trees, Stems
  ├── Tree / ModelTree — compresses files into Points
  │     └── PointLibrary — stores Points
  ├── TaskQueue — priority task execution
  ├── TieredCache — hot/memory/disk tiers
  └── PointCompressor — VQ / function fitting
```

| Component | Purpose |
|-----------|---------|
| **Point** | Compressed data unit (VQ cluster, function fit, or raw) |
| **PointProtocol** | ABC defining the contract for Points |
| **PointView** | Lazy decompression wrapper |
| **PointLibrary** | Thread-safe Point storage with search, batch ops, views |
| **Tree** | Generic file/data compressor — loads any array data into Points |
| **ModelTree** | Tree subclass with ML-specific skip logic (embeddings, biases) |
| **TaskQueue** | Priority task execution with worker pool |
| **Engine** | Process dispatch with Trees and Stems |
| **PGQ** | High-level facade combining all components |

## Quick start

```python
from domains.infrastructure.pugqeep import PGQ, Point, PointLibrary, Tree, ModelTree

# High-level facade
pgq = PGQ("my-model")
pgq.put("layer_0.weight", numpy_array)
data = pgq.get("layer_0.weight")

# Generic Tree — compress ANY numpy data (behavior trees, configs, graphs)
tree = Tree("behavior-tree", n_clusters=16)
tree.load_data({"node_0": arr_0, "node_1": arr_1, "edge_weights": arr_2})
restored = tree.get_data("node_0")

# ModelTree — ML-specific with skip logic for embeddings/biases
model_tree = ModelTree("gpt2", n_clusters=16)
model_tree.load_weights(model.state_dict(), num_workers=4)
weight = model_tree.get_weight("blocks.0.attn.c_attn.weight")

# Process management
future = engine.submit(my_fn, arg1, arg2)
result = future.result(timeout=10.0)
```

## Point interface

### PointProtocol (ABC)

Any Point implementation must satisfy:

| Attribute | Type | Description |
|-----------|------|-------------|
| `identity` | `str` | Unique identifier |
| `function_type` | `str \| FunctionType` | Compression method |
| `params` | `dict` | Function parameters |
| `accuracy` | `float` | Compression accuracy (0-1) |
| `residual` | `Optional[np.ndarray]` | Residual array |
| `dtype` | `str` | Original data dtype |
| `shape` | `tuple` | Original data shape |

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate` | `(n) -> np.ndarray` | Generate n values from stored function |
| `nbytes` | `() -> int` | Compressed size in bytes |
| `to_dict` | `() -> dict` | Serialize to dict |
| `to_bytes` | `() -> bytes` | Serialize to bytes |
| `from_dict` | `(d) -> PointProtocol` | Deserialize from dict |
| `from_bytes` | `(data, identity) -> PointProtocol` | Deserialize from bytes |

### FunctionType enum

| Value | Description |
|-------|-------------|
| `CLUSTER` | Vector quantization (centroids + assignments) |
| `LINEAR` | Linear fit (a*x + b) |
| `POLYNOMIAL` | Polynomial fit (a*x^2 + b*x + c) |
| `PERIODIC` | Periodic fit (a*cos + b*sin + w) |
| `RAW` | Uncompressed (stored as-is) |

### PointView (lazy decompression)

```python
from domains.infrastructure.pugqeep import PointLibrary

lib = PointLibrary("my-lib")
view = lib.view("layer_0.weight")  # no decompression yet

arr = view.generate()         # decompress full array now
arr = view[0:100]             # decompresses everything, then slices
len(view)                     # uncompressed element count
view.accuracy                 # compression accuracy
view.point.is_lossless        # True if accuracy == 1.0
```

## PointLibrary

Thread-safe Point storage with batch operations, search, and views.

```python
from domains.infrastructure.pugqeep import PointLibrary, Point

lib = PointLibrary("my-lib")

# Add / get
lib.add(Point(identity="w1", function_type="cluster", params={...}))
point = lib.get("w1")
found = lib.has("w1")  # or: "w1" in lib

# Batch ops
lib.add_many([point1, point2, point3])
points = lib.get_many(["w1", "w2", "w3"])
removed = lib.remove_many(["w1", "w2"])

# Search
all_points = lib.list_all()
cluster_points = lib.search_by_type("cluster")
layer_points = lib.search_by_type("cluster", "layer")

# Stats
stats = lib.stats()
# → {total_points, avg_accuracy, total_raw_bytes, total_compressed_bytes,
#    ratio, types, views_cached, ops}

# Lazy views
view = lib.view("w1")
best = lib.best_points(n=5)
worst = lib.worst_points(n=5)
```

### Iterator protocol

```python
for point in lib:
    print(point.identity, point.accuracy)

# Contains check
if "w1" in lib:
    print("found")
```

## Tree (generic)

Compresses ANY numpy data into Points — not just model weights.

```python
from domains.infrastructure.pugqeep import Tree

# Behavior tree nodes
tree = Tree("game-ai", n_clusters=16)
tree.load_data({
    "patrol_node": patrol_weights,
    "attack_node": attack_weights,
    "flee_node": flee_weights,
})
behavior = tree.get_data("patrol_node")

# Knowledge graph embeddings
tree = Tree("knowledge-graph", n_clusters=8)
tree.load_data(graph_embeddings)
```

## ModelTree (ML-specific)

Extends Tree with skip logic for embeddings and biases (discrete tensors
that shouldn't be VQ-compressed).

```python
from domains.infrastructure.pugqeep import ModelTree, save_library, load_library
from pathlib import Path
import numpy as np

tree = ModelTree("gpt2", n_clusters=16)

# Load weights (sequential)
weights = {"layer_0.weight": np.random.randn(768, 768)}
stats = tree.load_weights(weights)
# → {model, num_weights, total_raw_bytes, total_compressed_bytes, ratio, method}

# Load weights (parallel)
stats = tree.load_weights(weights, num_workers=4)

# Get weight (decompress on demand)
w = tree.get_weight("layer_0.weight")  # → np.ndarray

# Save / load library (module-level functions)
save_library(tree.library, Path("model.points.json"))
tree = load_library(Path("model.points.json"))
```

### Parallel operations

| Operation | Method | `num_workers` | Description |
|-----------|--------|---------------|-------------|
| Compress | `load_weights(..., num_workers=N)` | `0`=seq, `-1`=cpu_count | Parallel VQ compression |
| Decompress | `decompress_tree(tree, num_workers=N)` | `0`=seq, `-1`=cpu_count | Parallel decompression to dict |

## TaskQueue

Priority task execution with worker pool mode.

```python
from domains.infrastructure.pugqeep import TaskQueue, Task, TaskPriority

q = TaskQueue(name="training")

# Create and submit tasks
task = Task(name="train_epoch_1", data={"epochs": 1}, priority=TaskPriority.HIGH)
q.submit(task)

# Worker pool mode (auto-executes tasks)
q.start_workers(num_workers=4)

# Stats
stats = q.stats()  # {total, pending, running, completed, failed}
q.cancel(task.id)

# Shutdown
q.stop_workers(timeout=10.0)
```

### Task lifecycle

```
PENDING → RUNNING → COMPLETED
                    → FAILED (retries < max_retries → PENDING)
         → CANCELLED
```

### TaskPriority

| Value | Description |
|-------|-------------|
| `URGENT` | Highest priority |
| `HIGH` | Above normal |
| `NORMAL` | Default |
| `LOW` | Below normal |

## Engine

Process dispatch with Trees and Stems.

```python
from domains.infrastructure.pugqeep import Engine

engine = Engine("main")

# Spawn processes
proc = engine.spawn(my_function, arg1, arg2)
proc = engine.spawn(another_fn, priority=0)  # high priority

# Route to trees
engine.route("load_model", "data")
engine.route("train", "training")

# Worker pool
engine.start_workers(num_workers=4)
engine.spawn(fn, arg1, arg2, priority=1)  # use spawn, not submit

# Dispatch loop
engine.run(poll_interval=0.5)  # continuous
engine.dispatch()              # one-shot

# Shutdown
engine.stop_workers(timeout=10.0)
```

### Process lifecycle

```
CREATED → READY → RUNNING → COMPLETED
                         → FAILED
         → WAITING
         → CANCELLED
```

### Process

| Attribute | Type | Description |
|-----------|------|-------------|
| `fn` | `Callable` | Function to execute |
| `args` | `tuple` | Positional arguments |
| `kwargs` | `dict` | Keyword arguments |
| `id` | `str` | Unique identifier |
| `name` | `str` | Human-readable name |
| `status` | `ProcessStatus` | Current lifecycle state |
| `result` | `Any` | Return value (after completion) |
| `error` | `Optional[str]` | Error message (if failed) |
| `parent_id` | `Optional[str]` | Parent process ID |
| `children_ids` | `List[str]` | Child process IDs |

## Compression strategies

### Vector quantization (cluster)

Best for neural network weights (random-ish distributions).

```
Raw:       N × 4 bytes  (float32)
Compressed: k × 4 + N × 1 bytes  (centroids + uint8 assignments)
Ratio:     ~4:1 for large N
Accuracy:  ~95-99%
```

### Function fitting

Best for structured weights (periodic, linear, polynomial patterns).

```
Raw:       N × 4 bytes
Compressed: 8-12 bytes (2-3 coefficients)
Ratio:     ~100,000:1 for good fits
Accuracy:  ~80-95% (varies by pattern)
```

### When to use which

| Weight type | Best method | Typical ratio | Accuracy |
|-------------|-------------|---------------|----------|
| Neural net weights | `cluster` | 3-5:1 | 95-99% |
| Embedding tables | `raw` | 1:1 | 100% |
| Bias vectors (small) | `raw` | 1:1 | 100% |
| Attention patterns | `linear`/`polynomial` | 100-1000:1 | 80-95% |
| Positional encodings | `periodic` | 1000+:1 | 90-99% |

## Thread safety

- `PointLibrary` uses `threading.RLock` for all mutations
- `ProducerConsumerQueue` uses `queue.PriorityQueue` (thread-safe)
- `Engine._processes` and `Engine._trees` use `threading.Lock`
- `TaskQueue` operations are atomic (single-threaded dispatch)

## Integration with CancelManager

Long-running pugqeep operations wire into `CancelManager` for cancellation:

```python
from domains.infrastructure.cancel_manager import get_cancel_manager, OpType

mgr = get_cancel_manager()
op_id = mgr.register(OpType.TRAINING, "compress model")
mgr.start(op_id)

try:
    tree.load_weights(weights, num_workers=4)
    mgr.finish(op_id)
except Exception as e:
    mgr.finish(op_id, error=str(e))
```

## Tests

```bash
# All pugqeep tests
PYTHONPATH=. python3 -m pytest tests/test_pugqeep*.py tests/test_producer_consumer.py -v

# Specific suites
PYTHONPATH=. python3 -m pytest tests/test_pugqeep_point_interface.py -v  # Point protocol + views
PYTHONPATH=. python3 -m pytest tests/test_pugqeep_parallel.py -v         # Parallel batch ops
PYTHONPATH=. python3 -m pytest tests/test_pugqeep_producer_consumer.py -v # Integration tests
PYTHONPATH=. python3 -m pytest tests/test_producer_consumer.py -v         # Queue unit tests
```
