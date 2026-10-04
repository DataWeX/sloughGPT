# pugqeep — Point-Graph-Queue System

**Compressed persistence for numeric arrays** — not a messaging queue, and not a
grab-bag for "any file type". pugqeep compresses **arrays** (any component that
reduces to a `numpy.ndarray`) into `Point`s, then stores, indexes, queues, and
dispatches work over those Points.

**Location:** `packages/core-py/domains/infrastructure/pugqeep/`

## The three invariants

1. **Arrays → Points via a pluggable encoder.** A `Point` stores a _generator
   function_ instead of raw values. The encoder is one of: Vector Quantization
   (`cluster`), analytic fit (`linear` / `polynomial` / `periodic`), or `raw`
   (incompressible data stored as-is). "Any model or file of the same MIME type"
   is NOT a claim pugqeep makes — the claim is _"any component that reduces to a
   numpy array"_. Behavior trees, configs, graphs, and weights all qualify **only
   insofar as they are arrays**; the system does not parse their domain formats.

2. **`Point` / `PointProtocol` is the compaction boundary.** Everything upstream
   (encoder, `Tree`/`ModelTree`) produces Points; everything downstream (library,
   cache, views, serialization) consumes Points. A Point is lossy _unless_
   `accuracy == 1.0` (`point.is_lossless`), and carries its own `residual`
   (difference from exact) plus `dtype`/`shape` so reconstruction is faithful
   within the error budget.

3. **Library, cache, and queue are orthogonal infra behind the `PGQ` facade.**
   `PointLibrary` (storage/search/views), `TieredCache` (hot → memory → disk),
   and `TaskQueue`/`Engine` (priority execution) are independent concerns that
   `PGQ` composes — not features of compression itself.

### Error budget → raw fallback (the decode contract)

The encoder is chosen by fit: if an analytic/cluster fit meets the configured
accuracy gate it is stored; otherwise `raw` is used (lossless, 1:1). Embeddings
and small discrete tensors (biases) default to `raw` — see ModelTree. The `raw`
fallback is what guarantees the system degrades to _lossless_, never to _wrong_.

## Architecture

```
PGQ (facade)
  ├── Engine — process dispatch, Pools, Stems
  ├── Pipe — bounded execution admission (ProcessQueue + framed channel)
  ├── Tree / ModelTree — compresses arrays into Points
  │     └── PointLibrary — stores Points
  ├── TaskQueue — priority task execution
  ├── TieredCache — hot/memory/disk tiers
  └── PointCompressor — encoder: VQ | analytic fit | raw
```

| Component         | Purpose                                                     |
| ----------------- | ----------------------------------------------------------- |
| **Point**         | Compressed data unit (VQ cluster, function fit, or raw)     |
| **PointProtocol** | ABC defining the contract for Points                        |
| **PointView**     | Lazy decompression wrapper                                  |
| **PointLibrary**  | Thread-safe Point storage with search, batch ops, views     |
| **Tree**          | Generic compressor — loads any numpy array data into Points |
| **ModelTree**     | Tree subclass, ML-specific; skips VQ for embeddings/biases  |
| **TaskQueue**     | Priority task execution with worker pool                    |
| **Engine**        | Process dispatch with Pools and Stems                       |
| **PGQ**           | High-level facade combining all components                  |

## Quick start

```python
from domains.infrastructure.pugqeep import PGQ, Point, PointLibrary, Tree, ModelTree

# High-level facade
pgq = PGQ("my-model")
pgq.put("layer_0.weight", numpy_array)
data = pgq.get("layer_0.weight")

# Generic Tree — compress any numpy array
tree = Tree("game-ai", n_clusters=16)
tree.load_data({"patrol_node": arr_0, "attack_node": arr_1, "edge_weights": arr_2})
restored = tree.get_data("patrol_node")

# ModelTree — ML-specific: skips VQ for embeddings/biases
model_tree = ModelTree("gpt2", n_clusters=16)
model_tree.load_weights(model.state_dict(), num_workers=4)
weight = model_tree.get_weight("blocks.0.attn.c_attn.weight")

# Process management
proc = engine.spawn(my_fn, arg1, arg2)
engine.wait(timeout=10.0)
print(proc.result)
```

## Point interface

### PointProtocol (ABC)

Any Point implementation must satisfy:

| Attribute       | Type                   | Description                |
| --------------- | ---------------------- | -------------------------- |
| `identity`      | `str`                  | Unique identifier          |
| `function_type` | `str \| FunctionType`  | Compression method         |
| `params`        | `dict`                 | Function parameters        |
| `accuracy`      | `float`                | Compression accuracy (0-1) |
| `residual`      | `Optional[np.ndarray]` | Residual array             |
| `dtype`         | `str`                  | Original data dtype        |
| `shape`         | `tuple`                | Original data shape        |

| Method       | Signature                           | Description                            |
| ------------ | ----------------------------------- | -------------------------------------- |
| `generate`   | `(n) -> np.ndarray`                 | Generate n values from stored function |
| `nbytes`     | `() -> int`                         | Compressed size in bytes               |
| `to_dict`    | `() -> dict`                        | Serialize to dict                      |
| `to_bytes`   | `() -> bytes`                       | Serialize to bytes                     |
| `from_dict`  | `(d) -> PointProtocol`              | Deserialize from dict                  |
| `from_bytes` | `(data, identity) -> PointProtocol` | Deserialize from bytes                 |

### FunctionType enum

| Value        | Description                                   |
| ------------ | --------------------------------------------- |
| `CLUSTER`    | Vector quantization (centroids + assignments) |
| `LINEAR`     | Linear fit (a*x + b)                          |
| `POLYNOMIAL` | Polynomial fit (a*x^2 + b*x + c)              |
| `PERIODIC`   | Periodic fit (a*cos + b*sin + w)              |
| `RAW`        | Uncompressed (stored as-is)                   |

### PointView (lazy decompression)

```python
from domains.infrastructure.pugqeep import PointLibrary

lib = PointLibrary("my-lib")
view = lib.view("layer_0.weight")  # no decompression yet

arr = view.generate()  # decompress full array now
arr = view[0:100]  # lazy: only the requested slice is reconstructed
len(view)  # uncompressed element count
view.accuracy  # compression accuracy
view.point.is_lossless  # True if accuracy == 1.0
```

Slicing is **lazy** for both cluster and analytic-fit (linear/periodic/polynomial)
points: `view[a:b]` reconstructs only the requested indices — no full-array
materialization. `raw` and slice-cached access fall back to the cached path
(which is one-time, then O(1)).

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

Compresses any **numpy array** into Points — model weights, embeddings, features,
time series. (The array is the contract; the domain — weights, behavior-tree
nodes, graph embeddings — is irrelevant to compression.)

```python
from domains.infrastructure.pugqeep import Tree

# Any array data, keyed by name
tree = Tree("game-ai", n_clusters=16)
tree.load_data(
    {
        "patrol_node": patrol_weights,
        "attack_node": attack_weights,
        "flee_node": flee_weights,
    }
)
behavior = tree.get_data("patrol_node")

# Embeddings are arrays too; if they shouldn't be VQ'd, use ModelTree or raw method
tree = Tree("knowledge-graph", n_clusters=8, method="raw")
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

| Operation  | Method                                 | `num_workers`           | Description                    |
| ---------- | -------------------------------------- | ----------------------- | ------------------------------ |
| Compress   | `load_weights(..., num_workers=N)`     | `0`=seq, `-1`=cpu_count | Parallel VQ compression        |
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

| Value    | Description      |
| -------- | ---------------- |
| `URGENT` | Highest priority |
| `HIGH`   | Above normal     |
| `NORMAL` | Default          |
| `LOW`    | Below normal     |

## Engine

Process dispatch with Pools and Stems.

`Engine`, `Pool`, and `TaskQueue` are **synchronous by design** — the parallelism
comes from `ThreadPoolExecutor`/`fork`, never from `await`. Async hosts reach
them across an explicit seam (`asyncio.to_thread` / `Pool.submit`); see
_Execution Philosophy_ in `AGENTS.md`. PGQ itself imports no `asyncio`, so it
runs standalone from scripts, CLI, and tests.

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
engine.dispatch()  # one-shot

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

| Attribute      | Type            | Description                     |
| -------------- | --------------- | ------------------------------- |
| `fn`           | `Callable`      | Function to execute             |
| `args`         | `tuple`         | Positional arguments            |
| `kwargs`       | `dict`          | Keyword arguments               |
| `id`           | `str`           | Unique identifier               |
| `name`         | `str`           | Human-readable name             |
| `status`       | `ProcessStatus` | Current lifecycle state         |
| `result`       | `Any`           | Return value (after completion) |
| `error`        | `Optional[str]` | Error message (if failed)       |
| `parent_id`    | `Optional[str]` | Parent process ID               |
| `children_ids` | `List[str]`     | Child process IDs               |

## Pipe (bounded execution)

`Pipe` is the single admission door for the execution stack — its owned
`ProcessQueue` makes the stack's diameter knowable. Spawning is **fork-only**
(own `os.fork()` wrapper, no `multiprocessing`), and child I/O runs over a
socketpair with a framed wire protocol.

```python
from domains.infrastructure.pugqeep import Pipe, ProcessQueue, PipeClosed

pipe = Pipe(limit=64)          # bounded: capacity hit => block (backpressure)
proc = engine.spawn(fn, arg)   # admission is the FIRST mutation in spawn
pipe.close()                   # Engine.stop() closes it; run()/dispatch() re-open
```

- **Bound** — at capacity `put()` blocks on a `Condition`; no visited-set rejection.
- **`_retire_locked()`** runs on read and only at the ceiling (not per admission).
- **Ownership** — standalone `Pool` mints its own pipe; `Engine.pool()` injects
  Engine's. `Pool.shutdown()` closes only a pipe it minted (`_owns_pipe`).
- **`PipeClosed`** is raised once the pipe is closed — import it from
  `domains.infrastructure.pugqeep`, never `domain.infrastructure._internal`.

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

| Weight type          | Best method           | Typical ratio | Accuracy |
| -------------------- | --------------------- | ------------- | -------- |
| Neural net weights   | `cluster`             | 3-5:1         | 95-99%   |
| Embedding tables     | `raw`                 | 1:1           | 100%     |
| Bias vectors (small) | `raw`                 | 1:1           | 100%     |
| Attention patterns   | `linear`/`polynomial` | 100-1000:1    | 80-95%   |
| Positional encodings | `periodic`            | 1000+:1       | 90-99%   |

## Thread safety

- `PointLibrary` uses `threading.RLock` for all mutations
- `ProducerConsumerQueue` uses `queue.PriorityQueue` (thread-safe)
- `Engine` guards `_processes`, `_pending`, `_pools`, `_dependents` and
  `_completed` with one `threading.RLock`. Every compound operation — looping,
  test-and-append, draining, read-modify-write — holds it or goes through a
  locked snapshot helper (`_procs()`, `_pend()`, `_pool_map()`); scalar probes
  (`len`, `.get`) are single CPython operations and stay lock-free. Walking a
  live container while another thread spawns raises `RuntimeError: dictionary
changed size during iteration`, and `dispatch()` therefore _merges_ into
  `_pending` rather than replacing it — a wholesale replace drops any spawn
  landing mid-classification from the queue for good, leaving a process that is
  registered, never dispatched, and waited on until `wait_for()` times out.
- `Pool` guards `_stems` but **drops its lock across the blocking `admit()`** —
  holding it through a blocking put would serialise every other operation on
  the pool. That splits the capacity check from the registration, so two
  `branch()` calls racing could both pass the check and both register,
  overshooting `max_stems`. `branch()` reserves a slot for that window instead:
  the bound `len(_stems) + _reserved <= max_stems` holds at every instant, and
  the reservation is released on commit or on any failure path. The same
  reservation decides liveness — `_idle_locked()` counts a reserved-but-
  unregistered branch as work in flight, so the pool does not report `IDLE`
  while a `branch()` is still arriving.
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
PYTHONPATH="$PWD/packages/core-py" /home/mana/miniconda3/envs/sloughgpt/bin/python \
  -m pytest packages/core-py/tests/test_pugqeep*.py packages/core-py/tests/test_producer_consumer.py -q

# Specific suites
PYTHONPATH="$PWD/packages/core-py" /home/mana/miniconda3/envs/sloughgpt/bin/python \
  -m pytest packages/core-py/tests/test_pugqeep_point_interface.py -q  # Point protocol + views
PYTHONPATH="$PWD/packages/core-py" /home/mana/miniconda3/envs/sloughgpt/bin/python \
  -m pytest packages/core-py/tests/test_pugqeep_parallel.py -q         # Parallel batch ops
PYTHONPATH="$PWD/packages/core-py" /home/mana/miniconda3/envs/sloughgpt/bin/python \
  -m pytest packages/core-py/tests/test_pugqeep_pipe.py -q             # Bounded execution: Pipe/ProcessQueue
```
