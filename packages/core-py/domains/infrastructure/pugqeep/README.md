# pugqeep

Point-Graph-Queue — composable weight compression, tiered caching, and task queuing for ML systems.

## What It Does

Replaces raw weight storage with **generator functions** (Points). A 1M float32 tensor (4MB) becomes ~64KB of centroids + assignments. Reconstruction is O(1) lookup, not disk I/O.

```
Raw:    4MB tensor → 4MB memory → 4MB disk
Point:  4MB tensor → 64KB centroids → 64KB memory → 64KB disk
```

## Quick Start

### Compress a single array

```python
import numpy as np
from pugqeep import PointCompressor

weights = np.random.randn(1024, 1024).astype(np.float32)
compressor = PointCompressor(n_clusters=32)

point = compressor.compress_cluster(weights, identity="layer0.weight")
print(f"Accuracy: {point.accuracy:.3f}")
print(f"Compression: {weights.nbytes / point.nbytes():.1f}x")

# Reconstruct
reconstructed = point.generate(weights.size)
```

### Compress a full model

```python
from pugqeep import Tree

tree = Tree("my-model", n_clusters=16)
stats = tree.load_weights({
    "layer0.weight": np.random.randn(512, 512).astype(np.float32),
    "layer0.bias": np.random.randn(512).astype(np.float32),
    "layer1.weight": np.random.randn(256, 512).astype(np.float32),
})
print(f"Compression ratio: {stats['ratio']:.1f}x")

# Get a weight back
w = tree.get_weight("layer0.weight")  # numpy array, reconstructed on demand
```

### Tiered caching (Disk → Hot → Memory)

```python
from pugqeep import PGQ, Tier

sys = PGQ(name="my-pipeline")

# Store at different tiers
sys.put("frequent-data", array, tier=Tier.MEMORY)
sys.put("warm-data", array, tier=Tier.HOT)
sys.put("cold-data", array, tier=Tier.DISK)

# Auto-promotes on access
data = sys.get("cold-data")  # promoted to hot
```

### Task queue with priority

```python
from pugqeep import PGQ, Task, TaskPriority

sys = PGQ(name="pipeline")
sys.submit_task(Task(name="preprocess", data=input, priority=TaskPriority.HIGH))
sys.submit_task(Task(name="cleanup", data=input, priority=TaskPriority.LOW))

task = sys.next_task()  # gets "preprocess" first
sys.complete_task(task.id, result=output)
```

### Serialize to bytes (for network/storage)

```python
data = point.to_bytes()           # binary
point = Point.from_bytes(data)    # reconstruct

data = point.to_dict()            # JSON-safe
point = Point.from_dict(data)     # reconstruct
```

### Skip embeddings and biases

```python
from pugqeep import Tree, TreeConfig

config = TreeConfig(
    skip_embeddings=True,   # store as raw (discrete data)
    skip_biases=True,       # store as raw (small tensors)
    n_clusters=32,
)
tree = Tree("model", config=config)
tree.load_weights(weights_dict)
```

### Auto-save library on changes

```python
from pugqeep import PointLibrary, LibraryConfig
from pathlib import Path

config = LibraryConfig(
    name="my-weights",
    storage_dir=Path("./checkpoints"),
    auto_save=True,  # saves after every add/remove
)
lib = PointLibrary(config=config)
lib.add(point)  # auto-saved to ./checkpoints/my-weights.points.json
```

## Architecture

```
PGQ (facade)
  ├── Tree (one per loaded model)
  │     └── PointLibrary (stores Points)
  │           └── Point (generator function, not raw bytes)
  ├── TieredCache (Disk → Hot → Memory)
  └── TaskQueue (priority + retry + persistence)
```

## Point Types

| Type | How It Works | Best For |
|------|-------------|----------|
| `cluster` | Vector quantization (VQ with Lloyd's) | Neural network weights |
| `linear` | `a*i + b` fit | Smoothly varying weights |
| `polynomial` | `a*i² + b*i + c` fit | Curved distributions |
| `periodic` | `a*cos(i) + b*sin(i) + w` | Recurring patterns |
| `raw` | Base64-encoded original | Incompressible data |

## Config

All components accept config objects:

```python
from pugqeep import PointCompressor, CompressorConfig, TreeConfig, LibraryConfig

# Compressor
CompressorConfig(n_clusters=32, lloyd_iterations=10, residual_threshold=0.95)

# Model tree (skip discrete tensors)
TreeConfig(skip_embeddings=True, skip_biases=True, method="cluster")

# Library (auto-save)
LibraryConfig(name="weights", storage_dir=Path("./data"), auto_save=True)
```

## No External Dependencies

Only `numpy` + Python stdlib. No PyTorch, no sentence-transformers, no Pinecone.

## Tests

```bash
PYTHONPATH=packages/core-py python3 -m pytest packages/core-py/tests/test_pugqeep.py -v
```
