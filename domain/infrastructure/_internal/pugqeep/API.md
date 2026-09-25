# pugqeep API Reference

Point-Graph-Queue system — core infrastructure engine for spawning processes,
branching parallel tasks, and managing data across tiers.

## Architecture

```
Queue (core engine — main process)
  └── Tree (model instance — branches stems into parallel tasks)
        └── Graph/PointLibrary (context — what the tree knows)
              └── Point (star — function-calling capacity)
```

## Quick Start

```python
from pugqeep import PGQ

# Spawn the core engine
pgq = PGQ("infra")
pgq.spawn(load_config, "config.json")
pgq.spawn(start_server, port=8000)
pgq.run()

# Or use data operations
pgq.put("weights", numpy_array)
data = pgq.get("weights")
```

---

## PGQ Facade

Main entry point. Wraps Engine, Tree, Library, Cache, and TaskQueue.

### `PGQ(name, storage_dir, cache_dir, n_clusters, method, memory_max_mb, hot_max_mb)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"model"` | System name |
| `storage_dir` | `Path \| None` | `None` | Directory for persistent storage |
| `cache_dir` | `Path \| None` | `None` | Disk cache directory. `None` = disabled |
| `n_clusters` | `int` | `16` | Number of VQ clusters for compression |
| `method` | `str` | `"cluster"` | Compression method (`"cluster"` or `"function"`) |
| `memory_max_mb` | `int` | `512` | Max memory cache in MB |
| `hot_max_mb` | `int` | `128` | Max hot cache in MB |

### Factory Methods

#### `PGQ.from_model(model_id, n_clusters, method, storage_dir)`
Load a HuggingFace model and compress all weights.

#### `PGQ.from_file(path)`
Load a saved library from disk.

#### `PGQ.queue(model_ids, n_clusters, storage_dir)`
Create a `ModelQueue` with multiple models.

### Data Operations

#### `put(name, data, method, tier, compress) → Point | np.ndarray`
Store data, optionally compressing into a Point.

- `tier`: `"memory"` | `"hot"` | `"disk"`
- `compress`: `True` = compress, `False` = store raw in cache

#### `put_raw(name, data, tier, size_bytes, ttl)`
Store any data (not just numpy) in the cache.

#### `get(name) → np.ndarray | None`
Get data by name, decompressing if needed. Checks cache first, then point library.

#### `get_any(name) → Any | None`
Get any data from cache (not just numpy).

#### `has(name) → bool`
Check if data exists in library or cache.

#### `remove(name) → bool`
Remove data from all tiers.

#### `put_many(data, compress, method, num_workers) → dict`
Store multiple arrays at once. `num_workers`: 0=sequential, -1=cpu_count.

#### `get_many(names, num_workers) → dict`
Get multiple arrays at once.

#### `exists_many(names) → dict`
Check existence of multiple keys.

#### `remove_many(names) → int`
Remove multiple items. Returns count removed.

### Task Queue Operations

#### `submit_task(task) → Task`
Submit a task to the queue.

#### `next_task() → Task | None`
Get next task to process.

#### `complete_task(task_id, result) → Task | None`
Mark task as completed.

#### `fail_task(task_id, error) → Task | None`
Mark task as failed.

#### `cancel_task(task_id) → Task | None`
Cancel a task.

#### `get_task(task_id) → Task | None`
Get task by ID.

#### `list_tasks(status) → list[Task]`
List tasks, optionally filtered by status.

#### `pause_queue()` / `resume_queue()`
Pause/resume the task queue.

### Engine Operations (Process/Tree/Stem)

#### `spawn(fn, *args, name, timeout, priority, **kwargs) → Process`
Spawn a new process on the core engine.

#### `tree(name, max_stems, pool_workers) → Tree`
Create a Tree (model instance) on the core engine.

#### `branch(tree_name, processes) → Stem`
Branch a Stem of parallel processes on a Tree.

#### `run(poll_interval)` / `stop()`
Run/stop the core engine main loop.

#### `run_background(poll_interval) → Thread`
Start engine dispatch loop in a background thread (non-blocking).

#### `wait(timeout)` / `wait_all(timeout)`
Wait for all pending and running processes to complete.

#### `get_completed() → list[Process]`
Return all completed processes since last call.

#### `route(process_name, tree_name)`
Route processes by name to a specific tree.

#### `on_complete(callback)`
Register a callback for when a process completes.

### Training Integration

#### `submit_training(fn, job_id, tree_id, **kwargs) → str`
Submit a training job through the shared TrainingExecutor.

#### `training_status(job_id) → dict | None`
Get training job status.

#### `cancel_training(job_id) → bool`
Cancel a training job.

### Search

#### `search(query) → list[Point]`
Search points by identity.

#### `best(n) → list[Point]`
Get best points by accuracy.

### Persistence

#### `save(path) → Path`
Save library + task queue to disk.

#### `PGQ.load(path) → PGQ`
Load library + task queue from disk.

### Stats

#### `stats() → dict`
System statistics (tree, cache, queue).

#### `cache_stats() → dict`
Cache-only statistics.

#### `queue_stats() → dict`
Queue-only statistics.

#### `export_stats() → dict`
Export full system stats as a serializable dict.

#### `cleanup_cache() → int`
Remove expired cache entries. Returns count removed.

---

## Engine

Core infra engine — the vCPU.

### `Engine(name, max_trees, config)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"main"` | Engine name |
| `max_trees` | `int` | `16` | Maximum number of trees |
| `config` | `EngineConfig \| None` | `None` | Full engine configuration |

### Process Lifecycle

```
CREATED → READY → RUNNING → COMPLETED
                             → FAILED
                             → CANCELLED
```

### `Process(fn, args, kwargs, name, timeout)`

A unit of execution with lifecycle tracking.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Auto-generated 12-char hex ID |
| `name` | `str` | Human-readable name |
| `status` | `ProcessStatus` | Current lifecycle state |
| `result` | `Any` | Return value (on completion) |
| `error` | `str \| None` | Error message (on failure) |
| `timeout` | `float \| None` | Timeout in seconds |
| `depends_on` | `list[str]` | IDs of processes this depends on |

### `ProcessStatus` Enum

- `CREATED` — initial state
- `READY` — queued for execution
- `RUNNING` — currently executing
- `COMPLETED` — finished successfully
- `FAILED` — finished with error
- `CANCELLED` — cancelled by user

### `Stem(tree_id, processes)`

A branch of parallel execution from a Tree.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Auto-generated 12-char hex ID |
| `tree_id` | `str` | Name of the tree this branch is on |
| `processes` | `list[Process]` | Processes in this stem |
| `status` | `StemStatus` | Current state |

### `StemStatus` Enum

- `CREATED` — initial state
- `RUNNING` — processes executing
- `COMPLETED` — all processes completed
- `FAILED` — one or more processes failed

### `Tree(name, max_stems, pool_workers)`

Model instance that branches Stems of parallel tasks.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Tree identifier |
| `status` | `TreeStatus` | Current state |
| `max_stems` | `int` | Max concurrent stems (default 8) |
| `active_stems` | `int` | Current active stems (property) |

### `TreeStatus` Enum

- `IDLE` — no active stems
- `BRANCHING` — stems executing
- `STOPPED` — tree shut down

### Methods

#### `branch(processes) → Stem`
Submit processes to the tree's thread pool.

#### `wait_stem(stem, timeout) → Stem`
Wait for a stem to complete.

#### `store(key, value)` / `recall(key)`
Store/recall context data on the tree's graph.

#### `shutdown()`
Shut down the tree's thread pool.

### `GuardTree(name, config, max_stems, pool_workers)`

Tree that wraps processes in `SubprocessProcess` for OS-level isolation.

### Engine Methods

#### `spawn(fn, *args, name, tree, priority, timeout, depends_on, **kwargs) → Process`
Create and queue a process.

#### `tree(name, max_stems, pool_workers, guarded) → Tree`
Create a new tree.

#### `route(process_name, tree_name)`
Route processes by name to a tree.

#### `branch(tree_name, processes) → Stem`
Branch processes on a specific tree.

#### `dispatch() → int`
Dispatch pending processes to trees (one-shot). Returns count dispatched.

#### `run(poll_interval, on_progress)` / `stop()`
Run/stop the main dispatch loop.

#### `run_background(poll_interval, as_future) → Thread | Future`
Start dispatch loop in background. `as_future=True` returns a `Future`.

#### `wait(timeout)` / `wait_all(timeout)`
Wait for all processes to complete.

#### `get_process(proc_id) → Process | None`
Get a process by ID.

#### `list_processes(status) → list[Process]`
List processes, optionally filtered by status.

#### `cancel_process(proc_id, propagate) → int`
Cancel a process (and optionally its dependents). Returns count cancelled.

#### `cancel_tree(tree_name) → int`
Cancel all processes on a tree.

#### `cancel_all() → int`
Cancel all running/pending processes.

#### `on_complete(callback)`
Register a callback for process completion events.

#### `set_scheduling(policy)`
Set scheduling policy (`ROUND_ROBIN` or `FIRST`).

#### `health() → dict`
Engine health status.

#### `to_dict() → dict`
Full engine state as serializable dict.

#### `start_workers(num_workers, max_queue)` / `stop_workers(timeout)`
Start/stop producer-consumer worker threads.

#### `install_signal_handlers()` / `restore_signal_handlers()`
Manage SIGTERM/SIGINT handlers for graceful shutdown.

---

## TieredCache

Three-tier cache: Disk → Hot → Memory.

### `TieredCache(memory_max_mb, hot_max_mb, disk_dir, promote_threshold, auto_promote, eviction_policy)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_max_mb` | `int` | `512` | Max memory cache size in MB |
| `hot_max_mb` | `int` | `128` | Max hot cache size in MB |
| `disk_dir` | `Path \| None` | `None` | Disk directory. `None` = disabled |
| `promote_threshold` | `int` | `3` | Access count before promoting tier |
| `auto_promote` | `bool` | `True` | Auto-promote frequently accessed data |
| `eviction_policy` | `EvictionPolicy` | `LRU` | Eviction strategy |

### `Tier` Enum

- `DISK` — cold storage (persistent)
- `HOT` — fast local cache
- `MEMORY` — in-memory (fastest, volatile)

### `EvictionPolicy` Enum

- `LRU` — Least Recently Used
- `LFU` — Least Frequently Used

### Methods

#### `get(key) → Any | None`
Get data, promoting to hotter tier if needed. Thread-safe.

#### `peek(key) → Any | None`
Get data without promoting or updating access stats.

#### `put(key, value, tier, size_bytes, pinned, ttl)`
Store data at the specified tier.

- `pinned`: `True` = won't be evicted
- `ttl`: Time-to-live in seconds

#### `remove(key) → bool`
Remove data from all tiers.

#### `exists(key) → bool`
Check if key exists.

#### `list_keys(tier) → list[str]`
List keys, optionally filtered by tier.

#### `evict(target_tier, target_bytes) → int`
Evict entries from a tier to free space. Returns bytes freed.

#### `cleanup_expired() → int`
Remove all expired entries. Returns count removed.

#### `stats() → dict`
Cache statistics including hit rates, tier sizes, eviction counts.

---

## TaskQueue

Priority task queue with persistence, routing, and optional worker pool.

### `TaskQueue(name, storage_dir, max_size)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"default"` | Queue name |
| `storage_dir` | `Path \| None` | `None` | Persistence directory. `None` = in-memory only |
| `max_size` | `int` | `10000` | Maximum tasks in queue |

### `Task(name, data, priority, tree_id, max_retries, metadata)`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Auto-generated 12-char hex ID |
| `name` | `str` | Task type name (for handler lookup) |
| `data` | `Any` | Task payload |
| `status` | `TaskStatus` | Current state |
| `priority` | `TaskPriority` | Priority level |
| `tree_id` | `str \| None` | Assigned tree/instance |
| `result` | `Any` | Return value |
| `error` | `str \| None` | Error message |
| `retries` | `int` | Current retry count |
| `max_retries` | `int` | Max retries (default 3) |
| `metadata` | `dict` | Arbitrary metadata |

### `TaskStatus` Enum

- `PENDING` — waiting to be processed
- `RUNNING` — currently executing
- `COMPLETED` — finished successfully
- `FAILED` — finished with error
- `CANCELLED` — cancelled by user

### `TaskPriority` Enum

- `LOW` (0)
- `NORMAL` (1)
- `HIGH` (2)
- `URGENT` (3)

### Methods

#### `submit(task) → Task`
Submit a task. Auto-dispatches to workers if running.

#### `submit_many(tasks) → list[Task]`
Submit multiple tasks.

#### `submit_batch(items, priority) → list[Task]`
Create and submit multiple tasks from dicts.

#### `next() → Task | None`
Get the next task (highest priority, oldest first).

#### `complete(task_id, result) → Task | None`
Mark task as completed.

#### `fail(task_id, error) → Task | None`
Mark task as failed. Auto-retries if `retries < max_retries`.

#### `cancel(task_id) → Task | None`
Cancel a task.

#### `cancel_many(task_ids) → list[Task | None]`
Cancel multiple tasks.

#### `cancel_all() → int`
Cancel all pending tasks. Returns count cancelled.

#### `retry(task_id, reset_retries) → Task | None`
Retry a failed/completed task.

#### `retry_all(reset_retries) → int`
Retry all failed tasks.

#### `pause()` / `resume()`
Pause/resume the queue.

#### `register_handler(task_name, handler)`
Register a handler for a task type (used by worker pool).

#### `on_complete(callback)`
Register a completion callback.

#### `get_task(task_id) → Task | None`
Get task by ID.

#### `list_tasks(status) → list[Task]`
List tasks, optionally filtered by status.

#### `wait_for(task_id, timeout) → Task | None`
Wait for a specific task to complete.

#### `wait_all(timeout) → list[Task]`
Wait for all tasks to complete.

#### `clear_completed() → int`
Remove completed tasks. Returns count removed.

#### `save(path)` / `TaskQueue.load(path)`
Persist/restore queue to/from disk.

#### `start_workers(num_workers, max_queue)` / `stop_workers(timeout)`
Start/stop worker threads for automatic task execution.

#### `stats() → dict`
Queue statistics.

---

## Generic (Pluggable Architecture)

### `PGQGeneric(compressor, storage, registry)`

Composable facade — wire any strategy + storage + types together.

```python
from pugqeep.generic import PGQGeneric, registry

sys = PGQGeneric(compressor="cluster")
sys.put("weights", array)
```

### ABCs

#### `CompressionStrategy`
Base class for compression strategies.

```python
class MyStrategy(CompressionStrategy):
    name = "custom"

    def compress(self, data, identity, **kwargs) -> Point: ...
    def decompress(self, point, n) -> np.ndarray: ...
```

#### `StorageBackend`
Base class for storage backends.

```python
class S3Storage(StorageBackend):
    name = "s3"

    def load(self, name) -> Point | None: ...
    def save(self, point) -> None: ...
    def remove(self, name) -> bool: ...
    def has(self, name) -> bool: ...
    def list_all(self) -> list[Point]: ...
```

### Built-in Strategies

- `ClusterStrategy` — VQ-based compression (default)
- `FunctionStrategy` — function approximation
- `RawStrategy` — no compression, raw storage
- `AutoStrategy` — picks best strategy per array

### Built-in Storage

- `MemoryStorage` — in-memory dict
- `JSONStorage` — JSON files on disk
- `DirectoryStorage` — directory-based storage

### Registry

```python
from pugqeep.generic import registry

registry.compressors.register(MyStrategy())
registry.storage.register(MyStorage())
```

---

## Config Dataclasses

### `EngineConfig`
Full engine configuration. Contains `SubprocessConfig`, `RestartPolicy`, `MonitorConfig`.

### `SubprocessConfig`
Subprocess isolation settings: memory limits, CPU affinity, cwd, env, capture output.

### `RestartPolicy`
Restart behavior: max_restarts, delay, backoff strategy.

### `MonitorConfig`
Health monitoring: poll interval, stall timeout, restart behavior.

### `PointConfig`
Point creation: function type, clusters, threshold.

### `CompressorConfig`
Compression settings: clusters, Lloyd iterations, gap fill.

### `LibraryConfig`
Library settings: name, storage directory, auto-save.

### `TreeConfig`
Tree settings: name, clusters, method, skip embeddings/biases.

### `QueueConfig`
Queue settings: max trees, default clusters, storage, dedup.

---

## Data Types

### `ProcessStatus` / `StemStatus` / `TreeStatus`
Enum states for engine components.

### `TaskStatus` / `TaskPriority`
Enum states for task queue.

### `Tier` / `EvictionPolicy`
Enum states for cache.

### `EngineMetrics`
Thread-safe metrics tracker: spawned, completed, failed, cancelled, latency, throughput.

### `ResultCache`
LRU + TTL cache for deduplicating identical function calls.

### `ProcessMonitor`
Background thread for stall detection and restart callbacks.
