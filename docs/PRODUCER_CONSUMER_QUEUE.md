# ProducerConsumerQueue

General-purpose bounded work queue with priority scheduling, backpressure, and consumer thread pools.

**Location:** `packages/core-py/domains/infrastructure/producer_consumer.py`

## When to use

- Background work that must not block the request path (GC, cache eviction, metrics flushing)
- Parallel batch operations (compression, decompression, file I/O)
- Anywhere you would otherwise spawn raw `threading.Thread` — use a queue instead

Do not use for CPU-bound work that saturates cores — use `ProcessPoolExecutor` or the pugqeep Engine instead.

## Quick start

```python
from domains.infrastructure.producer_consumer import ProducerConsumerQueue, ShutdownMode

# Create a queue with4 consumer threads
q = ProducerConsumerQueue(
    maxsize=64,
    num_consumers=4,
    handler=lambda item: item() if callable(item) else None,
    name="my-bg",
)

q.start()

# Submit work (thread-safe, blocks if queue full)
q.put(my_task)
q.put_nowait(my_task)  # non-blocking, returns False if full

# Priority work (lower number = higher priority)
q.put(urgent_task, priority=0)
q.put(normal_task, priority=1)

# Shutdown (drain remaining work, then stop)
q.stop(timeout=10.0)

# Or drop remaining work immediately (set at construction)
q_drop = ProducerConsumerQueue(
    maxsize=64, num_consumers=2,
    handler=process,
    shutdown_mode=ShutdownMode.DROP,
    name="drop-on-exit",
)
q_drop.start()
# ... work ...
q_drop.stop(timeout=2.0)  # drops remaining immediately
```

## API

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `maxsize` | `int` | `0` | Max queue capacity. `0` = unbounded. |
| `num_consumers` | `int` | `1` | Number of consumer threads. |
| `handler` | `Callable[[T], Any]` | `None` | Called for each item. Must be thread-safe. |
| `priority` | `bool` | `False` | Enable priority scheduling. |
| `shutdown_mode` | `ShutdownMode` | `DRAIN` | `DRAIN` (finish work) or `DROP` (discard). |
| `name` | `str` | `"pcq"` | Name for logging. |

### Sync API

| Method | Signature | Description |
|--------|-----------|-------------|
| `put` | `(item, timeout=None, priority=0) -> bool` | Enqueue. Returns `False` if full/stopped. |
| `put_nowait` | `(item, priority=0) -> bool` | Non-blocking put. |
| `get` | `(timeout=None) -> (bool, T \| None)` | Dequeue. Returns `(False, None)` on timeout. |
| `task_done` | `() -> None` | Mark task done (joinable queue semantics). |

### Async API

| Method | Signature | Description |
|--------|-----------|-------------|
| `async_put` | `(item, timeout=None, priority=0) -> bool` | Async put (bridges to sync via `to_thread`). |
| `async_get` | `(timeout=None) -> (bool, T \| None)` | Async get (bridges to sync via `to_thread`). |

### Lifecycle

| Method | Signature | Description |
|--------|-----------|-------------|
| `start` | `() -> None` | Start consumer threads. |
| `stop` | `(timeout=5.0) -> None` | Stop consumers. DRAIN or DROP per `shutdown_mode`. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `qsize` | `int` | Current queue depth. |
| `empty` | `bool` | Queue is empty. |
| `full` | `bool` | Queue is at capacity. |
| `active_consumers` | `int` | Number of consumers currently processing. |
| `is_running` | `bool` | Queue is started and not stopped. |
| `metrics` | `dict` | `{enqueued, consumed, dropped, errors, queued, active_consumers}`. |

## Priority mode

When `priority=True`, items are wrapped as `(priority, sequence, item)`. Lower priority numbers are dequeued first. Sequence number ensures FIFO within the same priority.

```python
q = ProducerConsumerQueue(priority=True, num_consumers=2, handler=process)
q.start()

q.put(critical_work, priority=0)   # processed first
q.put(normal_work, priority=1)     # processed second
q.put(background_work, priority=2) # processed last
```

## Global queue

```python
from domains.infrastructure.producer_consumer import get_producer_consumer_queue

q = get_producer_consumer_queue()  # lazy singleton, maxsize=256, 4 consumers
```

## Current consumers in sloughGPT

| Consumer | Location | Purpose | Maxsize | Consumers |
|----------|----------|---------|---------|-----------|
| `model-server-bg` | `model_server.py:802` | GC, background work | 32 | 2 |
| TaskQueue workers | `pugqeep/task_queue.py:391` | Priority task execution | configurable | configurable |
| Engine workers | `pugqeep/engine.py:622` | Process spawn dispatch | configurable | configurable |
| ModelTree parallel | `pugqeep/model_tree.py:123` | Weight compression | `cpu_count` | `cpu_count` |
| Decompress parallel | `pugqeep/model_tree.py:387` | Weight decompression | `cpu_count` | `cpu_count` |
| Facade `put_many` | `pugqeep/facade.py:560` | Batch data writes | `cpu_count` | `cpu_count` |
| Facade `get_many` | `pugqeep/facade.py:612` | Batch data reads | `cpu_count` | `cpu_count` |

## Shutdown behavior

| Mode | Behavior |
|------|----------|
| `DRAIN` | Wait for queue to empty (up to `timeout`), then send poison pills and join threads. |
| `DROP` | Send poison pills immediately, join threads. Remaining items are dropped. |

Poison pills are `None` items (or `_PriorityItem(priority=999999, item=None)` in priority mode) that break consumers out of their `get()` loop.

## Design decisions

- **`queue.PriorityQueue` for priority mode** — thread-safe, no custom locking needed.
- **`queue.Queue` for non-priority mode** — simpler, lower overhead.
- **Daemon threads** — consumers do not prevent process exit.
- **Metrics under lock** — `_metrics_lock` protects counters from race conditions.
- **Async bridge via `run_in_executor`** — avoids duplicating queue logic for async contexts.
