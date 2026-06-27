# Foundations: Pre-LLM Infrastructure

Six layers that should exist under the LLM architecture, not tangled in it.
Each builds on the previous.

## Tracker

| Layer | Issue | Status | Built |
|-------|-------|--------|-------|
| Task Queue | — | ✅ | `domains/infrastructure/task_queue.py` |
| Event Bus | [#36](https://github.com/DataWeX/sloughGPT/issues/36) | ✅ | `domains/infrastructure/event_bus.py` |
| Config System | [#31](https://github.com/DataWeX/sloughGPT/issues/31) | ❌ | — |
| Error Taxonomy | [#35](https://github.com/DataWeX/sloughGPT/issues/35) | ❌ | — |
| Rate Limiter | [#32](https://github.com/DataWeX/sloughGPT/issues/32) | ❌ | — |
| Data Repository | [#34](https://github.com/DataWeX/sloughGPT/issues/34) | ❌ | — |
| Lifecycle Manager | [#33](https://github.com/DataWeX/sloughGPT/issues/33) | ❌ | — |

## Why These Exist

- **Task Queue**: Async priority queue with worker pool, pause/resume/cancel, retries, dependency tracking, SSE events. Already built — the foundation for everything below.
- **Event Bus**: Decouple components that currently
  use direct function calls. `emit()` / `on()` / `once()`.
  Task Queue SSE callbacks wire into it.
- **Config System**: Replace `os.getenv` with typed schemas,
  file-based defaults, env override, runtime reload.
- **Error Taxonomy**: Recoverable vs fatal, error codes,
  retry policies. Replace `except Exception` everywhere.
- **Rate Limiter**: Per-endpoint, per-user, per-model concurrency.
  Token bucket + 429 backpressure.
- **Data Repository**: Abstracted CRUD with caching, migration,
  consistent interface across sessions/datasets/knowledge.
- **Lifecycle Manager**: Ordered startup/shutdown, health gates,
  graceful drain. Registered services with dependency graph.
