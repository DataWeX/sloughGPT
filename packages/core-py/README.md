## Core Python (`domains`)

`packages/core-py/` is on the Python path as the **`domains`** package (and related **`utils`**) when you install from the repo root (**`python3 -m pip install -e ".[dev]"`**).

Training, inference, models, and infrastructure code live under **`domains/`**. The API imports these modules; keep heavy logic here instead of in **`apps/api/server/`** route handlers. Trainer-native **`.soul`** checkpoints embed **`stoi` / `itos` / `chars`** for char-LM eval; see **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*).

### Key infrastructure modules

| Module | Purpose | Docs |
|--------|---------|------|
| `domains.infrastructure.producer_consumer` | General-purpose bounded work queue with priority, backpressure, consumer thread pools | [PRODUCER_CONSUMER_QUEUE.md](../../docs/PRODUCER_CONSUMER_QUEUE.md) |
| `domains.infrastructure.pugqeep` | Point-Graph-Queue system: compressed data points, model trees, task queues, engine | [PUGQEEP.md](../../docs/PUGQEEP.md) |
| `domains.infrastructure.cancel_manager` | Cancellation for long-running operations | [AGENTS.md](../../AGENTS.md) |
| `domains.infrastructure.model_server` | Model lifecycle, backends, circuit breaker | [AGENTS.md](../../AGENTS.md) |
| `domains.infrastructure.process_guard` | Subprocess crash isolation, auto-restart | [AGENTS.md](../../AGENTS.md) |

### Core domains

| Domain | Purpose |
|--------|---------|
| `domains.training` | Training pipelines, executor, distillation |
| `domains.inference` | Vector store, KV cache, providers |
| `domains.feedback` | LoRA eval, per-user adapter |
| `domains.cognitive` | Soul engine, metacognition |
| `domains.infrastructure` | ProcessGuard, ModelServer, TaskQueue, CancelManager |

See **docs/STRUCTURE.md** and **docs/AI_SOFTWARE_ENGINEERING.md**.
