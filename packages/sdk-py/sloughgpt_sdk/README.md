# SloughGPT Python SDK

A Python client library for the SloughGPT API. Every method maps to a real backend endpoint — no simulated or local-only API surface.

## Features

| Category | Features |
|----------|----------|
| **Core** | Sync/Async HTTP client, streaming, batch processing |
| **Server** | Health, models, chat, generate, tokenizer, system, souls, knowledge |
| **Training** | Fine-tune jobs, auto-train, checkpoints, recovery |
| **Tools** | Caching, benchmarks, profiling, live model registry |
| **CLI** | Full command-line interface |

## Installation

```bash
python3 -m pip install sloughgpt-sdk
```

For all features:
```bash
python3 -m pip install "sloughgpt-sdk[all]"
```

## Development (monorepo)

When developing inside the **sloughGPT** repository, install from the **repo root** (`python3 -m pip install -e ".[dev]"`) and run **`python3 -m pytest tests/test_sdk.py`** (same checks as CI job **`sdk-test-py`** in **`.github/workflows/ci_cd.yml`**).

## Quick Start

```python
from sloughgpt_sdk import SloughGPTClient, ChatMessage

client = SloughGPTClient(base_url="http://localhost:8000")

# Text generation
result = client.generate("Hello!")
print(result.generated_text)

# Chat
result = client.chat([ChatMessage.user("Hi!")])
print(result.message.content)
```

## Training jobs

`POST /training/start` expects a `TrainingRequest` JSON body. Pass **`log_interval`** and **`eval_interval`** as keyword arguments so live metrics on **`GET /training/jobs`** refresh at the cadence you want (defaults match the web Console: 10 / 100). Trainer **`.soul`** files on the server include **`stoi` / `itos` / `chars`** so char-LM eval decodes cleanly; formats are summarized in [`docs/policies/CONTRIBUTING.md`](../../../docs/policies/CONTRIBUTING.md) (*Checkpoint vocabulary*). **`get_training_status`** / **`list_training_jobs`** may return a **`checkpoint`** path with the same native semantics.

```python
job = client.start_training(
    "slough-base",
    "shakespeare",
    epochs=2,
    batch_size=8,
    learning_rate=1e-3,
    log_interval=10,
    eval_interval=100,
)
job_id = job.get("id") or job.get("job_id")
status = client.get_training_status(job_id)
all_jobs = client.list_training_jobs()
```

## Simple Tracking

```python
# Track metrics during training
with client.track("training-v1") as t:
    for epoch in range(10):
        acc = train()
        t.log("accuracy", acc)
        t.next_step()
```

## Model Registry (live server)

The registry methods proxy the real `ModelRegistry` running inside the server — no client-side state.

```python
models = client.list_registry_models()
stats = client.get_registry_stats()
best = client.get_registry_best()
detail = client.get_registry_model(models[0]["model_id"])
```

## Security & Rate Limits

```python
keys = client.get_security_keys()
audit = client.get_audit_log()
status = client.get_rate_limit_status()
```

## Caching

```python
from sloughgpt_sdk import InMemoryCache

cache = InMemoryCache(ttl=3600)  # 1 hour
cache.set("key", "value")
value = cache.get("key")
```

## Benchmarks

```python
from sloughgpt_sdk import Benchmark

bench = Benchmark()
result = bench.run("My operation", lambda: do_work(), iterations=1000)
print(f"{result.ops_per_second} ops/sec")
```

## CLI Tool

```bash
# Generate text
sloughgpt-cli generate "Hello world"

# Chat
sloughgpt-cli chat "What is Python?"

# Registry (live server)
sloughgpt-cli registry list
sloughgpt-cli registry info gpt2
sloughgpt-cli registry best
sloughgpt-cli registry stats

# Metrics
sloughgpt-cli metrics
```

## All SDK Modules

```python
from sloughgpt_sdk import (
    # Core
    SloughGPTClient,
    AsyncSloughGPTClient,

    # Models
    GenerateRequest, GenerationResult,
    ChatMessage, ChatRequest, ChatResult,
    BatchRequest, BatchResult,
    ModelInfo, DatasetInfo,
    HealthStatus, SystemInfo, MetricsData,
)
```

## Error Handling

```python
from sloughgpt_sdk import SloughGPTClient
from sloughgpt_sdk.exceptions import APIError, RateLimitError

try:
    result = client.generate("Hello")
except RateLimitError:
    print("Rate limited")
except APIError as e:
    print(f"API error: {e.message}")
```

## License

MIT License
