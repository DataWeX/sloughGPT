# Python-First Architecture

Source of truth for HOW we write code. Read before implementing anything.

---

## The Two Rules

### 1. Minimal API per feature

Every feature exposes the smallest possible API. Users shouldn't need to read docs to use it.

```python
# GOOD: 3 functions, clear purpose
from sloughgpt.training import train, load, status
from sloughgpt.inference import generate, chat
from sloughgpt.knowledge import ingest, search

# Use it
model = train("shakespeare.txt")
response = generate(model, "Write a sonnet")
results = search("what did I write about?")
```

```python
# BAD: 20 functions, confusing
from sloughgpt.training import (
    train, load, status, get_config, set_config,
    validate_data, preprocess, postprocess,
    export_model, import_model, compare_models,
    get_metrics, plot_loss, save_checkpoint,
    load_checkpoint, list_checkpoints, delete_checkpoint,
    ...
)
```

### 2. Standalone features

Each feature works like a pip package. You import it, call it, get a result. No server, no setup, no boilerplate.

```python
# This should just work
from sloughgpt.training import train

model = train("data.txt")

# This too
from sloughgpt.inference import generate

response = generate(model, "Hello")

# And this
from sloughgpt.knowledge import ingest, search

ingest(model, "notes.pdf")
results = search("what are my notes about?")
```

---

## The Problem

We keep writing code when the API already exists:

```python
# WRONG: writing training logic when train() exists
from domain.training._internal.slonet import SloTransformer
from domain.training._internal.lora import LoRAWrapper

model = SloTransformer(config)
lora = LoRAWrapper(model)
lora.train(dataset)
# Just do: model = train("data.txt")

# WRONG: writing inference logic when generate() exists
from domain.inference._internal.slonet_provider import SloNetChatProvider

provider = SloNetChatProvider(model_path)
tokens = provider.generate(prompt, max_tokens=100)
# Just do: response = generate(model, "Hello")

# WRONG: writing chat logic when chat() exists
from domain.chat._internal.domain import ChatDomain

chat = ChatDomain(engine=provider)
response = chat.respond(messages=[...])
# Just do: response = chat(model, "Hello")
```

---

## How to Build

### 1. Each feature = one file with minimal API

```
domain/
  training/
    __init__.py    # export: train, load, status
    _train.py      # implementation (private)
  inference/
    __init__.py    # export: generate, chat
    _generate.py   # implementation (private)
  knowledge/
    __init__.py    # export: ingest, search
    _ingest.py     # implementation (private)
```

### 2. Public API in `__init__.py`

```python
# domain/training/__init__.py
from domain.training._train import train, load, status

__all__ = ["train", "load", "status"]
```

That's it. Three functions. Users import from here, never from `_internal/`.

### 3. Implementation is private

```python
# domain/training/_train.py
from __future__ import annotations


def train(data: str, epochs: int = 3) -> str:
    """Train a model. Returns model path."""
    # All the复杂 logic goes here
    # Users never see this
    return model_path


def load(path: str) -> Model:
    """Load a trained model."""
    ...


def status() -> dict:
    """Get training status."""
    ...
```

### 4. Web router is just plumbing

```python
# apps/api/server/routers/training.py
from domain.training import train, status


@router.post("/train")
async def train_endpoint(request: TrainRequest):
    return train(data=request.data, epochs=request.epochs)


@router.get("/status")
async def status_endpoint():
    return status()
```

---

## What Changes

| Current                      | New                                   |
| ---------------------------- | ------------------------------------- |
| 20 functions per feature     | 3 functions per feature               |
| Import from `_internal/`     | Import from `__init__.py`             |
| Need to understand internals | Just call the function                |
| Write training code          | `model = train("data.txt")`           |
| Write inference code         | `response = generate(model, "Hello")` |
| Write chat code              | `reply = chat(model, "Hi")`           |

---

## Checklist

For each feature, ask:

- [ ] Can a user do everything with just the exported functions?
- [ ] Are there ≤5 exported functions?
- [ ] Does it work without a running server?
- [ ] Can I use it without reading the implementation?

If any answer is no, simplify.

---

_This doc overrides any conflicting patterns. When in doubt, fewer APIs, simpler usage._
