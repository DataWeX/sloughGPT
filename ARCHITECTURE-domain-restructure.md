# Domain Package Restructure

## Problem

The `packages/core-py/domains/` directory is a dumping ground — 30+ unrelated subsystems crammed into one namespace. There's no meaningful boundary between `billing`, `consciousness`, `shell`, and `multimodal`. Everything imports everything, and the namespace `domains.` is meaningless.

## Solution

Extract each domain into a standalone package with:

1. **Clean public API** via `__init__.py`
2. **`_internal/` directory** for implementation details
3. **No cross-package imports** — use protocols/events for communication
4. **Independently testable** — `cd packages/memory && pytest`

## Target Structure

```
packages/
  memory/                    # Reference implementation
    __init__.py              # Public API: MemoryConfig, get_memory_service, ...
    _internal/
      __init__.py
      config.py              # MemoryConfig singleton
      embedding.py           # Self-contained n-gram embedding
      provider.py            # MemoryProvider protocol + KnowledgeMemoryProvider
      service.py             # MemoryService facade
      consolidation.py       # plan_consolidation (pure function)
      task_memory.py         # Task queue handlers + archive ops
      maintenance.py         # Background scheduler

  slo-net/                   # Core ML engine (next)
    __init__.py
    _internal/
      model.py
      inference.py
      training.py
      export.py

  voice/                     # TTS/STT (after slo-net)
    __init__.py
    _internal/
      tts.py
      stt.py
      phoneme.py
```

## Pattern

### Public API (`__init__.py`)

```python
"""Memory layer - chat- and task-agnostic auto-memory facade."""

from memory._internal.config import MemoryConfig
from memory._internal.service import MemoryService, get_memory_service
# ... re-export all public symbols

__all__ = ["MemoryConfig", "MemoryService", "get_memory_service", ...]
```

### Internal Implementation (`_internal/`)

```python
# _internal/config.py
from __future__ import annotations
import os

class MemoryConfig:
    """Runtime configuration for the auto-memory layer."""
    # ... implementation
```

### Cross-Domain Dependencies

**Rule**: No top-level imports from other packages. Use lazy imports or dependency injection.

```python
# GOOD: Lazy import inside function body
def _get_store(self):
    from domains.learner.knowledge import get_knowledge_memory
    return get_knowledge_memory()

# GOOD: Protocol for storage seam
class MemoryProvider(Protocol):
    def store(self, content: str, topic: str, source: str) -> bool: ...

# BAD: Top-level cross-package import
from domains.learner.knowledge import KnowledgeFact  # DON'T
```

### Embedding Utilities

Self-contained math functions live in `_internal/embedding.py`:

```python
from memory._internal.embedding import cosine_similarity, ngram_embed
```

No dependency on `domains.inference.vector_store` — the functions are copied because they're pure math with no business logic.

## Migration Strategy

1. **Create new package** with `_internal/` structure
2. **Add compatibility shim** in old `domains/memory/__init__.py` that re-exports from new package
3. **Update consumers** gradually — old imports still work via shim
4. **Remove shim** once all consumers are migrated

## Checklist for Next Domain

- [ ] Pick domain (slo-net, voice, vm, etc.)
- [ ] Analyze dependencies and public API surface
- [ ] Create `packages/<name>/` with `_internal/` structure
- [ ] Copy self-contained utilities (embedding, math, etc.)
- [ ] Use lazy imports for cross-domain deps
- [ ] Add compatibility shim in old location
- [ ] Update `pyproject.toml` package discovery
- [ ] Test import paths work
