# sloughGPT Refactoring Prompt

Copy this into a new opencode tab session.

---

## Prompt

```
Read docs/PYTHON_FIRST.md.

RULES:
1. Minimal API per feature — max 5 exported functions per domain module
2. Each feature stands alone — import, call, result. No server, no internals.

YOUR TASK:

PHASE 1: Audit
- List every domain module under domain/
- For each, count exported functions in __init__.py
- Flag any with >5 exports — these need trimming
- Flag any that require _internal/ imports to use
- Output: audit.md

PHASE 2: Simplify APIs
- For each domain module, reduce to ≤5 essential functions
- Example training: train(), load(), status()
- Example inference: generate(), chat()
- Example knowledge: ingest(), search()
- Move implementation to _private.py files
- Export only the minimal API from __init__.py

PHASE 3: Fix imports
- Every router must import from domain, not _internal/
- Every CLI command must import from domain, not _internal/
- Every test must import from domain, not _internal/
- Find all _internal/ imports and replace them

PHASE 4: Verify
- Run pytest — all tests must pass
- For each changed module, verify: import → call → result works
- Example: from sloughgpt.training import train; train("test.txt")

OUTPUT: Show before/after for each module changed.
```

---

## What This Fixes

```python
# BEFORE: users write implementation code
from domain.training._internal.slonet import SloTransformer
from domain.training._internal.lora import LoRAWrapper
model = SloTransformer(config)
lora = LoRAWrapper(model)
lora.train(dataset)

# AFTER: users just call the API
from sloughgpt.training import train
model = train("data.txt")
```

```python
# BEFORE: 20 exported functions
from sloughgpt.training import (
    train, load, status, get_config, set_config,
    validate_data, preprocess, postprocess, export_model,
    import_model, compare_models, get_metrics, plot_loss,
    save_checkpoint, load_checkpoint, list_checkpoints,
    delete_checkpoint, resume_training, cancel_training,
    get_logs
)

# AFTER: 3 exported functions
from sloughgpt.training import train, load, status
```

```python
# BEFORE: server required
import requests
requests.post("http://localhost:8000/api/train", json={"data": "file.txt"})

# AFTER: just Python
from sloughgpt.training import train
model = train("file.txt")
```
