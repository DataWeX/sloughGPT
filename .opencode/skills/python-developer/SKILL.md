# Python Developer Skill

Adapted to sloughGPT coding style. Use this for all Python development.

## Imports

```python
from __future__ import annotations  # Use in production files

# Standard lib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third party (if needed)
import numpy as np

# Local
from domains.logging import CLILogger
from .base import Logger, LogLevel
```

**Rules:**
- Use `from __future__ import annotations` in production files (not required in tests)
- Relative imports within packages: `from .base import Logger`
- Lazy imports for optional deps (torch, etc.):
  ```python
  try:
      from domains.models import SloughGPTModel
  except ImportError:
      SloughGPTModel = None
  ```
- Use `TYPE_CHECKING` guard for type-only imports:
  ```python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from domains.training.tracking import ExperimentTracker
  ```

## Type Hints

```python
def get_batch(self, split: str = "train") -> tuple:
    """Get a batch of data."""
    ...

def prepare_data(
    data_path: str,
    block_size: int,
    tokenizer: Optional[Any] = None,
) -> tuple:
    ...
```

**Rules:**
- Use `typing` module imports: `Optional[str]`, `Dict[str, Any]`, `List[str]`
- No PEP 604 union syntax (`str | None`) — use `Optional[str]`
- Annotate public method return types: `-> None`, `-> str`, `-> dict[str, Any]`
- Internal helpers can skip type hints

## Error Handling

```python
# Domain logic — raise explicit exceptions
raise ValueError(f"Dataset not found: {name}")
raise ValueError(f"Cannot resume from '{resume_path}': checkpoint is unreadable ({exc})")

# Infrastructure — degrade gracefully
try:
    from domains.infrastructure.output_buffer import install_log_bridge
    install_log_bridge()
except Exception as e:
    logger.debug("OutputBuffer bridge unavailable: %s", e)
    return None
```

**Rules:**
- Domain logic: raise `ValueError` or `KeyError` with descriptive f-string messages
- Infrastructure: catch broad exceptions, log at debug/warning level, continue
- No custom exception classes unless the domain specifically needs them

## Docstrings (Google Style)

```python
def prepare_data(
    data_path: str,
    block_size: int,
    tokenizer: Optional[Any] = None,
) -> tuple:
    """Prepare training data from a text file.

    Converts raw text into integer sequences suitable for training.
    Supports both BPE tokenizers and character-level fallback.

    Args:
        data_path: Path to a UTF-8 text file.
        block_size: Context window length for each training sample.
        tokenizer: Optional SloBPE-compatible tokenizer. If None, falls
            back to character-level encoding.

    Returns:
        (data, vocab_size, stoi, itos) where data is a 1-D numpy int array,
        vocab_size is the vocabulary size, stoi/itos are mapping dicts.
    """
```

**Rules:**
- Module docstrings: describe purpose, usage examples with `Usage::`
- Public functions: `Args:` and `Returns:` blocks
- Private helpers: one-line docstring or none
- Class docstrings: describe purpose, list constructor params in `Parameters:`

## Class Structure

```python
from dataclasses import dataclass, field
from enum import Enum

# Config DTOs — use @dataclass
@dataclass
class TrainerConfig:
    vocab_size: int = 0
    n_embed: int = 128
    n_layer: int = 4
    epochs: int = 10
    learning_rate: float = 3e-4

# Constants — use Enum
class DatasetType(Enum):
    TEXT = "text"
    CODE = "code"
    CONVERSATION = "conversation"

# Performance-critical — use __slots__
class SloughGPTBlock:
    __slots__ = ("ln_1", "attn", "ln_2", "mlp")
    ...

# Everything else — plain classes
class CheckpointManager:
    ...
```

**Rules:**
- `@dataclass` for config/value objects
- `Enum` for type-safe constants
- `__slots__` only in hot paths (neural network, compression)
- Plain classes for everything else

## Naming

```python
# Functions/methods — snake_case
def setup_logging() -> None: ...
def get_request_id() -> str: ...
def _format_human(record: logging.LogRecord) -> str: ...

# Classes — PascalCase
class CLILogger(Logger): ...
class TrainerConfig: ...

# Constants — UPPER_SNAKE_CASE
_NO_COLOR = os.environ.get("NO_COLOR")
_KNOWN_KEYS = {"tag", "op", "request_id"}

# Private — _leading_underscore
_request_id: str = ""
_collect_extras(record)

# Logger names — slo.* namespace
logger = logging.getLogger("slo.trainer")
logger = logging.getLogger("slo.training.datasets")
```

## Logging

```python
import logging

# Module-level logger
logger = logging.getLogger("slo.my_module")

# Structured logging with extra
logger.info("Registered: %s (%s)", config.name, config.dataset_type.value,
    extra={"tag": "TRAIN"})

# Lazy %-style formatting (not f-strings in log calls)
logger.info("Step %d/%d | Loss: %.4f", step, total, loss,
    extra={"tag": "TRAIN"})
```

**Rules:**
- Module-level: `logger = logging.getLogger("slo.<name>")`
- Use `extra={"tag": "TAG"}` for structured fields
- Lazy `%s` formatting, not f-strings in log calls
- Levels: `debug` = diagnostics, `info` = normal flow, `warning` = recoverable, `error` = failure

## Testing (pytest)

```python
import pytest
from unittest.mock import patch

class TestMyFeature:
    """Tests for MyFeature."""

    @pytest.fixture(autouse=True)
    def _clean(self):
        """Reset state before each test."""
        yield
        # cleanup

    def test_basic(self):
        """Test basic functionality."""
        result = my_function("input")
        assert result == "expected"

    @pytest.mark.parametrize("input,expected", [
        ("a", 1),
        ("b", 2),
        ("c", 3),
    ])
    def test_parametrized(self, input, expected):
        assert my_function(input) == expected

    def test_error(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="not found"):
            my_function("invalid")
```

**Rules:**
- Class-based grouping by feature area
- `@pytest.fixture(autouse=True)` for setup/teardown
- `@pytest.mark.parametrize` for multiple test cases
- `pytest.raises` for error testing
- Helper functions for test data creation

## File Organization

```python
"""Module docstring describing purpose."""

from __future__ import imports

# ── Imports ──────────────────────────────────────────────────────────

import ...

# ── Constants ────────────────────────────────────────────────────────

_KNOWN_KEYS = {...}

# ── Classes ──────────────────────────────────────────────────────────

class MyClass:
    ...

# ── Functions ────────────────────────────────────────────────────────

def my_function() -> None:
    ...
```

**Rules:**
- Module docstring at top
- `# ── Section ──────────` dividers in large files
- `__all__` in `__init__.py` for public API
- Lazy `__getattr__` for optional dependencies in `__init__.py`
- Domain-driven directory structure: `domains/<domain>/`

## __init__.py Pattern

```python
"""Package docstring."""

from __future__ import annotations

# Eager imports (always available)
from .base import Logger, LogLevel
from .config import LogFormatter

# Lazy imports (optional dependencies)
LAZY_IMPORTS = {
    "TrainingUX": ".training_ux",
}

__all__ = ["Logger", "LogLevel", "LogFormatter", "TrainingUX"]

def __getattr__(name):
    if name in LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(LAZY_IMPORTS[name], package=__name__)
        obj = getattr(module, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```
