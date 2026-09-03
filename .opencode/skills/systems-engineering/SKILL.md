---
name: systems-engineering
description: >
  Use when building OS components, kernel modules, device drivers, init system
  services, Buildroot configurations, x86 VM extensions, or shell TUI infrastructure
  in sloughGPT. Also covers all Python development conventions. Covers the Dait kernel,
  x86 VM, v86 browser Linux, custom Buildroot image builds, and Python coding standards.
---

# Systems Engineering + Python Skill

## Overview

This skill covers two areas:
1. **Python conventions** — imports, types, error handling, docstrings, naming, logging, testing, file organization
2. **Systems architecture** — kernel, VM, devices, init system, shell TUI, Buildroot

## Python Conventions

### Imports

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

### Type Hints

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

### Error Handling

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

### Docstrings (Google Style)

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

### Class Structure

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

### Naming

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

### Logging

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

### Testing (pytest)

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

### File Organization

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

### `__init__.py` Pattern

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

---

## Systems Architecture

### When to Use

- Writing or modifying kernel code (scheduler, memory, syscalls, addons)
- Writing or modifying device drivers (DeviceBus, DeviceDriver, fd I/O)
- Writing or modifying init system services (runlevels, service lifecycle)
- Extending the x86 VM (new instructions, syscalls, RBAC roles)
- Building or configuring Buildroot images
- Working on the shell TUI (pane engine, surfaces, borders, cursor lifecycle)
- Working on VFS (mount points, /dev/*, /proc/*, host fs bridging)

### Architecture Reference

```
┌─────────────────────────────────────────────────┐
│                User Applications                │
│  (Shell TUI, VM Console, Web Dashboard, CLI)    │
├─────────────────────────────────────────────────┤
│                Init System                      │
│  runlevels: 0=halt 1=single 2=multi 3=full      │
│  services: kernel, agent-orchestrator,          │
│            knowledge-worker, api-server         │
│  features: dependency ordering, respawn,        │
│            health checks, graceful shutdown     │
├────────────────────┬────────────────────────────┤
│    Dait Kernel     │        x86 VM             │
│  ┌──────────────┐  │  ┌────────────────────┐   │
│  │ Scheduler    │  │  │ X86CPU             │   │
│  │  (priority)  │  │  │  16 registers      │   │
│  │ TensorMemory │  │  │  64KB memory       │   │
│  │  (numpy)     │  │  │  INT 0x80 syscalls │   │
│  │ DeviceMgr    │  │  │ RBAC               │   │
│  │  (fd-based)  │  │  │  USER/ADMIN/KERNEL │   │
│  │ SyscallTable │  │  │ VGA @ 0xB8000      │   │
│  │  (INT 0x80)  │  │  │ Training Bridge    │   │
│  │ AddonLoader  │  │  └────────────────────┘   │
│  └──────────────┘  │                           │
├────────────────────┴────────────────────────────┤
│              Device System                      │
│  DeviceBus → DeviceDriver → fd I/O              │
│  /dev/llm, /dev/embedding, /dev/training        │
├─────────────────────────────────────────────────┤
│              VFS Layer                          │
│  MountTable: /dev/* (devices),                  │
│  /proc/* (kernel info), host fs (real paths)    │
└─────────────────────────────────────────────────┘
```

### Key Files

| Component | File | Key Classes |
|-----------|------|-------------|
| Kernel | `packages/core-py/domains/shell/kernel.py` | `Kernel`, `Scheduler`, `TensorMemory`, `SyscallTable` |
| Init | `packages/core-py/domains/shell/init.py` | `InitSystem`, `ServiceManager`, `ServiceDefinition` |
| Devices | `packages/core-py/domains/shell/devices.py` | `DeviceSystem`, `DeviceBus` |
| Device Drivers | `packages/core-py/domains/shell/device_system.py` | `DeviceDriver`, fd-based I/O |
| VFS | `packages/core-py/domains/shell/addons/filesystem.py` | `VFSAddon`, `MountTable` |
| x86 VM | `packages/core-py/domains/shell/vm.py` | `X86CPU`, `X86Assembler`, `ProcessTable`, `Scheduler` |
| VM Engine | `packages/core-py/domains/shell/vm_engine.py` | `VMEngine`, `Breakpoint`, `StepEvent` |
| VM Syscalls | `packages/core-py/domains/shell/vm.py` | `SYSCALL_TABLE` (INT 0x80 dispatch) |
| VM RBAC | `packages/core-py/domains/shell/vm_permissions.py` | `Role`, `Permission`, `X86RBAC` |
| VM Training | `packages/core-py/domains/shell/vm_training_bridge.py` | `TrainingBridge` |
| VM Programs | `packages/core-py/domains/shell/vm_programs.py` | Built-in assembly programs |
| Addons | `packages/core-py/domains/shell/addons/` | `neural.py`, `filesystem.py`, `shell_ui.py` |
| Runtime | `packages/core-py/domains/shell/runtime.py` | `DaitRuntime` (boot/shutdown orchestration) |
| Shell TUI | `packages/core-py/domains/shell/tui_repl.py` | `TuiRepl`, `_draw_borders`, `_render_*` |
| Pane Engine | `packages/core-py/domains/shell/pane.py` | `Rect`, `Border`, `Pane`, `PaneLayout` |
| Surfaces | `packages/core-py/domains/shell/surface.py` | `TextSurface`, `LogSurface`, `clip`, `_display_width` |
| Console | `packages/core-py/domains/shell/console.py` | `Console`, `_TuiSpinner` |
| Shell IO | `packages/core-py/domains/shell/io.py` | `ShellIO`, `ConsoleIO`, `MemoryIO` |
| v86 Controller | `apps/web/lib/v86-controller.ts` | `V86Controller` |
| v86 Hook | `apps/web/hooks/useV86.ts` | `useV86()` |
| VM API | `apps/api/server/routers/vm.py` | `/vm/run`, `/vm/builtins`, `/vm/info` |

### Kernel Development

#### Syscall Table (INT 0x80)

The x86 VM dispatches syscalls via `INT 0x80` with `eax` = syscall number:

| Number | Name | Args | Returns | RBAC |
|--------|------|------|---------|------|
| 1 | SYS_EXIT | code | — | USER |
| 2 | SYS_PRINT | addr, len | — | USER |
| 3 | SYS_SCAN | addr, max | bytes_read | USER |
| 4 | SYS_OPEN | path_addr, mode | fd | USER |
| 5 | SYS_READ | fd, buf_addr, len | bytes_read | USER |
| 6 | SYS_WRITE | fd, buf_addr, len | bytes_written | USER |
| 7 | SYS_CLOSE | fd | 0 | USER |
| 8 | SYS_DEV_OPEN | dev_id | fd | ADMIN |
| 9 | SYS_DEV_CALL | fd, cmd, arg_addr | result | ADMIN |
| 10 | SYS_DEV_CLOSE | fd | 0 | ADMIN |
| 28 | SYS_TRAIN_START | config_addr | job_id | ADMIN |
| 29 | SYS_TRAIN_STATUS | job_id | status | ADMIN |
| 30 | SYS_TRAIN_GET_RESULT | job_id | result_addr | ADMIN |

#### Adding a New Syscall

1. Define the syscall number in `vm.py` (add to `SYSCALL_TABLE`)
2. Implement the handler function in `vm.py`
3. Add RBAC permission mapping in `vm_permissions.py`
4. Add the assembler mnemonic if user-facing
5. Write tests in `tests/test_vm*.py`
6. Update this skill doc

#### Adding a New Kernel Addon

1. Create `packages/core-py/domains/shell/addons/<name>.py`
2. Implement the `Addon` protocol from `base.py`
3. Register via `kernel.install_addon(<name>)` in `runtime.py`
4. Add test in `tests/test_shell_runtime.py`

#### Dynamic Module Loading

```python
from domains.shell.addons.module_loader import ModuleLoader

loader = ModuleLoader(addon_dirs=["path/to/addons"])
loader.set_kernel(kernel)

available = loader.discover()
addon = loader.load("my_addon")
addon = loader.reload("my_addon")
loader.unload("my_addon")
print(loader.summary())
```

#### VM Debugger

```python
from domains.shell.vm_debugger import Debugger

debugger = Debugger()
debugger.set_output(print)

engine = debugger.engine
engine.load_source(source)
debugger.load_symbols(source)

debugger.bp_set("main")
debugger.bp_set(0x1000, "loop_start")

debugger.stepi()
debugger.step_over()
debugger.step_out()

debugger.dump_regs()
debugger.dump_flags()
debugger.dump_memory(0x1000, 64)
debugger.dump_stack(8)

trace = debugger.continue_exec()
analysis = debugger.analyze_trace(trace)
```

#### Adding a New Device Driver

1. Subclass `DeviceDriver` in `device_system.py`
2. Implement `open()`, `call()`, `close()` methods
3. Register with `DeviceSystem` in `devices.py`
4. The device appears at `/dev/<name>` in VFS

### Buildroot Build System

#### Directory Structure

```
buildroot/
├── configs/
│   └── sloughgpt_defconfig       # Buildroot defconfig
├── overlays/
│   ├── etc/
│   │   ├── init.d/               # Init scripts (S00symlink, S01mount, ...)
│   │   ├── fstab                 # Mount table
│   │   └── profile               # Shell profile
│   ├── usr/bin/                  # Custom binaries
│   └── root/                     # Root home directory
├── packages/
│   └── sloughgpt/
│       ├── sloughgpt.mk          # Package makefile
│       └── Config.in             # Kconfig entry
├── post-build.sh                 # Rootfs customization hook
├── post-image.sh                 # Image creation hook
└── README.md                     # Build instructions
```

#### Build Commands

```bash
# Setup Buildroot (first time)
make -C buildroot sloughgpt_defconfig

# Build full image
make -C buildroot

# Build only custom package
make -C buildroot sloughgpt-rebuild

# Clean and rebuild
make -C buildroot clean && make -C buildroot
```

#### v86 Integration

The v86 browser VM loads a raw disk image. After Buildroot builds:

1. The image is at `buildroot/output/images/rootfs.ext2`
2. Convert for v86: `dd if=buildroot/output/images/rootfs.ext2 of=buildroot/output/images/buildroot.img bs=512`
3. Serve from `apps/web/public/buildroot/` or a CDN
4. Update `LINUX_IMAGE_URL` in `apps/web/hooks/useV86.ts`

### Shell TUI Architecture

#### Pane Layout

```python
layout = PaneLayout([
    Pane("console", ratio=0.3, min_rows=3, border=Border("all")),
    Pane("output", ratio=0.7, min_rows=3, border=Border("all")),
    Pane("status", fixed=1),
    Pane("input", fixed=1),
])
regions = layout.compute(term_rows, term_cols)
```

#### Rendering Pipeline

```
TuiRepl._render_all()
  ├─ _draw_borders(stdscr, regions)     # Border characters
  ├─ _blit(win_console, log_lines, oy, ox)  # Log surface content
  ├─ _blit(win_output, output_lines, oy, ox)  # Output surface content
  ├─ _render_status(win_status)          # Status bar
  └─ _render_input(win_input)            # Input line + cursor
```

#### Cursor Lifecycle

All ANSI cursor operations must be guarded:

```python
if self._io._is_tty():
    self._io.write("\x1b[?25l")  # hide
# ... work ...
if self._io._is_tty():
    self._io.write("\x1b[?25h")  # show
```

#### CJK Support

`clip()` in `surface.py` uses `unicodedata.east_asian_width()` for proper
display-width truncation. CJK characters count as 2 columns.

---

## Conventions

- **No PyTorch in kernel**: SloNet is pure NumPy. Never `import torch`.
- **No hardcoded paths**: Use `Path(__file__).resolve().parents[n]`.
- **No external downloads at runtime**: Buildroot builds offline.
- **SSE envelope**: `{"stream":"...","phase":"...","status":"...","data":{},"meta":{},"message":""}`.
- **RBAC at syscall boundary**: Role checked before dispatch.
- **Addons are modular**: Each addon registers via `install_addon()`.
- **ProcessGuard**: Circuit breaker; 3 failures → 30s open.
- **Metal accelerator**: Disable during train_step/train_batch/generate for embed_dim ≤ 128.
- **Test alongside**: Every public function gets a test.

## Testing Checklist

- [ ] `python3 -m py_compile <file>` after every edit
- [ ] `make test-py ARGS="tests/test_shell_*.py -x -q"` for shell changes
- [ ] `make test-py ARGS="tests/test_vm*.py -x -q"` for VM changes
- [ ] `make test-py ARGS="tests/test_shell_tui_repl.py -x -q"` for TUI changes
- [ ] Full shell suite before completion: `make test-py ARGS="tests/test_shell_*.py -q"`

## Running Python Code

```bash
# Core library
PYTHONPATH=packages/core-py .venv/bin/python -c "from domains.infrastructure.pugqeep import Tree; print('ok')"

# API server
cd apps/api && .venv/bin/python -m uvicorn server.main:app --port 8000

# CLI
.venv/bin/python -m apps.cli.src.cli

# Tests
PYTHONPATH=packages/core-py .venv/bin/python -m pytest packages/core-py/tests/test_file.py -x -v
```
