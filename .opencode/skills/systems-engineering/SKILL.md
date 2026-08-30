---
name: systems-engineering
description: >
  Use when building OS components, kernel modules, device drivers, init system
  services, Buildroot configurations, x86 VM extensions, or shell TUI infrastructure
  in sloughGPT. Covers the Dait kernel, x86 VM, v86 browser Linux, and custom
  Buildroot image builds.
---

# Systems Engineering Skill

## Overview

This skill provides the architectural knowledge and coding conventions for building
OS-level components in sloughGPT. The Dait operating system has two deployment
targets: a custom Buildroot image for the v86 browser VM, and a Python x86 emulator
for training and development.

## When to Use

- Writing or modifying kernel code (scheduler, memory, syscalls, addons)
- Writing or modifying device drivers (DeviceBus, DeviceDriver, fd I/O)
- Writing or modifying init system services (runlevels, service lifecycle)
- Extending the x86 VM (new instructions, syscalls, RBAC roles)
- Building or configuring Buildroot images
- Working on the shell TUI (pane engine, surfaces, borders, cursor lifecycle)
- Working on VFS (mount points, /dev/*, /proc/*, host fs bridging)

## Architecture Reference

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

## Key Files

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

## Kernel Development

### Syscall Table (INT 0x80)

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

### Adding a New Syscall

1. Define the syscall number in `vm.py` (add to `SYSCALL_TABLE`)
2. Implement the handler function in `vm.py`
3. Add RBAC permission mapping in `vm_permissions.py`
4. Add the assembler mnemonic if user-facing
5. Write tests in `tests/test_vm*.py`
6. Update this skill doc

### Adding a New Kernel Addon

1. Create `packages/core-py/domains/shell/addons/<name>.py`
2. Implement the `Addon` protocol from `base.py`
3. Register via `kernel.install_addon(<name>)` in `runtime.py`
4. Add test in `tests/test_shell_runtime.py`

### Dynamic Module Loading

Use `ModuleLoader` for runtime addon management:

```python
from domains.shell.addons.module_loader import ModuleLoader

loader = ModuleLoader(addon_dirs=["path/to/addons"])
loader.set_kernel(kernel)

# Discover available modules
available = loader.discover()

# Load a module
addon = loader.load("my_addon")

# Hot-reload during development
addon = loader.reload("my_addon")

# Unload
loader.unload("my_addon")

# Query state
print(loader.summary())
```

### VM Debugger

Use `Debugger` for interactive debugging:

```python
from domains.shell.vm_debugger import Debugger

debugger = Debugger()
debugger.set_output(print)

# Load source
engine = debugger.engine
engine.load_source(source)
debugger.load_symbols(source)

# Set breakpoints
debugger.bp_set("main")
debugger.bp_set(0x1000, "loop_start")

# Step through
debugger.stepi()           # single instruction
debugger.step_over()       # step over CALL
debugger.step_out()        # run until return

# Inspect state
debugger.dump_regs()
debugger.dump_flags()
debugger.dump_memory(0x1000, 64)
debugger.dump_stack(8)

# Continue execution
trace = debugger.continue_exec()

# Analyze
analysis = debugger.analyze_trace(trace)
```

### Adding a New Device Driver

1. Subclass `DeviceDriver` in `device_system.py`
2. Implement `open()`, `call()`, `close()` methods
3. Register with `DeviceSystem` in `devices.py`
4. The device appears at `/dev/<name>` in VFS

## Buildroot Build System

### Directory Structure

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

### Build Commands

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

### v86 Integration

The v86 browser VM loads a raw disk image. After Buildroot builds:

1. The image is at `buildroot/output/images/rootfs.ext2`
2. Convert for v86: `dd if=buildroot/output/images/rootfs.ext2 of=buildroot/output/images/buildroot.img bs=512`
3. Serve from `apps/web/public/buildroot/` or a CDN
4. Update `LINUX_IMAGE_URL` in `apps/web/hooks/useV86.ts`

### Defconfig Essentials

A minimal Buildroot defconfig for sloughgPT should include:

- Architecture: x86_64 (for v86 compatibility)
- Kernel: Linux 6.x (minimal config)
- Init: BusyBox init or custom
- Packages: busybox, bash, coreutils, python3 (optional)
- Filesystem: ext2/raw
- Overlay: custom rootfs overlay for Dait packages

## Shell TUI Architecture

### Pane Layout

The pane engine (`pane.py`) provides pure-geometry layout:

```python
layout = PaneLayout([
    Pane("console", ratio=0.3, min_rows=3, border=Border("all")),
    Pane("output", ratio=0.7, min_rows=3, border=Border("all")),
    Pane("status", fixed=1),
    Pane("input", fixed=1),
])
regions = layout.compute(term_rows, term_cols)
```

### Rendering Pipeline

```
TuiRepl._render_all()
  ├─ _draw_borders(stdscr, regions)     # Border characters
  ├─ _blit(win_console, log_lines, oy, ox)  # Log surface content
  ├─ _blit(win_output, output_lines, oy, ox)  # Output surface content
  ├─ _render_status(win_status)          # Status bar
  └─ _render_input(win_input)            # Input line + cursor
```

### Cursor Lifecycle

All ANSI cursor operations must be guarded:

```python
if self._io._is_tty():
    self._io.write("\x1b[?25l")  # hide
# ... work ...
if self._io._is_tty():
    self._io.write("\x1b[?25h")  # show
```

### CJK Support

`clip()` in `surface.py` uses `unicodedata.east_asian_width()` for proper
display-width truncation. CJK characters count as 2 columns.

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
