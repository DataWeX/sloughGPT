---
description: >
  Run systems engineering on the specified component. Usage: /systems <target>
  Designs and implements OS components, kernel modules, drivers, Buildroot configs,
  or VM extensions.
agent: systems-engineer
---

# Systems Engineer Command

Design and implement OS-level components for the Dait operating system.

## Usage

```
/systems <target>
```

## Examples

```
/systems kernel                    # Work on the Dait kernel
/systems init                      # Work on the init system
/systems vm                        # Work on the x86 VM
/systems buildroot                 # Build/configure custom Buildroot image
/systems driver <name>             # Write a new device driver
/systems syscall <name>            # Add a new syscall
/systems addon <name>              # Write a new kernel addon
/systems pane                      # Work on the shell TUI pane engine
/systems surface                   # Work on content surfaces
/systems tui                       # Work on the TUI display layer
/systems v86                       # Work on browser Linux (v86)
```

## What Gets Built

| Target | Output |
|--------|--------|
| `kernel` | Scheduler, memory, syscalls, addon loader changes |
| `init` | Service definitions, runlevel changes, dependency graph |
| `vm` | CPU instructions, syscall table, RBAC rules |
| `buildroot` | Defconfig, overlays, packages, build scripts |
| `driver` | New DeviceDriver subclass + registration + tests |
| `syscall` | New INT 0x80 handler + RBAC + assembler mnemonic + tests |
| `addon` | New kernel addon module + registration + tests |
| `pane` | Pane layout engine changes (Border, Pane, PaneLayout) |
| `surface` | TextSurface, LogSurface, clip, CJK support |
| `tui` | TuiRepl rendering, input handling, cursor lifecycle |
| `v86` | V86Controller, useV86 hook, image build integration |

## Verification

After each change:
```bash
python3 -m py_compile <file>
make test-py ARGS="tests/test_shell_*.py -x -q"
```

## Output

The agent will report:
1. Files changed
2. Architecture impact
3. Tests run and results
4. Buildroot build status (if applicable)
