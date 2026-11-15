---
title: One device contract across all four seams
date: 2026-10-06
status: in-progress
branch: feat/device-contract
tags: [shell, devices, vm, kernel, refactor, architecture]
---

# Problem

Which methods a device had depended on which base class it happened to
inherit — so each device was reachable through exactly one seam:

| base | had | seam |
|---|---|---|
| `vm.Device` | `call`, `info` | `DeviceBus` |
| `DeviceDriver` | `ioctl`, `info` | `DeviceManager` (fd) |
| `AIDevice` | `read`, `write` | `devices.DeviceManager` (path) |
| standalone (npu, display, tensor, storage, network, input) | `call`, `ioctl`, `info` | both |

Two different classes were both named `DeviceManager`. Costs, each found by
actually driving a seam rather than by reading code:

- `vm.py` `DEV_CALL` calls `_devices.call(...)`, but the kernel
  `DeviceManager` had no `call()` — the trap hit while building the NPU bridge.
- `devices.DeviceManager.list_devices()` calls `.info()` on every registered
  `AIDevice`, which had none → `AttributeError` listing `/dev/llm`.
- display/input/network/storage/tensor each carried an *identical* copy of
  `call()` raising a bare `Exception`, so `DEV_CALL` classified a normal
  device refusal as a crash ("exception") instead of `DeviceFault` ("error").
- `npu_device.call("info")` failed because `ioctl` only accepts uppercase
  `IoctlCommand`, and it raised `Exception` too.

# The contract

Two faces of one operation; each world already had exactly one, this gives
each the other.

| face | returns | on failure |
|---|---|---|
| `call(method, *args)` | the value | raises `DeviceFault` |
| `ioctl(command, *args)` | `SyscallResult` | never raises |

# Landed

| commit | contents |
|---|---|
| `c31be407c` | `scripts/benchmark_device_dispatch.py` (ns/call for 4 dispatch shapes, legacy body kept as reference, `--ci` >20% gate); hoisted the deferred `import` out of `vm.Device.call`'s hot path — 2978 → 1348 ns/call |
| `a740af3ff` | `device_ops.OpTable` (Mapping) + `vm.Device.call()` as the shared dispatcher; 8 verbatim dispatcher bodies deleted (Tensor/Slonet/Multimodal/Engine/SlonetTraining/NPUVM/Cycles/RenderNeural); each table keeps its exact fault string |
| `41b095c93` | `DeviceDriver.call()` (unwraps ioctl + translates the case dialect once), `vm.Device.ioctl()` (wraps call), `DeviceManager.call(fd)/info(fd)`, `AIDevice.info/call/ioctl`, `devices.DeviceManager.call(name)/info(name)`, five copy-pasted `call()` bodies → `DeviceFault`, `_dispatch_command` → `DeviceFault`; new `packages/core-py/tests/test_device_contract.py` (24 tests, all four seams) |
| `11c3b870f` | `.slnc` is the NPU's only model format; file readers off the model path (branch `feat/npu-slnc-only`) |

# Verified

- `tests/` **3214 passed**, `apps/api/server/tests` **1223 passed**, ruff clean
  (run after each of the two commits above).
- 102-file device/VM/kernel/neural core-py run: 54 failures, **all** in the
  recorded baseline `.tmp_names.txt` — zero new. The remaining drift
  (`no attribute 'open'`, `DeviceTable.read_fd`) is stale pre-refactor API
  usage, tracked on card `33773fce`.

# Deliberately not done

- **Op-name aliasing across seams** — `call("load_model")` vs `ioctl("LOAD")`
  still differ. That is the op-table layer, not the interface; the directive
  was to fix the interfaces first.
- **`load_file`** for the non-model readers (npy/csv/py) moved off the NPU's
  model path in `11c3b870f` — pending the IO-split design.
- Hardware-faithful `vm.py` devices (Clock/CMOS/PIT/NIC/Disk/IRQ/VGA/PS2) keep
  bespoke op sets on purpose.

# Next

1. **`load_slnc` (models) / `load_file` (data)** — the npy/csv/py readers
   removed from the NPU's model path in `11c3b870f` live again under a
   file-reading API, on the shared layer rather than per device.
2. **OpSpec layer** — per-op arity/result declared on the table so an arity
   mistake faults at the boundary instead of inside a handler
   (`_tokenize() takes 3 but 4 given` was the original symptom), plus
   cross-seam op-name aliasing (`call("load_model")` vs `ioctl("LOAD")`).
3. Remaining core-py buckets (~336 of 517), then a PR for
   `fix/shell-repl-tests`.
