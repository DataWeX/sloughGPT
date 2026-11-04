---
id: 20261104_090736_npu-unify-one-device-two-seams
status: todo
board: 20261104_090736_npu-unify-one-device-two-seams
title: "Unify the two NPUDevice classes into one device on both seams + fix boot shadowing, CLI crash, .npy arg swap; validate with real Qwen inference through the VM device path"
tags: [shell, npu, devices, vm, refactor, bugs]
sprint: ""
gh: null
---
# NPU unify — one device, two seams (guest `call()` / kernel `ioctl()`)

## Problem (found by code dig, 2026-11-04)

Two `NPUDevice` classes with copy-paste lineage have drifted:

- `domain/shell/_internal/kernel_npu.py` — `DeviceDriver`, `PIPE`, no `COMPUTE`/`LIST_COMMANDS`,
  `__call__(model_name, input_data)`, `open()`/lock.
- `domain/shell/_internal/npu_device.py` — standalone, `PIPELINE`, `COMPUTE`, `IoctlCommand`,
  `NPUModel`, `__call__(input_data, model_name="")` — **reversed argument order**.

Concrete bugs hiding in the duplication:

1. **Boot shadowing** — `runtime.py:340-368` registers the standalone npu from
   `create_default_devices()` into `_devices` + `_device_system` ("npu", by "shell"), then
   **overwrites** `_device_system["npu"]` with `NPUVMDevice(kernel_npu.NPUDevice)` (by "kernel").
   `DeviceSystem.register` is a silent dict assign. Result: VM `DEV_OPEN "npu"` sees the kernel
   NPU; VFS/`/dev/npu`/DeviceManager sees the standalone one — **two `_models` registries under
   one device name** (invisible loads, double 2.5 GB weight copies).
2. **CLI model path crashes on line 1** — `apps/cli/src/cli.py:849` does
   `from domain.shell import NPUDevice` → ImportError (not exported anywhere); then calls
   nonexistent `load_model()`; fallback-to-mock below is unreachable dead code.
3. **`.npy` load broken in both** — `load()` calls `self._load_numpy(path, key)` but the def is
   `_load_numpy(self, name, path)` (swapped), and the body routes `.npy` to
   `SlonetChatProvider.from_slnc`. The correct helper `_load_numpy_array` has zero call sites in
   both files.
4. Spec drift — `docs/vm-devices-spec.md` documents the standalone vocabulary (`PIPELINE`,
   `COMPUTE`, 13 cmds) while the VM path actually runs the kernel class (`PIPE`, no `COMPUTE`).

## Plan (approved by user 2026-11-04)

Branch `feat/npu-unify`:

1. One `NPUDevice` = unified `kernel_npu.NPUDevice(DeviceDriver)`; absorb `IoctlCommand` input,
   `COMPUTE`, `LIST_COMMANDS`, `NPUModel`/`model()`; `PIPE` + `PIPELINE` both accepted (alias);
   `__call__` unified to `(model_name, input_data)` with a compat dispatch for the standalone
   `(input_data, model_name=...)` shape. `npu_device.NPUDevice` becomes a thin compat subclass/
   re-export so no importer breaks.
2. Register **one shared instance on both seams** in `runtime.py` (no overwrite; device_system
   and `_devices["npu"]` are the same object) — cross-seam model visibility.
3. Fix bugs 2 (CLI uses the unified device; `load_model` → `ioctl("LOAD", ...)`; real import)
   and 3 (`.npy` → `_load_numpy_array` with correct args).
4. Validation, two tiers:
   - fast contract tests: cross-seam registry visibility, merged vocabulary (PIPE/PIPELINE),
     `call()`+`ioctl()` faces, failure paths (existing `test_device_contract.py` stays green);
   - **slow real-inference gate**: `NPUVMDevice` (the VM guest face — `DEV_CALL` vocabulary)
     loads `models/hf-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/model.slnc` and answers
     real-world prompts (fact question, multi-turn chat, long context); property assertions
     (non-empty, within token budget, stops), skipif model absent, offline, `-o addopts=''`.
5. Update `vm-devices-spec.md` to the merged surface; affected suites via
   `scripts/run_affected.sh`; benchmark dispatch (new `scripts/benchmark_npu_device.py` if none
   fits); merge green per SOP.

## Why it matters

The CLI is the safe prototype harness and the VM is the real surface — they must share ONE
device so "prototyped in CLI == works in VM" (user, 2026-11-04). The VM NPU device is a
scaled-down real machine: validate it with **actual real inference on real-world scenarios and
data**, not mocks (mocks only for failure-path branching).
