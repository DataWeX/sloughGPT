---
id: 20261104_090736_npu-unify-one-device-two-seams
status: in_progress
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

## Results (2026-11-04)

Branch `feat/npu-unify` (worktree `wt-npu`, based on main 5d217fc6c).

**Landed**
1. `kernel_npu.NPUDevice` is the one driver; `npu_device.NPUDevice` = re-export shim;
   `from domain.shell import NPUDevice` restored via the lazy `_LAZY_IMPORTS` table
   (the root cause of the CLI crash, not just a routed-around symptom) + added to `__all__`.
2. Merged vocabulary: `PIPE`/`PIPELINE` alias, `COMPUTE`, `LIST_COMMANDS`, `NPUModel`.
   Unknown-command message carries both pinned dialects ("unknown command" + "unknown ioctl").
3. `__call__` accepts both historical shapes: `(model_name, input)` and `(input, model_name=...)`.
4. Boot registers ONE shared instance behind the VM face — no more silent overwrite / dual registry.
5. `.npy` loader arg-order fixed; CLI `load_model` → `ioctl("LOAD", ...)`; CLI provider fetch → `npu.model(m).provider`.
6. `NPUVMDevice._IOCTL_MAP["generate"]` now passes `max_tokens` through (was silently dropped).

**Bugs the real-inference gate caught (mocks had been hiding them)**
- `SloNetChatProvider` has **no** `generate_numpy` — that protocol lives on the *private* inner
  model. `_execute_text` only worked against mocks; every real text call raised
  `DeviceFault: 'SloNetChatProvider' object has no attribute 'generate_numpy'`.
  Fixed: prefers the provider's public `generate(prompt, max_tokens)` when `generate_numpy`
  is absent. Token count is reported from the continuation alone (the public API echoes the prompt).
- Benchmark caught my own 10x blowup: `info()` rebuilt `TensorDevice.list_commands()`
  (sorted over 59 ops) on every call → INFO 1.59µs → 17.3µs. Fixed by resolving the static
  command set once at `__init__`.

**Benchmark (interleaved base/current on the same machine state, load ~1.4, 5000 iters, p50)**

| op | base | current | Δ |
|---|---|---|---|
| ioctl_INFO | 1.59µs | 2.72µs | +71% (+1.13µs) |
| ioctl_CALL_text | 5.96µs | 6.80µs | +14% (+0.84µs) |
| vmface_generate | 8.80µs | 10.52µs | +19% (+1.72µs) |
| ioctl_LIST_COMMANDS | 1.18µs | 1.33µs | +13% (+0.15µs) |

Accepted: the INFO payload doubled (5 keys → 10 keys, `compute_ops` is contract-required by the
standalone face); micro-breakdown puts `info()` itself at 1.57µs of the 2.72µs, so the delta is
payload volume, not waste. Absolute cost stays sub-3µs on the DEV_CALL path.
(Uncontrolled runs on a box at load 8.05 showed CALL at 13µs — noise; always interleave.)

**Tests**
- fast: `test_npu_unified.py` (19) + `test_kernel_npu.py` + `test_npu_device.py` +
  `test_device_contract.py` + `test_vm_devices_new.py` + `test_vm_devices.py` → **484 passed**
- slow (real Qwen2.5-0.5B `.slnc`, offline, property assertions): `test_npu_real_qwen.py` → **5 passed**
- `ruff check` clean on all changed files; `mypy` clean on the NPU modules.
  (`ruff format` on `vm_devices.py` is pre-existing red at base — not reformatted, to keep the diff honest.)

**OPEN FINDING — needs approval (not implemented)**
`NPUVMDevice` advertises `tokenize`, `embed`, `forward` in its own docstring
(`DEV_CALL R2, R0, tokenize, qwen, Hello world` …), but verified against the REAL device with the
REAL model: all three raise `DeviceFault: NPU: op not supported`. `detokenize` same class.
Neither historical NPU class ever had them — a **pre-existing** gap, not a regression from this card.
`info`, `generate`, `load_model`, `unload_model` all work.
Proposed fix (small): add `tokenize`/`detokenize`/`embed`/`forward` delegation methods on the
unified `NPUDevice` that forward to the loaded provider. Awaiting user go-ahead per approval rule.
