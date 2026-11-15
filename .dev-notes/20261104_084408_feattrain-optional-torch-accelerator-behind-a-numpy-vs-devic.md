---
id: 20261104_084408_feattrain-optional-torch-accelerator-behind-a-numpy-vs-devic
status: done
board: 20261104_084408_feattrain-optional-torch-accelerator-behind-a-numpy-vs-devic
title: "feat(train): optional torch accelerator behind a numpy-vs-device parity gate"
tags: ["models", "training", "gpu", "parity", "torch"]
sprint: "model-loading-v1"
gh: null
---

# feat(train): optional torch accelerator behind a numpy-vs-device parity gate

Scope: option 4 from the CPU/GPU discussion ("optional torch backend behind a parity gate") — built on the seam that already exists instead of a second training loop.

**Measured before building (port, don't rebuild).**
- Training math already runs on a hand-written autograd engine (`domain/training/_internal/slonet.py`, 8648 L) whose hot ops go through `_matmul` / `_accel_op` into `domain/slolib/_internal/gpu.py::get_accelerator()`. That layer already has `_CUDABackend` (CuPy), `_MetalBackend`, `_OpenCLBackend`, `_CPUBackend`, a bounded probe (`SLO_ACCEL_PROBE_TIMEOUT`), and dispatch counters `{gpu_matmuls, cpu_matmuls, fallbacks, h2d_bytes, d2h_bytes}` (`dispatch_stats()`).
- `TrainerConfig.device` (`train_pipeline.py:430`) is inert: `"auto"` -> `"cpu"`, `_setup_device()` returns `"cpu"` unconditionally, so the accelerator is only ever reached by accident.
- `cupy 14.2.0` IS installed and the host has a GTX 1050 (CC 6.1) — a GPU training route exists today. `torch` is NOT installed (`ModuleNotFoundError`); disk is 99% (2.1 GB free), so CUDA-torch is not installable here.
- No parity test exists between a device arm and the NumPy arm: existing GPU tests assert ops against numpy refs, but never a *training trajectory*, and never that dispatch actually left the CPU.

**Build (one seam, N backends, one gate).**
1. `_TorchBackend(_Accelerator)` in `domain/slolib/_internal/gpu.py` — same protocol as `_CUDABackend` (`to_device/from_device/matmul/...`), torch tensors, **opt-in** (`SLO_ACCEL_BACKEND=torch`) so default selection stays byte-identical.
2. Explicit backend selection: `SLO_ACCEL_BACKEND=cpu|torch|cupy|auto`; an explicit request that is unavailable must refuse loudly instead of silently degrading (the silent-degradation class the dispatch counters were added for).
3. Make `TrainerConfig.device` truthful: `auto` resolves through the accelerator and the resolved name is recorded (StepTrace + training_info); `device="cpu"` forces the NumPy backend.
4. Parity gate `tests/core-py/test_train_device_parity.py`, three levels: op parity vs numpy refs; **trajectory parity** (same init, same batches, N train steps -> loss/grad-norm drift within tol) with a tripwire asserting `dispatch_stats()["gpu_matmuls"] > 0` (otherwise the "GPU" arm was secretly NumPy and the gate proves nothing); and a **gate self-test** (a deliberately drifted backend must be caught).
5. `scripts/benchmark_train_device.py` — steps/sec + dispatch attribution per backend on one real preset, JSON out; this decides whether GPU training is worth it at current model sizes.

**Known limit, stated up front:** dispatch is per-op with host round-trips (`to_device`/`from_device` inside `_matmul`), so GPU wins only above the 16384-element output threshold. At TinyShakespeare sizes it may lose; the benchmark is the arbiter, not the assumption.

**Out of scope:** nn.Module rewrite / second trainer (AGENTS.md one-loop rule), `.soul` format, inference paths, boot.


---
**CLOSED 2026-11-04 — landed on main `ef8c88bce` (lane `feat/train-device`, pushed `HEAD:main`).**

**Built (device story):**
- `SLO_ACCEL_BACKEND=cpu|cuda|metal|opencl|torch|auto` (+ aliases `cupy`/`gpu`/`nvidia`, `mps`/`apple`, `numpy`). Unset/`auto` keeps the historical priority order byte-identically; an explicit value resolves or **raises** (`accelerator_for()` / `select_backend()`). Boot probe logs and serves CPU rather than dying.
- `_TorchBackend(_Accelerator)` on the existing seam — numpy in / numpy out, opt-in only. No second trainer, no `nn.Module` rewrite (one-loop rule).
- `TrainerConfig.device` no longer collapsed in `__post_init__` (validated instead); `_setup_device()` resolves once, records the resolved backend, and `train`/`evaluate`/`generate` push it as a **thread-local pin** — `device="cpu"` now really silences the GPU, for that job only. Default stays `"cpu"`; API `device=None` → `"cpu"`, so defaults are unchanged.
- `tests/core-py/test_train_device_parity.py`: op parity + **trajectory parity** + `dispatch_stats()["gpu_matmuls"] > 0` tripwire + negative control (1% matmul error must be caught; parameters are the sensitive arm — cross-entropy ~4.6 masks the loss arm).
- `scripts/benchmark_train_device.py`: ms/step, tokens/s, dispatch attribution, host↔device bytes, drift vs CPU, JSON out.

**Fixed (found by the gate; pre-existing on main, not device-related):**
`SloLinear._get_weight_T` cached `weight.T()` as a **graph node**. It is not in `parameters()`, so `zero_grads` never cleared it, and every `backward()` added to it (matmul backward + transpose backward both `a.grad.data += ...`): the real weight's gradient grew **1×/2×/3×** — 9 params (`ff.w1/w3.weight`, `ff.w2`, `attn.W_o`, `lm_head`), up to **2.0 relative drift after 3 identical steps**; invisible at step 1, likely masked by grad clipping. Fixed by building the transpose node per forward pass (strided view, one allocation); `_weight_T_contig` (inference data cache) untouched. `test_get_weight_T_cached` replaced (it asserted the unsafe identity); `TestRepeatedBackward` pins "3 identical backwards == 1" and **fails on the old code** (verified: 9 params drifted, worst 2.000).

**Measured:** `benchmark_train_device.py` n_embed=256/4L: CPU **695.6 ms/step** vs CUDA **582.3** = **1.19×**, 30.9 MB/step host↔device, loss drift **3.2e-07**. `benchmark_slonet_training.py`: gate 55.4K params **1.17 s/s**, loss 0.0007, PPL 1.0; tiny 21.3K **2.37 s/s**, loss 3.071, PPL 21.56.

**Gates:** parity **13 passed / 1 skipped** (torch absent → its refusal path is covered, its execution is not), device/GPU/config closure **521 passed**, slonet/train closure **443 passed**, ruff check+format clean. Three tests that encoded the old inert contract updated (`auto` preserved for `_setup_device`; `set_accelerator_precision("fp16")` follows the backend's `_fp16_available` instead of asserting a CPU host).

**Limits / out of scope:** torch unexecuted here (99% disk, 2.1 GB free — CUDA-torch not installable, CPU-torch needs approval); per-op dispatch with host round-trips means 1.19× is the ceiling at this shape. `.soul` format, inference paths, boot untouched.


---
**SUPERSEDED IN PART 2026-11-04 — landed `0576757ef` (lane `feat/train-device`, pushed `HEAD:main`).** User ruling: *"we don't need torch, we kick the torch out and replace it with SloNet and it works."*

1. **`_TorchBackend` is deleted** (~250 lines, `domain/slolib/_internal/gpu.py`). The backend seam is now `cpu|cuda|metal|opencl`. Removal is a **loud refusal, not a silent fallback**: `_REMOVED_BACKENDS` holds the reason once, so `select_backend("torch")` / `accelerator_for("torch")` / `TrainerConfig(device="torch")` all raise `torch was removed — SloNet's NumPy engine is the only training backend`; a stale `SLO_ACCEL_BACKEND=torch` in a shell profile still boots (probe logs, serves CPU — pinned by `test_removed_backend_in_the_env_still_boots`).
2. **`TrainerConfig` now validates against `gpu.known_backends()`** instead of its own hand-kept set — one declaration, so the config vocabulary cannot drift from the seam. Side effect: `opencl` / `numpy` were always supported by the seam but rejected by the config; they are accepted now.
3. **Correction to this card's note above:** an unspecified API `device` resolves to **`"auto"`**, not `"cpu"` — `SloughGPTTrainer` does `device=device or "auto"` (`train_pipeline.py:1058`) and `_setup_device()` then pins the resolved backend. Compute is unchanged; only the sentence was wrong.
4. **Tests:** parity **15 passed / 0 skipped** (was 13 passed / 1 skipped — the torch arm became `test_torch_is_refused`; the unavailable-declaration test now uses a known-but-unservable backend so it stops skipping). Device/GPU/config/trainee closure **717 passed / 6 skipped**, slonet/train closure **309 passed**, ruff clean.

Everything device-independent from the original note still stands: the selection seam, the thread-local pin, the truthful resolved-device record, the parity gate with its `gpu_matmuls` tripwire and negative control, `benchmark_train_device.py`, and the `_get_weight_T` gradient-accumulation fix.

