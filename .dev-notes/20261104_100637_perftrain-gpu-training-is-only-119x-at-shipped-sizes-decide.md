---
id: 20261104_100637_perftrain-gpu-training-is-only-119x-at-shipped-sizes-decide
status: todo
board: 20261104_100637_perftrain-gpu-training-is-only-119x-at-shipped-sizes-decide
title: "perf(train): GPU training is only 1.19x at shipped sizes — decide keep / batch / drop"
tags: ["training", "gpu", "performance", "benchmark", "decision", "area:training"]
sprint: "model-loading-v1"
gh: null
---

# perf(train): GPU training is only 1.19x at shipped sizes — decide keep / batch / drop

Follow-up to card `20261104_084408` (device story landed `ef8c88bce`). The accelerator is now a *choice* instead of an accident, so the next question is whether that choice is worth anything at the sizes we actually train. This card is the measured verdict + the decision, not new machinery.

**Measured — `scripts/benchmark_train_device.py`** (new, landed with 084408), n_embed=256 / 4 layers / 4 heads, batch=4, block=64, 5 steps after 2 warmup, CuPy 14.2.0 on a GTX 1050 (CC 6.1):

| device | ms/step | tok/s | gpu matmuls/step | cpu matmuls/step | MB/step host<->device | loss drift vs CPU |
|---|---|---|---|---|---|---|
| cpu | 695.55 | 368.1 | 0 | 20 | 0 | — |
| auto -> cuda | 582.26 | 439.7 | 20 | 0 | 30.9 | 3.2e-07 |

**1.19x.** Earlier hand prototype at the same depth with batch=8/block=128 (2x the bytes): numpy 803 ms/step vs cupy 677 ms/step — also **1.19x**, with 85 MB/step of traffic. The ratio is stable across shapes, which says the limit is the dispatch design, not the arithmetic.

**Why it is capped (measured, not assumed).** `slonet._matmul` dispatches only when `out_elems >= 16384`, and each dispatch is numpy-in / numpy-out: H2D both operands, cuBLAS, D2H the result — 20 host round-trips per training step, each with a stream sync. At 256-dim/4-layer the matmuls are small enough that PCIe + sync dominate. `dispatch_stats()` attributes all 20 to gpu with 0 fallbacks, so this is the real GPU path, not a silent CPU fallback.

**Options (the card's actual deliverable is the ruling).**
1. **Keep as-is** — GPU is opt-in (`SLO_ACCEL_BACKEND=cuda|auto`), CPU is the default for `TrainerConfig()`, and 1.19x is free for anyone who asks. Zero maintenance. Honest cost: the torch arm is declared but unexecuted here (torch absent; disk 99%, 2.1 GB free — see 084408 limits).
2. **Batch the dispatch** — keep activations resident on device for the whole step and cut ~20 round-trips to a handful. Real win, but it needs a device-tensor abstraction threaded through the hand-written autograd graph (`Tensor.data` is an ndarray everywhere, including checkpoints) — that is the "second engine" risk AGENTS.md's one-loop rule warns about. Scope must be proven small first.
3. **Fuse the per-step matmuls** — same idea at the kernel level (the fused-QKV / packed-FFN paths already exist for inference via `_get_weight_T_contig` and `_fuse_quant_weights`), no autograd change; reuses code we own.
4. **Drop GPU training** — keep the accelerator for inference only and delete the training pin/dispatch path. Only if the user's bar says 1.19x is not worth a maintenance seam.

**Blocking input from the user:** what win makes GPU training worth keeping? Below that bar, option 4 becomes the honest answer and 084408's device selection reduces to a CPU/torch switch.

**Known second-order effect to fold into the ruling:** `TrainerConfig()` constructed directly (`domain/infrastructure/_internal/training_queue.py:140`, the auto-training path) used to get CuPy matmuls *by accident* while labelled cpu; it now pins CPU as advertised. That path is now genuinely ~1.19x slower on a GPU host — the fix working, but it belongs in whatever we decide here.

**Out of scope:** torch benchmarking (unexecutable here), inference (already has its own fused/contiguous paths), `.soul` format, a second training loop.


---
**MEASUREMENT UPDATE 2026-11-04 (after torch removal `0576757ef`) — the 1.19x does not reproduce as a stable number; it is parity-within-noise at this shape.** Same script, same model (`n_embed=256`, 4 layers, 4 heads, batch=4, block=64), GTX 1050 + CuPy 14.2.0, host load average **8–11** (other sessions running):

| run | steps | cpu ms/step | cuda ms/step | ratio |
|---|---|---|---|---|
| earlier (this card) | 5 | 695.6 | 582.3 | **1.19x** |
| 1 | 5 | 331.7 | 325.0 | 1.02x |
| 2 | 5 | 243.2 | 252.9 | 0.96x |
| 3 | 5 | 294.3 | 225.2 | 1.31x |
| 4 | 20 | 297.0 | 269.6 | **1.10x** |

Absolute times swing 2–3x run to run because the box is shared; the *ratio* lands somewhere in 0.96x–1.31x, i.e. **roughly 1.0–1.2x**, never a clean win. The mechanism is unchanged and still the reason: ~20 per-op dispatches per step, each numpy-in / numpy-out with an H2D, a cuBLAS call and a D2H + sync (`dispatch_stats()` attributes all 20 to gpu, 0 fallbacks), so PCIe + sync eat the arithmetic at 256-dim/4-layer.

This weakens option 2 (batch the dispatch) as a *training* investment: the ceiling is the traffic, and the traffic is per-op by construction of the hand-written autograd graph. It strengthens the two cheap answers — **keep as it is** (opt-in, zero maintenance, honest label) or **drop the training pin** and leave the accelerator to inference, which already has the fused/contiguous paths (`_get_weight_T_contig`, `_fuse_quant_weights`). Still blocked on the same input: what win makes GPU training worth keeping?

