---
id: 20261107_113837_training-throughput-14-stepss-vs-documented-31-trainer-threa
status: done
board: 20261107_113837_training-throughput-14-stepss-vs-documented-31-trainer-threa
title: "training throughput 1.4 steps/s vs documented 31+ (trainer thread ~18% CPU)"
tags: ["training", "performance", "area:training"]
sprint: ""
gh: null
landed: 66179db05 + efc6d673b (main)
---

Evidence job_cbc6162a: device=cpu (by design, SloNet pure numpy), @_gc_parked active on train() so the documented GC fix applies, yet measured 1.4 steps/s while train_pipeline.py:864 documents 29 vs 31+ steps/s with GC parked. API process averages ~18% CPU during the run -> thread mostly waiting, not computing. Known incident class: trace.py documents job_c87a2dd6 (0.4 steps/s, GPU idle, silent CPU fallback). Overhead visible in logs: eval every 100 steps (~2.5s) + checkpoint .soul save roughly every 60-90s (16 saves/28min). Use StepTrace tripwires (device, gpu_matmuls, cpu_matmuls, fallbacks) to profile where the time goes.

## Resolution — 2026-11-16, pushed `efc6d673b` (fix `66179db05`, sync `efc6d673b`)

**Three causes, one card:**

1. **The reported 1.4 was a whole-run metric, not step speed.** `_steps_per_sec()` = `global_step / elapsed`, counting eval (~2.3s) + checkpoint save (**22s pre-fix**) every 100 steps: `100 / (47.6 + 2.3 + 22) ≈ 1.39` is exactly the card's number. The save path was already fixed the same week (`b3bcea99d` gzip meta + `213dafc6a` binary optimizer sidecar, 16.53s → 0.470s) — re-measured here at **521ms steady**. This also explains the ~18% CPU: the thread was blocked on save/eval, not waiting on a lock.
2. **"Documented 31+" was a micro-figure.** `e6d254688` (Sep 21) `_gc_parked` docstring — a parked-GC measurement on a small config, never whole-run throughput. Docstring now labels it as such and points at `scripts/benchmark_slonet_training.py` as source of truth; `_steps_per_sec` docstring said "Rolling" but computes whole-run — corrected, citing this card.
3. **The genuine compute gap: gelu.** cProfile on the evidence job's shape (128d/4L/4H/block128/batch8) put ~50% of train-step wall in gelu fwd+bwd (tanh term computed twice in forward, `sqrt(2/pi)` three times, backward closures rebuilding the elementwise derivative per call). Fixed as one-pass `_gelu_tv()` + hoisted constants + derivative cached in closures.

**Evidence:** back-to-back A/B raw **2.97 → 4.18 steps/s (+41%)**; benchmark history recorded (`--kind training`, compare `[OK] no regression`): gate **2.5 → 14.87**, tiny **5.92 → 10.73**, small **0.99 → 3.42**, medium **0.15 → 0.46** steps/s with **bit-identical final losses** (0.0007/3.3579/3.0232/2.5548).

**Gates:** ruff clean; `domain_baseline_rehash --check` no drift (hashes-only 4-line diff); seam gates 14 passed; fast affected closure 121 passed / 5 failed — all 5 pristine-proven pre-existing (ml_types-cuda, numba, datasets-SSRF ×2, e2e needs :5173). Slow-phase journey suite running in background at push time (0 failures) — Test Rule right-sizes a training-loop change to unit + affected closure + benchmark; CI full suite is the backstop.

**Residual (not this card):** remaining step time is real numpy compute (`np.matmul` ufunc time attributed to the caller — `bk` tottime is matmul, not Python overhead); next lever is structural (fused kernels / slolib) if pursued.
