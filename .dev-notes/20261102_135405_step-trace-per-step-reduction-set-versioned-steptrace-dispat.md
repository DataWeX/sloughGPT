---
id: 20261102_135405_step-trace-per-step-reduction-set-versioned-steptrace-dispat
status: done
board: 20261102_135405_step-trace-per-step-reduction-set-versioned-steptrace-dispat
title: "step-trace: per-step reduction set (versioned StepTrace) + dispatch residency counters"
tags: ["training", "telemetry", "observability"]
sprint: ""
gh: null
---

# step-trace: per-step reduction set (versioned StepTrace) + dispatch residency counters

Goal: a fixed, versioned per-step trace (the 'reduction set' design — ship receipts, not cargo) so silent-degradation failures like job_c87a2dd6 (0.4 steps/s, GPU never used) surface on step 1 instead of hour 3.

Scope v1:
- domain/slolib/_internal/gpu.py: module-level dispatch counters (gpu_matmul, cpu_matmul, fallback, h2d_bytes, d2h_bytes) at both backends' matmul + drain_dispatch_stats() delta API.
- domain/training/_internal/trace.py (new): STEP_TRACE_VERSION, STEP_TRACE_FIELDS ratchet, frozen StepTrace dataclass, fingerprint_batch() (blake2b of batch bytes), to_dict().
- train_pipeline.py: assemble StepTrace per step (drain counters, fingerprint x, grad_norm from clip site), bounded ring (deque maxlen=128) + get_step_trace() accessor; include residency fields in on_progress emit.
- Deferred to v2 (schema is versioned): queue_depth, update_norm, ES probe fields — require the async pipeline / ES mode that does not exist yet. No fake fields.

Gates: TDD (tests/test_step_trace.py first, red then green), ruff, targeted pytest, benchmark_slonet_training.py for no-regression.
