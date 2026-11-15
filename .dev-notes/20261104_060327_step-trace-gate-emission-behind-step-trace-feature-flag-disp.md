---
id: 20261104_060327_step-trace-gate-emission-behind-step-trace-feature-flag-disp
status: done
board: 20261104_060327_step-trace-gate-emission-behind-step-trace-feature-flag-disp
title: "step-trace: gate emission behind step_trace feature flag (dispatch counters always-on)"
tags: ["training", "telemetry"]
sprint: ""
gh: null
---

# step-trace: gate emission behind step_trace feature flag (dispatch counters always-on)

Goal: put the StepTrace emission behind the existing feature-flag registry (domain/shared/_internal/feature_flags.py) — env vars and feature flags never mix. Flag gates ONLY emission (ring assembly + on_progress residency fields); dispatch counters in domain/slolib/_internal/gpu.py stay unconditional so the step-1 tripwire cannot be switched off. Default: enabled.

Scope:
- feature_flags.py: register step_trace (ENABLED, description).
- config/feature_flags.json: "step_trace": "enabled" (parity with registry).
- train_pipeline.py: one is_enabled('step_trace') seam at ring append + on_progress residency keys (device stays unconditional).
- tests/test_step_trace.py: flag off -> no ring traces + on_progress omits residency keys, but dispatch_stats() counters still grow.
Gates: ruff, baseline, step-trace + feature-flag tests, affected sweep, training benchmark, domain rehash.
