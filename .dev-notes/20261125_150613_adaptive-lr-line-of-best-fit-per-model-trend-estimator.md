---
id: 20261125_150613_adaptive-lr-line-of-best-fit-per-model-trend-estimator
status: todo
board: 20261125_150613_adaptive-lr-line-of-best-fit-per-model-trend-estimator
title: "adaptive-config: replace flat lr average with per-model line-of-best-fit (trend up/down)"
tags: ["training", "adaptive-config", "area:domain"]
sprint: ""
gh: null
---

# adaptive-config: per-model line-of-best-fit learning-rate estimator

**Origin:** user proposal during card 20261107_112103 triage (2026-11-25) — "the learning rate should be variable, based on model-level statistics [knowing whether to] increase or decrease it — line of best fit."

**Problem:** `AdaptiveConfigEngine._recommend_from_history` (domain/training/_internal/adaptive_config.py) computes the recommended `learning_rate` as a **flat quality-weighted average** of past runs. It has no notion of trend, no per-model fit, and no robustness: a contaminated sample (e.g. the `TrainingOutcome.learning_rate = 0.0` sentinel records, incident CI run 45612603707) collapses the recommendation to an unappliable value. The incident itself was patched (sentinel filter + `lr > 0` guard at application, landed with card 20261107_112103) — this card is the *feature* that makes the estimator statistically sound instead of merely guarded.

**Build:**
- Per-model (and per-method) regression of lr against run outcome (quality_score / final_loss) — line of best fit — so the recommendation **moves with evidence**: trending up when later runs at higher lr perform better, down when they degrade.
- Robustness requirements (the incident's lesson, non-negotiable): the estimator must be **total** — defined for n=0 (fallback heuristics), n=1, and contaminated samples — and must never emit `lr <= 0` (PipelineConfig rejects it in `__post_init__`).
- Keep the existing size/model/method matching tiers; only the estimator under `_recommend_from_history` changes.

**Not in scope:** the sentinel filter and application guard (already landed); `TrainingOutcome` default-lr cleanup (separate if ever needed).

**Evidence / acceptance:**
- Existing `test_adaptive_infra.py` recommend tests stay green (they pin `lr > 0`, `based_on_runs`, fallback `confidence == 0.3`).
- New tests: monotone trend (quality improves as lr rises → recommendation rises), reverse trend, single-run degeneracy, poisoned-sample safety.
- Recommendation values remain applicable: every emitted lr passes `PipelineConfig` validation.
