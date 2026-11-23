---
id: 20261101_182428_refactorinference-reconcile-the-two-meta-weight-bound-sets-a
status: done
board: 20261101_182428_refactorinference-reconcile-the-two-meta-weight-bound-sets-a
title: "refactor(inference): reconcile the two meta-weight bound sets at the seam"
tags: ["cleanup", "inference", "feedback", "tech-debt", "duplication"]
sprint: ""
gh: null
---

# refactor(inference): reconcile the two meta-weight bound sets at the seam

Two bound sets now govern meta-weight output at two layers, and the lower one is generation-UNSAFE.

(1) DOMAIN — _SAFE_BAND in domain/feedback/_internal/meta_weights.py get_adjustment() (commit 472565281): tight neutral+/-band window. temp (0.3,1.1), rep (1.0,1.25), top_p (0.6,1.0), top_k (10,120). Fixes the /chat infinite-repetition loop at the source.

(2) TRANSPORT — _META_WEIGHT_BOUNDS in apps/api/server/routers/inference.py _apply_meta_weights() (commit 3384d08dd, card 1e13f836): clamps to GenerateRequest FIELD bounds, drift-gated by tests/test_meta_weights_clamp.py. temp (0,2), rep (0.5,2), top_p (0,1), top_k (0,500).

THE PROBLEM: (2) is schema-valid but generation-unsafe. repetition_penalty 0.5 passes pydantic and ACTIVELY REWARDS REPEATING TOKENS — it is the loop itself. The live /chat bug was temp 1.5 / top_k 5 / rep 0.80; with bounds (2) alone, a saturated accumulator still lands on rep 0.5. (1) keeps the real path out of reach, so the hazard is latent, not live — but any bypass of get_adjustment() (mock, refactor, alternate manager) restores it.

ALSO DUPLICATED CONCERN: (2)'s comment claims the bounds mirror GenerateRequest, but the schema bounds and the generation-safe bounds are different facts written twice at one seam. Decide ONE authority:
  a) raise _META_WEIGHT_BOUNDS rep floor 0.5 -> 1.0 and tighten temp/upper, then make (1) the only other clamp; or
  b) keep (2) purely as schema validation, rename to say so, and let (1) own generation safety — but then assert      the domain band is applied on every path (incl. cached nudges, which bypass get_adjustment()).

DECIDED 2026-11-01: ship the domain band only (472565281) and card this, to avoid rewriting another contributor's landed contract (test_meta_weights_clamp.py asserts rep 0.2 -> 0.5) in the same change. NOTE the nudge cache: _META_WEIGHT_CACHE stores nudges for 5s, and a cached nudge bypasses get_adjustment() entirely, so the band does NOT protect a cache hit if the nudge was computed pre-band or by an unbanded path.

Acceptance: one documented authority for generation-safe bounds; test_meta_weights_clamp.py updated to the chosen semantics; a test proving the band is applied on the cache-hit path too.
