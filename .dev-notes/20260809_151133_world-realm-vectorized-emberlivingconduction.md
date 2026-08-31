---
id: 20260809_151133_world-realm-vectorized-emberlivingconduction
title: World realm: vectorized ember/living/conduction
status: done
tags: world-realm,simulation,performance
created: 2026-08-09T15:11:33.678050+00:00
---

World realm: vectorized ember/living/conduction

Vectorized the last loop-based material physics in simulation.py (cell_update_ember, cell_update_living, cell_update_conduction), replacing per-cell Python loops with numpy batching/broadcasts. Fixed three ember neighbor-offset off-by-ones, a living indexes->indexer bug, and a conduction energy[idx] += idx bug; removed stale grid.energy astype cast.

Semantics preserved (spec: snapshot-burn, first-free-neighbor ascending, symmetric pairwise diffusion):
- ember: burns from a per-tick fuel snapshot (fuel = energy[ember_ids]); exhausted -> stone; radiated energy half conserved via symmetric neighbor split, heat half leaves pool; burn_temp keep rule intact.
- living: growth claims air neighbors in ascending flat-index first-free order; 'claimed' mask resolves cross-offset collisions deterministically (no RNG); growth cost + transfer_fraction honored.
- conduction: symmetric pairwise diffusion among metal cells only (indexer double-strided shape fix).

Tests: +3 in tests/test_environment.py (ember adjacent-ember own-fuel burn 50+25/6, living later-neighbor fallback, living shared-target claimed-once). First attempt had a wrong ember expectation (asserted 50.0; correct is 50 + 25/6 from neighbor radiation) -- fixed assertion, implementation was correct. Fixed one broadcast bug in ember (boolean mask & length-2 array) by switching to ember_ids indexing.

Verified: full scoped suite 536/536 pass (512 prior scope + 21 driver + 3 new), exit 0, pycache cleared. Journal note + board sync pending.