---
id: 20260809_142710_world-realm-vectorized-diffusion-waves
title: World realm: vectorized diffusion + waves
status: done
tags: world-realm,simulation,performance
created: 2026-08-09T14:27:10.015090+00:00
---

World realm: vectorized diffusion + waves

Vectorized the loop-based physics in simulation.py (the perf bottleneck from the material-behaviors milestone).

cell_update_diffusion: 6-axis np.roll pass with symmetric pairwise exchange (add diff to self, subtract rolled-back diff from neighbor) - conserves energy/temperature totals exactly. Was triple-nested loop with double-visit in-place updates.
cell_update_waves: 6-direction pass; each cell emits signal*speed toward the rolled neighbor, transfer taken from the running field - non-negative, total conserved. Was cell-ordered loop with asymmetric emission.

Decision (user-approved): accept vectorized semantics. NOT bit-for-bit identical to old loops - diffusion now exchanges rate (0.1) per pair/tick vs old effective 2*rate*(1-rate)=0.18; waves now broadcast symmetrically (old was cell-order asymmetric). Vectorized matches the diffusion_rate knob and the 'spread outward' spec intent. Verified via reconstructed loop-reference comparison: both conserve energy/signal exactly, no negatives, but max|dE|~6.9 and max|dsig|~78 per tick on a tiny grid.

Verification:
- 512/512 tests pass (117 sim + 32 evolution + 22 memory + 275 shell_repl + 22 scene_persistence + 18 signal + 26 environment), exit 0, collection matches.
- py_compile clean.
- Perf on full 64x32x64 with terrain, seeded: ~0.037 s/tick vs ~1.8s/tick baseline (~48x faster); 4 babies alive after 5 ticks; grid energy 62.5K (terrain deposits now diffuse out instead of being converted to heat by concentrated embers).

pycache cleared.