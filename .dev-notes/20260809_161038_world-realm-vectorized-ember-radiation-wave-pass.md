---
id: 20260809_161038_world-realm-vectorized-ember-radiation-wave-pass
title: World realm: vectorized ember radiation + wave pass
status: done
tags: world-realm,simulation,performance
created: 2026-08-09T16:10:38.286117+00:00
---

World realm: vectorized ember radiation + wave pass

Perf session on the world realm grid.

1. Ember radiation scatter (shipped): replaced 12 full-grid np.roll passes per tick in cell_update_ember with targeted per-neighbor flat-index adds (modulo wrap). Emitters are ember_ids; receivers are ember_ids + per-axis stride offsets. Verified bit-identical energy/temperature vs the roll formulation; energy conservation exact. Full-sim benchmark (4 babies, 64x32x64): 23.7 ms/tick -> 10.2 ms/tick (2.3x).

2. Diffusion roll audit (no change): tried fusing the two-step energy += diff; energy -= roll(diff,-shift) into one Laplacian form. Two variants measured: (a) 2-roll e_plus/e_minus -> bit-identical but SLOWER (10.3ms isolated vs 3.9ms original, extra temporaries); (b) fused Laplacian + float32 rate -> ~1ms faster isolated but only 4.6e-5 FP deviation from original, which breaks scene-replay determinism for saved grids. Reverted both; diffusion stays as the 24-roll/tick baseline (algorithmic floor for the symmetric pairwise-exchange stencil).

3. Wave pass micro-opt (shipped): removed the np.where(signal > 0, ...) guard in cell_update_waves since signal is invariant non-negative; transfer = signal * speed is bit-identical (verified) and drops one full-size array alloc per pass. Isolated benchmark 4.1ms -> 2.8ms (1.4x); full-sim effect masked by noise.

Result: 261 simulation/world tests pass (env, simulation, signal, world_driver, scene_persistence, evolution, memory), exit 0. Pycache cleared.