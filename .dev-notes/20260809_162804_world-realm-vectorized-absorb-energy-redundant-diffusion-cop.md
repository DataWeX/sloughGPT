---
id: 20260809_162804_world-realm-vectorized-absorb-energy-redundant-diffusion-cop
title: World realm: vectorized absorb_energy + redundant diffusion copy removed
status: done
tags: world-realm,simulation,performance
created: 2026-08-09T16:28:04.540464+00:00
---

World realm: vectorized absorb_energy + redundant diffusion copy removed

Two bit-identical perf wins on the world-realm simulation.

1. absorb_energy vectorized (simulation.py): replaced the per-baby 5x5x5 triple loop (125 world.idx() calls) with a cached _cube_offsets cube (dx slowest, dz fastest, matching loop order). Gather material/energy once, mask ORGANIC, np.minimum(energy, 1.0) scatter subtract. Duplicate flat indices (grid dim < 2*radius+1 -> wrap aliasing) fall back to the exact sequential loop since the reference loop recomputes min per visit on the running value. Verified bit-identical over 2000 random cases (dims 2..64, radius 1/2, pos near/at wraps); accumulation dtype matches NEP50 float32 semantics. Micro: 493us -> 154us per baby (3.2x).

2. cell_update_diffusion: dropped the redundant trailing .astype(np.float32) on energy/temp - under NEP 50 the whole loop is already float32 (roll, sub, python-float rate multiply all preserve float32). reshape(-1) alone is bit-identical (verified) and saves one full-grid copy. Standalone 5.8ms -> ~4.2ms.

Both validated: 261 world tests pass (environment, simulation, signal, world_driver, scene_persistence, evolution, memory), exit 0, pycache cleared. Full-sim benchmark noisy due to concurrent load (median ~8.9ms/tick at 4 babies); diffusion remains the algorithmic bit-identical floor.