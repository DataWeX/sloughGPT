---
id: 20260809_141046_world-realm-material-behaviors-terrain-generation
title: World realm: material behaviors + terrain generation
status: done
tags: world-realm,materials,terrain
created: 2026-08-09T14:10:46.646901+00:00
---

World realm: material behaviors + terrain generation

Environment build: material behaviors + terrain generation.

Material behaviors (all computed every tick, animated by energy/temperature/signal, knobs in WorldParams):
- ORGANIC: rots (organic_metabolism), ignites into EMBER above ignition_temp, feeds babies (existing absorb_energy)
- EMBER: burns fuel, radiates energy to neighbors + heat as temperature, sustains burn_temp, exhausted -> stone
- LIVING: spends energy to turn adjacent air into ORGANIC (deterministic, no RNG)
- WATER: damps signal (water_signal_dampen), relaxes temperature (water_cool_rate)
- METAL: extra pairwise energy diffusion (metal_conduction_boost)
- Temperature: whole world relaxes toward ambient_temp (ambient_cooling)

Wiring: new cell_update_{combustion,metabolism,ember,living,water,conduction,temperature,materials} in simulation.py; cell_update_default now runs diffusion -> waves -> materials -> temperature -> conservation.

Terrain: generate_world(grid, params, seed) uses a LOCAL np.random.default_rng(seed) — deterministic on (grid_size, world_seed), never the global stream, so snapshot restore stays RNG-neutral. Layout: stone floor (y=0), water pools + organic patches (y=1), buried ember vents. Opt-in via WorldParams.generate_world=True (default False keeps all existing tests green). SimScene.spawn_babies now drops babies on the ground surface in generated worlds.

Tests: tests/test_environment.py — 26 tests (combustion/metabolism/ember/living/water/metal/temperature + terrain determinism/RNG-neutrality + scene integration incl. resume-matches-continuous with terrain + babies-survive-longer-with-food). Full scope 486 existing + 26 new = 512 pass (exit 0). py_compile clean, pycache cleared.

Docs: WORLD_REALM.md materials table now marks each behavior Computed; params table updated; Built-and-Working gained Material Behaviors + Terrain Generation rows and a 26-test row; Energy Economy section documents material sinks.

Verified on full 64x32x64 with terrain: 4 babies alive, terrain energy ~18.6K over 5 ticks (~1.8s/tick — dominated by the pre-existing loop-based diffusion; vectorization is the tracked follow-up).