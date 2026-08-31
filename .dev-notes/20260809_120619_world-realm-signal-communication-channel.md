---
id: 20260809_120619_world-realm-signal-communication-channel
title: World realm: signal communication channel
status: done
tags: world-realm,communication
created: 2026-08-09T12:06:19.735276+00:00
---

World realm: signal communication channel

Signal communication channel is now first-class in the world realm.

Emit: write_cell/place_material with MATERIAL_SIGNAL transfer the deposited energy into grid.signal at that cell (a transfer, not creation — energy accounting stays conservation-safe). Out-of-bounds writes rejected.

Waves: cell_update_waves already propagated grid.signal; verified it now carries written broadcasts to neighbors each tick (wave_speed x (1 - signal_decay)).

Perceive: get_nearby_cells now returns the signal array (empty read too); _perception_features exposes it as a learnable 5th cells feature = min(mean(signal), 1.0); WorldParams.cells_input_dim bumped 4 -> 5, so all Perceptron/Genome/EpisodicMemory archetypes auto-resized via params (evolution tests read shapes only from params.*_input_dim — no breakage).

Learning: the signal row of the cells perceptron is reinforced when signal presence correlates with reward; untouched when no signal present (verified).

Tests: 18 new in tests/test_signal_communication.py (emission, perception, propagation, learnable row, end-to-end B-perceives-A loop, snapshot survival, full tick loop). Scope set (simulation+evolution+memory+shell_repl+scene_persistence+signal) 486/486 pass, exit 0. py_compile clean, pycache cleared.

Docs: docs/WORLD_REALM.md updated — Communication section now documents the live channel (emission/waves/perception/learning/persistence table), emergent behavior #3 marked live, Built-and-Working table gains the Signal Communication row + 18-test row.

Next: none blocked. Channel is ready for the emergent-communication milestone experiments.