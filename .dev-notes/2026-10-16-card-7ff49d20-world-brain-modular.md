---
id: 7ff49d20
title: "World-Model Bridge follow-up: modular/composable native SloNet model for the world"
status: done
board_id: Main
date: 2026-10-16
branch: feat/world-brain-modular
tags: [world, model-bridge, slonet, modular, brain]
---

# Card 7ff49d20 — modular/composable native SloNet world brain

## Ruling (user 2026-10-16)

Keep the LANDED ModelBrain (6610c59db, in main) as default; build OUR model
from the SloNet core made MORE MODULAR AND COMPOSABLE so it fits the world
server/world and the model stack. Four deliverables:

1. **SloNet core modular/composable**: a world brain built from our native
   stack (`domain/training/_internal/slonet.py` — SloLinear/backward/SloAdam)
   configurable/serialized/composed behind the brain seam
   (`WorldParams.brain_kind`, `world_driver --brain`), trained in-world.
2. **PORT from stranded 64a3503af (port, DO NOT MERGE; quote sha)**:
   (a) `decide_every` action-reuse window + staleness-by-design (update()
   must NOT invalidate the cache mid-window);
   (b) safe 0.5 fallback when the brain raises inside a tick;
   (c) `BRAIN_KINDS` registry + `RandomBrain` (replaces `if kind == "model"`
   forks);
   (d) `brain_from_dict` fail-loud on unknown kind;
   (e) benchmark rigor: multi-seed averaging + comparative gates.
3. **MISSING IN BOTH (add)**: trained-vs-untrained mean-energy gate — the
   only gate that proves learning HELPS (landed gates prove plumbing only).
4. **Fix-if-touched (grandfathered)**: `--brain` must pass
   `brain_hidden_units` (today `main()` never passes it -> H=None -> silently
   Perceptron); benchmark warmup exclusion + repeats; route message brain
   through `make_brain`.

## Work log

- 2026-10-16: note created; worktree `.wt-wbrain` branch
  `feat/world-brain-modular` from integration-sync `d195b5956` (freshest
  complete lane). NOTE: `6610c59db` is on main but NOT on d195b5956 — the
  world-file lane diff IS exactly the landing, so cherry-pick expected clean.
- Base lane has the single-test-root reconciliation: tests live in
  `tests/core-py/` (packages/core-py/tests/ does not exist on this lane) —
  the landed test file lands at `tests/core-py/test_model_brain.py` there.
- 2026-10-16 (continued): Stage A cherry-picked as `aa4caa8e8` (adapts test
  location + skips the absent `domain/shell/__init__.py` export). WIP for
  this card found in the worktree and completed:
  1. **Registry/composition**: `BRAIN_KINDS` + `make_brain` + `brain_from_dict`
     live in `model_brain.py` (fail-loud on unknown kind; legacy +
     explicit-tag perceptron payloads load); `simulation.py` drops its two
     `if kind == "model"` forks; message brain routed through `make_brain`.
  2. **`SlonetBrain`** — native `SloLinear` stack + native `SloAdam`, whole
     stack trainable (stranded froze it — defect left behind), configurable
     (`model_n_embed`/`model_n_layer`/`brain_decide_every`), serialized whole,
     `update()` hot path (pure-numpy backward) + `update_native()` autograd
     reference (parity-tested, multi-step exact); `_weight_T` cache cleared
     up front (SloLinear's cached transpose is not invalidated in-place).
  3. **Ports from 64a3503af**: `decide_every` reuse window + staleness-by-
     design (on SlonetBrain AND landed ModelBrain, inert at default 1),
     0.5 fallback on brain-raise, `RandomBrain`, fail-loud `brain_from_dict`.
  4. **Fixes**: `--brain`/`--hidden-units` now reach `WorldParams`
     (`brain_hidden_units` — the flag lied before); random-kind
     `perceptron_move` roundtrip no longer reverts to Perceptron.
  5. **Benchmark**: multi-seed + warmup-excluded + three comparative gates
     (value trained-vs-untrained mean energy, floor vs random, perf ≥80%).
     Config finding: 32/2 fails value AND perf; **16/1 @ decide_every=8
     passes all three** (+34.3 / +10.5 / 0.87) → shipped as defaults.
  6. Tests: 48 in `tests/core-py/test_model_brain.py` (22 locked originals
     unchanged + 26 new); `docs/WORLD_REALM.md` follow-up section added.


## Gates (2026-10-16, worktree .wt-wbrain)

| Gate | Command | Result |
| --- | --- | --- |
| Unit | `pytest tests/core-py/test_model_brain.py -q -p no:randomly` | **48 passed** (22 locked + 26 new), 14.9s |
| Blast radius | same incantation over evolution/environment/civilization/memory/predation/benchmark_seasons + world_driver/simulation/world_bridge | **416 passed**, 142s, exit 0 |
| Lint | `ruff check` + `ruff format --check` on domain/shell/_internal, scripts/benchmark_world_brain.py, the test file | clean (97 files formatted) |
| Benchmark | `python scripts/benchmark_world_brain.py` (3 seeds × 60 ticks + 5 warmup, load avg ~8.8) | **PASS exit 0** — value +34.25 (53.60 vs 19.35), floor +10.50 (53.60 vs 43.09), perf 78.8 vs 84.1 tps (0.94 ≥ 0.80) |

Commits on `feat/world-brain-modular` (base d195b5956 + Stage A cherry-pick aa4caa8e8):
`7ee09bea9` registry+SlonetBrain (ports quoted 64a3503af) ·
`1b31d35b8` benchmark rigor + CLI fixes (port quoted 64a3503af) ·
`b6744416f` 26 new tests · `f2c046058` WORLD_REALM.md section.

## Status: in_review (work complete; foreman verifies and closes)
