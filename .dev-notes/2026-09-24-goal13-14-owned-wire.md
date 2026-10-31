---
id: 20260924_035_goal13-14-owned-wire
status: done
title: Wire OwnedObjective + ExperienceSampler into one loop
tags: roadmap,training,owned-model
---
# Goals 13+14 — owned objective + owned data wire-up

## Plan
1. Branch feat/goal13-14-owned-wire
2. TrainerConfig: `owned_objective: bool = False` (or weights)
3. SloughGPTTrainer: construct OwnedObjective when enabled; apply in train_step after model(x,y) with batch metadata when available
4. data_path `"experience"` (or `experience://`) → ExperienceSampler as train split source
5. Tests: objective changes loss/metrics when metadata present; experience sampler loads corpus/feedback; trainer trains with data_path=experience
6. ROADMAP 13/14 Done if acceptance met; no fourth loop (scatter gate)

## State
- DONE on `feat/goal13-14-owned-wire-v2` (clean worktree `/tmp/opencode/goal1314` off main; primary tree blocked by foreign dirty files)
- Goal 13: `TrainerConfig.use_owned_objective` / `owned_tool_weight` / `owned_memory_weight` + legacy kwargs; `OwnedObjective` applied in `SloughGPTTrainer.train_step` after EWC, before the single `(loss * scale).backward()`. Metrics expose `owned_*` breakdown. No metadata → pure next-token identity.
- Goal 14: `data_path="experience"` | `experience:<corpus>` loads chat+feedback via `load_experience_text` in `prepare_data`; `ExperienceSampler` is the train-batch source; synthetic seed corpus when empty. `_owned_metadata` set with `memory_match` when the experience sampler is active.
- Tests: `tests/test_owned_objective_experience.py` (9) + smoke/long-run/convergence/EWC/continual/scatter/router contracts — 51 passed; ruff clean.
- ROADMAP 12–14 section: goals 13/14 struck Done (12 still open).
