# Goal 16: land tiny convergence test on main

**Status:** done
**Sprint:** 2026-09
**gh:** —

## Task
Land `tests/test_training_convergence.py` (goal 16 release gate) on main.
Close goal 15 status-transition gap.

## Result
- Branch: `feat/goal16-convergence-test`
- `tests/test_training_convergence.py` landed (gate PASS: final 1.33e-05 < 0.01 in 120 steps, seed 0, e64/1L)
- `tests/test_training_status_transitions.py` (goal 15): consciousness idle→run→terminal, self-train not_started→running→exited, controller idle→running→completed, job running→completed/failed
- Fixed `scripts/benchmark_slonet_training.py` data_path leak (gate corpus was overwriting later configs)
- Recorded via `benchmark_results.py record --kind training`
- ROADMAP goals 15 + 16 marked Done
- ruff clean; 9+ related tests green

## MERGED (2026-09-24)

- Branch `feat/goal16-convergence-test` merged to `main` with `--no-ff` (`ada81b83`).
- Post-merge: 9 related tests green; ruff clean; open kanban cards 0.
- Note: merge also carried parallel-session commit `2d850085` (trait-weights MogDB isolation + feedback router test) that landed on the same branch while we worked — not authored by us, left in place.
- main is ahead of origin/main by 288 (not pushed — do not push unless asked).
