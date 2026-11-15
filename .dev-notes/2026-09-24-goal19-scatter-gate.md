# Goal 19: scatter gate

**Status:** done
**Sprint:** 2026-09
**gh:** —

## Task
ROADMAP goal 19 — new training surfaces (pages, routers, adapters) rejected unless they feed the one loop. Adapters only; a fourth loop fails CI.

## Result
- Branch: `feat/goal19-scatter-gate`
- `tests/contract/test_scatter_gate.py` — pair-signal detector (`loss.backward`/`.backward()` + `optimizer.step`/`clip_gradients`) over `domain/**` + `packages/core-py/domains/**`
- 19-file shrink-only allowlist pinned (multimodal/shell/feedback/known trainers as-is)
- Routers: no loop bodies; no `training_handler`/`train_pipeline`/`SloughGPTTrainer` imports; new `*train*`/`*lora*` routers must declare `domain.training` or `domain.cognition` facade (allowlist: self_train subprocess, consciousness adapter, lora_eval, user_adapters, feedback)
- Negative canary file verified to fail the gate, then removed
- Contract + status + convergence: 17 tests green; ruff clean
- ROADMAP goal 19 marked Done; kanban `20260924_032` → done

## MERGED (2026-09-24)

- Branch `feat/goal19-scatter-gate` merged to `main` with `--no-ff`.
- main is not pushed (do not push unless asked).
