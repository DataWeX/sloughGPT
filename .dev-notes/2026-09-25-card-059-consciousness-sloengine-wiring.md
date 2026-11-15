# Card 059 — Build Order #6: wire consciousness at both inference choke points

- **Status**: done
- **Merged**: `5c788ca4d` on main (commits `294dc86de` + `188d7b16e`, branch `feat/consciousness-sloengine-wiring`)
- **Date**: 2026-09-25

## What shipped

**SloEngine choke point** (`domain/core/_internal/soul.py`)
- `_init_consciousness()` holds the process-wide `get_consciousness()` engine (best-effort; failure = hook off).
- Level gate read **at call time** — runtime `ConsciousnessConfig.level` changes arm/disarm without re-init.
- `generate()` calls `_post_consciousness()` on both the cache-hit and fresh paths; exactly one `process()` + `save()` per generation.
- Skips empty/placeholder (`[Error`, `[Slo`) responses; any failure is swallowed (generation never breaks).
- Narrative surfaced in `reasoning_chain` (`consciousness: level=N narrative=...`) and `_cognitive_state.last_narrative`.

**Router chat path** (`apps/api/server/routers/inference.py`)
- New `_consciousness_post_process(user_msg, full_response) -> {"narrative", "level"}` (level-gated, best-effort).
- `_run_post_gen_tasks` returns that result; chat stream yields `CONSCIOUSNESS complete` SSE with `level` + `self_insight` (frontend already parses `self_insight`).
- Pre-gen qualia SSE ping unchanged; the two paths are disjoint (chat goes through `provider.chat_stream`, not `SloEngine.generate`) so there is never a double `process()`.

**Why not remove the router block**: recon proved the server chat path never calls `SloEngine.generate` (providers do their own generation), so the router block IS the chat path's single post-gen owner — dedupe-by-deletion would have regressed chat.

**Docs (part A)**
- ROADMAP goal 9 rewritten to describe the real two-choke-point wiring (was falsely claiming a nonexistent mechanism).
- PRODUCT row 15 "Built, not wired" → "Wired"; debt row updated; Build Order #6 marked done.

**Tests**: `tests/test_sloengine_consciousness.py` (level-0 off, runtime re-arm, single-process-per-generate, placeholder skip, exception safety, cache-hit) + `apps/api/server/tests/test_consciousness_post_process.py` (helper gate/narrative/failure + `_run_post_gen_tasks` contract with hermetic stubs — real `get_learner()` pollutes later stream tests).

**Benchmark**: `scripts/benchmark_consciousness_hook.py` (init / generate / direct hook cost). Results: generate ~2.21s main vs 2.22-2.52s branch (HD-memory quantum search dominates, noise-level), off-gate hook 4.6 us/generate (negligible vs 2.2s), level-1 process+save 28.8 ms (opt-in only).

## Incidents / notes
- **Stale merge concluded**: fix-recovery had an unconcluded merge of `fix/runtime-errors` (started 07:22 by a crashed session, MERGE_HEAD stale 2.75h) blocking main. Concluded it as `7c1f5c0eb` (their own branch, pure auto-merge; their unstaged WIP left untouched). One eslint error in merge output fixed as part of resolution: `db.test.ts` inline `typeof import('./db')` to `import type * as DbModule`.
- **`.venv` symlinked** into fix-recovery (worktrees lack it; lint-staged ruff task ENOENT was killing all hook tasks).
- Parallel/sibling session co-authored docs + parts of the implementation in this worktree (call-time gate wording, "option 2 idempotent re-experience" note); content merged into the same commits.
- Pre-existing (not this card): apps suite failures (collections/companion/datasets/knowledge/middleware/status/training_finetuned — the other agent's active lanes); cross-suite consciousness stream pollution when root tests run before apps tests (reproduced on clean main).
