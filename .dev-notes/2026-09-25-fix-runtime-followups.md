---
id: 20260925_060_fix_runtime_followups
status: done
title: Fix all: npm ci on main, pre-existing red suites, typeof-import hook trap, start API server
tags: [followup, npm, test-debt, hooks, smoke]
---
# Fix all — runtime-error batch leftovers (card 97ad90c1)

## Scope (from board card 97ad90c1, follow-up to card 0e1fa738)
1. **npm ci on main** — strui + sdk-ts use npm-unsupported `workspace:*`; package-lock out of sync. WIP exists (dirty package.json ×2 + package-lock in fix-recovery, mtime 12:03).
2. **Pre-existing red suites** (verified identical on pristine-main baseline): backend: collections 22, companion 2, datasets 2, status-ready 1, training_finetuned 1, middleware extension-filter 1; frontend: page 14, datasets 7, token-tree 1, training/runs 2, settings 1-2, multimodal 1, vm 1-3, companion 2-5. WIP exists (middleware.py + companion/datasets/status tests dirty since 12:34).
3. **Pre-commit hook trap** — files containing `typeof import('...')` fail `consistent-type-imports`.
4. **API server** — start from main, smoke check.

## Plan
- Worktree `/home/mana/Documents/sloughGPT-fixtests`, branch `feat/runtime-followups`, carry over the 6h-idle WIP from fix-recovery.
- Review WIP diffs → complete items 1–2; then 3, then 4.
- Gates: affected suites green vs baseline, ruff, hook lint clean, API smoke curl OK.

<!-- claimed by opencode 2026-09-25T18:35:00+01:00 -->

## Result (2026-09-25) — items 3 + 4 done (this session)

**Item 3 — typeof-import hook trap: FIXED** (merge `b7f8b4b5f`, branch commits `6c613028b`)
- Root cause: shared `packages/eslint-config/index.js` had `@typescript-eslint/consistent-type-imports: 'error'` with default `disallowTypeAnnotations: true` → every `typeof import('...')` type query (11 files) errored `` `import()` type annotations are forbidden `` and lint-staged `eslint --fix` cannot fix it → any commit touching those files was blocked.
- Fix: `['error', { disallowTypeAnnotations: false }]`. `typeof import()` in tests is the standard lazy-module-typing idiom.
- Same commit also unblocked `apps/web/app/api/planner/helpers.test.ts`: replaced 6 `try { require('fs')... } catch {}` cleanup stubs with the file's existing static fs import + comment-bearing catches (`no-require-imports` + `no-empty` — the only remaining errors in the trap file set).
- Verified: eslint on all 11 trap files → 0 errors; `helpers.test.ts` vitest 24/24; pre-commit passes on .py commits (after `.venv -> primary/.venv` symlink, the worktree convention used by fix-recovery/journeys).

**Item 4 — API server start + smoke: DONE, found & fixed a real bug** (merge `b7f8b4b5f`, commit `878d776d3`)
- Started from main on **:8001** (8000 held by the parallel journeys server): `uvicorn apps.api.server.main:app`, detached, log `/tmp/api_smoke_8001b.log`. **Left running.**
- Smoke: `/health` 200, `/health/ready` 200, `/health/startup-progress` 200, `/openapi.json` 601 paths, `/models` 200, `/consciousness/health` 200 (degraded=disabled), `/infer/health` 200 (no_model).
- **Bug found**: `/health/ready`'s `app_lifecycle` was pinned at `starting` forever. The READY stage set `STARTUP_PHASE=running/ready (step 9/9)`, then BACKGROUND-stage hooks re-entered `_phase3_wandb`/`_phase4_multimodal` and unconditionally rewrote phase to `wandb_server`/`multimodal (step 6/9)` → `_is_app_ready()` false permanently. Fix: `_advance_startup_phase()` guard — no phase write once `running|ready`.
- Verified: restart → `phase: ready 9/9` (survives background), `app_lifecycle: ready`; `model: not_loaded` is honest (3× `model_loader: load failed` for Qwen/Qwen2.5-0.5B offline, 120s wait timeout) so overall status stays `not_ready` by design without weights.
- Tests: `test_readiness_gate.py` + `test_startup_routers.py` 54/54; `-k "health or ready or startup"` 165/165; ruff clean.

**Item 1 — npm ci**: DONE by parallel session (`c2c5e2d0f`, `workspace:*` → `file:` deps + lockfile regen). Verified: fresh worktree `npm ci` exit 0.

**Item 2 — red suites**: backend DONE by parallel session (`78c5af685`, 6 repairs). Frontend (page14/datasets7/token-tree/training-runs/settings/multimodal/vm/companion) IN PROGRESS by that session (dirty test files at merge time). This card stays `in-progress` until frontend suites land.

## Split (user-approved 18:40)
- Parallel session: frontend red suites (+ items 1–2 backend, already committed).
- This session: item 3 hook trap + item 4 API smoke — DONE.

## Result (2026-09-26) — ALL ITEMS VERIFIED, CARD CLOSED

Independent verification on `18af683af` (fresh `npm ci` in sloughGPT-fixtests):
1. **npm ci** — exit 0 (fresh worktree + post-lockfile-fix re-run);
   `react-router-dom` now in lockfile via `77b9f548a`.
2. **Red suites** — backend 6 repaired by parallel session (`78c5af685`);
   frontend all green (`77b9f548a` dep declarations + `18af683af` mock/suite
   repairs). **Full vitest: 832/832 files, 7981/7981 tests PASS**
   (one earlier `benchmark/page` failure was a full-run flake — 13/13 in
   isolation, absent from the confirming rerun).
3. **typeof-import hook trap** — FIXED by this session (`6c613028b`),
   merged `b7f8b4b5f`.
4. **API server** — running on :8001, smoke all 200s; readiness bug found &
   fixed (`878d776d3`, `app_lifecycle` reaches `ready`).

Final gates (fixtests @ `18af683af`): `tsc --noEmit` exit 0; vitest 832/832;
ruff clean; readiness/startup/health pytest 54+165 green.
Note: whole-tree eslint has 154 pre-existing errors in 129 files — **0 overlap**
with card commits (staged-file lint gate passes via lint-staged); tracked as
debt, not this card.

## Append (parallel session, final) — frontend confirm + flake harden
- Full vitest on fix-recovery `18af683af`: **831/832 files, 7980/7981 green**; sole failure `session/page > loads archived sessions` was a load-race (18/18 ×3 standalone). Hardened with `waitFor` on the render assert → `e8b5b2ea7`.
- Post-fix gates: `tsc --noEmit` 0; session suite 18/18 ×3; eslint 0 errors; backend pytest 1161/1161 (rerun at `78c5af685`+); `npm ci --dry-run` 0.
- Independent API smoke on :8013 (8001 held by other session): `/health` healthy, `/health/ready` `app_lifecycle: ready`, `/live`, `/status`, `/knowledge/stats`, `/chat/health`, `/health/detailed` (score 86), openapi 601 paths, 404 correct — then stopped.
