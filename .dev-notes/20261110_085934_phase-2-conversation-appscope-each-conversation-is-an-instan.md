---
id: 20261110_085934_phase-2-conversation-appscope-each-conversation-is-an-instan
status: done
board: 20261110_085934_phase-2-conversation-appscope-each-conversation-is-an-instan
title: "Phase 2: conversation AppScope — each conversation is an instance of the app (mini-app)"
tags: [chat, app-scope, contract, frontend, backend]
sprint: ""
gh: null
---

# Phase 2: conversation AppScope — each conversation is an instance of the app

Board card `20261110_085934` (description = the user-confirmed design of
2026-11-10). Branch `feat/app-scope`, landed to main 2026-11-12.

## What landed

Push `e828d7728..754ec2388` (3 commits, 12:20):

- `b2d016251 feat(app-scope): Phase 2 — each conversation is an instance of the app`
  - `resolve(global, session, conversation)` pure function, first-set-wins
    precedence (conversation → session → global), invoked ONCE at the turn
    seam (`stream_turn`), passed as explicit value on `TurnRequest`.
  - Frozen `AppScope` value object (model settings, system prompt, tools,
    memory/knowledge, summary); delta persists on the session record as an
    extra field (backfill-free); exposed via descriptor on the grandfathered
    `PUT /chat/sessions/{id}` — no new route, baseline unchanged.
  - TS half: AppScope Pydantic model → `gen_ts_contracts` → frontend derives
    type (`conversation-scope-store.ts`, `useChatModelSettings` rework).
  - UI: inline settings widget composed on the existing chat page (surface
    ladder — no new page).
- `ad00d32ff docs(agent-sync): session-contract lane entry`
- `754ec2388 chore(domain): rehash domain-baseline — inference gains AppScope`
  (rehash during rebase conflict resolution: `inference 032874e0 → 821f2f96`).

## Gates (run on the rebased pre-push stack)

- app-scope tests: 35 passed (17 `tests/core-py/test_app_scope.py` + 18
  `apps/api/server/tests/test_app_scope.py`)
- contract static 615/48 ✓, manifest --check ✓, ts contracts --check ✓,
  in-process contract suite 50 passed
- domain baseline 8 passed (rehash --check: no drift), feature tags ✓,
  board chain 0
- frontend: tsc 0, eslint 0, vitest 48 files / 632 tests passed
  (`npx vitest run --changed origin/main` from `apps/web`)
- CI `43217020818`: completed **success** (Mobile CI `43217020528` success)

## Notes / incidents

- Rebase onto fresh main hit `domain/domain-baseline.json` conflict → took
  main's bytes + `domain_baseline_rehash.py` (documented precedent).
- `test_np_in_scope` (`KeyError: '__import__'`, `vm_devices._py_eval`) fails
  isolated AND on pristine main — pre-existing, NOT Phase 2. Surfaced for
  carding.
- Runtime-config worktree corruption (25 files truncated to 0 bytes by an
  interrupted rebase at 13:26, not this lane) restored via
  `git restore --worktree .` before card 2's landing.

## Not touched

Auth/transport/soul/planner (stay global); resolution is one-way
global→conversation (never writes back); two-tool-tier boundary respected
(model-facing only); `/benchmark` page, `docs/INDEX.md`,
`tests/core-py/test_results/*.json`.
