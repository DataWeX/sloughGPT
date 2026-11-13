---
id: 20260929_071_routers_doc_truthup
status: doing
title: "Card 071 — docs/routers.md truth-up + contract test (674 routes pinned)"
tags: [testing, docs, contract, api]
sprint: current
gh: null
---
# Card 071 — routers.md truth-up

## Problem

`docs/routers.md` had drifted badly from reality (same anti-drift pattern as
070's API.md):

- **674 routes served** (probe = startup route registration, verified vs live
  OpenAPI: 674/674, missing 0, extra 0)
- doc had **392 rows**, of which **41 were stale** (documented, not served) and
  **323 served routes were undocumented** (48% coverage)
- stale patterns: `/ratelimit/*` → `/rate-limit/*`, `/registry` → `/registry/models`,
  `/session/list` → `/chat/sessions*`, `/context/store-fact` → `/context/fact`,
  whole `/collections*` pipeline API, `/memory/get-config` → `/memory/config`

## Delivered (branch `feat/routers-doc-truthup`, worktree sloughgpt-routers-doc)

1. `docs/routers.md` — truthed up: 41 stale rows removed, 323 missing rows
   added (every new row's description taken from the handler's docstring via
   probe OpenAPI summary — zero TODOs), section header `/ratelimit` →
   `/rate-limit`, dead `Collections` section removed, 19 new router sections
   added (consciousness, mobile, workspaces, settings, tenants, users, tokens,
   dashboard, monitor, recovery-adjacent, etc.). Final: **674 rows == 674
   served, stale 0, missing 0**.
2. `tests/contract/test_routers_doc.py` (new, 2 tests): parses doc table rows,
   compares against probe-app route enumeration; asserts stale==0 and
   missing==0; anti-vacuous floor `len(live) >= 600`.

## Route discovery (the hard part — fastapi 0.139.2)

- `app.routes` pre-lifespan = only 4 routes (health/status/dashboard/consciousness);
  everything else registers inside lifespan via `startup.py`.
- `get_all_routers()` (51 routers) + `training.router` + main.py's 4 routers
  = source set, but `include_router` in fastapi >=0.139 is **lazy**: it stores
  `_IncludedRouter` wrappers without `.methods`/`.path`, so naive
  `route.path` iteration silently loses routes (first attempt: 550/674, then
  635/674 with training's lazy includes still missing).
- Fix: build a probe app, `include_router` everything, enumerate via
  `probe.openapi()["paths"]` (materializes lazily). Result: **674 == live 674**.
  Duplicate-operation-id UserWarnings suppressed around `openapi()`.

## Gates

- ruff: clean (check + format on own files)
- `pytest tests/contract/`: **23 passed in 8.3s** — 070 envelope 5 + router_boundary 4 +
  routers_doc 2 + scatter_gate 5 + web_api_paths 2 + testing_doc 5 (grew with each seam)

## Blockers

- Same as 067: `main` held by parked-dirty api-std worktree (29 staged files,
  concurrent session). Merge queued behind `feat/journey-library` (auto-watcher
  `~/opencode-logs/merge-watcher.sh` polls and ff-merges 067 first; 071 merges
  after, needs `git merge main` sync first since it branched from `74f392f5e`).

## Files

- `docs/routers.md` (+403/-52)
- `tests/contract/test_routers_doc.py` (new, 106 lines)
- generator: `~/opencode-logs/truthup_routers_doc.py` (one-off tooling)

## Seam 3: docs/TESTING.md truth-up (commit 7e69d8366)

Audit found stale/false claims; fixed in same change + guard test:

- `-n auto` in all backend commands **fails locally** (pytest-xdist is CI-only;
  verified `unrecognized arguments: -n`) — removed, documented CI-only status.
- Example file `test_inference.py` under packages/core-py/tests/ does not
  exist (it lives in root tests/); dead `brainstorm` tree example → real
  `consciousness/help` pair.
- Stale self-reported coverage table (100%/95% claims) → replaced with
  generate-it commands (`npm test -- --coverage`, `pytest --cov=domain`).
- "Pre-commit Hooks" section showed a manual command only → real hooks:
  .pre-commit-config.yaml (tsc, py-compile, no-commit-to-branch main, …),
  .husky pre-commit (npx lint-staged: eslint/prettier/ruff), pre-push advisory.
- CI section: named real commands (ci.yml `npm run test:lib`, ci_cd.yml
  `npx vitest run`, reusable-ci-core pytest subset + xdist install).
- Added: root tests/ suite + pointer to tests/README.md (no duplication),
  PYTHONPATH recipe for server/contract, both pytest.ini addopts+markers,
  venv requirement, contract suite count (23).

Guard: `tests/contract/test_testing_doc.py` (5 tests) — referenced paths
exist, no stale claims (`-n auto`, 100% table, brainstorm), pytest.ini
markers documented, npm scripts real, doc's contract count == live
collection (subprocess). Mutation-verified: re-adding `-n auto` fails
test_no_stale_claims.

### Files (seam 3)
- `docs/TESTING.md` (+59/-40)
- `tests/contract/test_testing_doc.py` (new, 5 tests)
- format-only on test_routers_doc.py / test_web_api_paths.py (lint-staged parity)

### Report log
- 05:27 — contract suite 23/23 green; commit 7e69d8366; merge-watcher +
  post-merge-verify relaunched with 24h windows (pids 132093/132116, pidfiles
  in ~/opencode-logs/); api-std main advanced to f1ecf8a6f (other session
  commits directly to main while still carrying 29 staged files); 071 branch
  needs `git merge main` sync at merge time.
