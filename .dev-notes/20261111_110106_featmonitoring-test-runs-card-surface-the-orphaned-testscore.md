---
id: 20261111_110106_featmonitoring-test-runs-card-surface-the-orphaned-testscore
status: wip
board: 20261111_110106_featmonitoring-test-runs-card-surface-the-orphaned-testscore
title: "feat(monitoring): test-runs card — surface the orphaned tests/core-py/test_results/*.json (zero consumers today)"
tags: [monitoring, contract, test-results, frontend, backend]
sprint: ""
gh: null
---

# feat(monitoring): test-runs card — surface the orphaned test-run artifacts

User ruling 2026-11-11 ("visualize these things... the test results/findings and those libraries we use" → answered "1 and 5" = both subjects). Card description is the design charter: inline `TestRunsCard` composed on `/monitoring` (surface ladder — no new page), ONE descriptor contract `test_results.list` (tier (b) external utility tooling, operator surface, never model-facing) via `create_router`, summarize-don't-dump with raw arrays one click away.

## Work (branch `feat/test-runs`, from origin/main `71adebbe4`, worktree `fe08f8/test-runs-card`)

TDD: RED first both sides.

- Backend: `apps/api/server/routers/test_results.py` — `ToolSpec test_results.list` (idempotent → GET `/test-results`, `auth_scope="authenticated"`, `version="1"`), `load_results()` summarizes every `*.json` under `tests/core-py/test_results/` (read-only; corrupt file dropped whole with warning — fail-safe; absent dir → `[]`), rows `{file, status, total, passed, failed, mtime, entries}` freshest-first. Tests: `apps/api/server/tests/test_test_results_router.py` (4).
- Projections regenerated: `gen_router_manifest` (44 modules, `test_results` appended), `gen_ts_contracts` (3 artifacts × +27 lines).
- Frontend: `apps/web/components/monitoring/TestRunsCard.tsx` — self-fetches via `apiGet('/test-results')` (http-client rule), per-file `FoldSection` where the summary row (name + status badge + `passed/total` + `timeAgo`) is the `<summary>` and raw entries are the collapsed body (native details = collapsed by default). Composed onto `monitoring/page.tsx` behind the `loaded` gate (+4 lines). Tests: `TestRunsCard.test.tsx` (6, incl. the collapse-by-default click assertion — jsdom details toggle probed first).

## Gates

- pytest `test_test_results_router.py` 4/4; feature-tags validate; `gen_router_manifest --check`; `gen_ts_contracts --check`; `check_contract --static` (615/48 baseline untouched — descriptor route is not hand-built); `test_ts_contracts` freshness 12 passed; ruff check+format clean.
- vitest `TestRunsCard.test.tsx` 6/6 + `monitoring/page.test.tsx` 29/29 (unmodified — real apiGet fails fast in jsdom); `tsc --noEmit` clean; eslint 0 errors (1 pre-existing warning at page.tsx:201, untouched).
- change-scoped `run_affected.sh` (testmon DB snapshotted from the main checkout via sqlite backup — fresh-worktree full-run avoided): 355 selected.

## Not touched

`/benchmark` page/cards, `docs/INDEX.md`, `tests/core-py/test_results/*.json` (read-only data), notification surfaces, avion.
