---
id: 20261112_064807_featmonitoring-gen-run-history-local-store-recorder-get-gen
status: done
board: 20261112_064807_featmonitoring-gen-run-history-local-store-recorder-get-gen
title: "feat(monitoring): Gen run history — local store + recorder + GET /gen-runs + GenRunsCard (gate health, regression attribution)"
tags: ["monitoring", "contract", "tooling", "cli", "frontend"]
sprint: ""
gh: null
---

# feat(monitoring): Gen run history — local store + recorder + GET /gen-runs + GenRunsCard (gate health, regression attribution)

User ruling 2026-11-12: log Gen's (coordinator agent) work history and score it against previous runs. Metrics v1 = **gate health** + **regression attribution**. **No gh / GitHub IDs anywhere** — attribution is fully local: board card transitions (reopened cards), `tests/core-py/test_results/*.json`, and gate exit codes captured at event time. Website receives stats **snapshot-first** (poll `GET /gen-runs`); no SSE/`EVENT_SCHEMA` tier in v1.

Why: Gen's history today is prose (`docs/AGENT_SYNC.md`) written at session end, so a session that dies mid-flight records nothing and `compare` has nothing to delta. The linking seam already exists — `.opencode/plugins/ui-error-watcher.ts` already fetches the app API (`SLO_API_URL ?? http://localhost:8000`) and `session-tracker.ts` already hooks `tool.execute.after`. Port-don't-rebuild: `scripts/benchmark_results.py` already owns the canonical-store + `record/history/compare` pattern (`SLO_BENCH_DATA_DIR` → `~/.local/share/sloughgpt/benchmark_results`, card 20261031_133226) — reuse its store shape, never a repo-relative path.

## Plan (owned paths only)

1. `scripts/gen_runs.py` (new) — `record | history | compare --vs previous|first`. Canonical store `SLO_GEN_DATA_DIR` → `~/.local/share/sloughgpt/gen_runs/` (JSONL, append-only, `source: local`). Per run: run_id (worktree+branch+sha), cards moved todo→done, cards reopened, gate lines with exit codes, red→green delta from `tests/core-py/test_results/*.json`, wall-clock. All derived from artifacts — never a number the agent types in. Idempotent on run_id (dedupe, not double-append).
2. `.opencode/plugins/gen-run-recorder.ts` (new) — hooks `tool.execute.after` for shell gate commands + card closes; appends the local JSONL line **first** (no network — `:8000` is often down mid-session).
3. `apps/api/server/routers/gen_runs.py` (new) — `ToolSpec gen_runs.list`, idempotent → `GET /gen-runs`, `auth_scope="authenticated"`, `version="1"`, tier (b) external utility tooling (operator surface, never model-facing). Missing/corrupt store → `[]`, never a 500. Then `scripts/gen_router_manifest.py` + `scripts/gen_ts_contracts.py`.
4. `apps/web/components/monitoring/GenRunsCard.tsx` (new) + 4 lines in `apps/web/app/(app)/monitoring/page.tsx` — self-fetch via `apiGet('/gen-runs')` (http-client rule), summarize-don't-dump: per-run summary row (date, gate pass ratio, reds closed vs reds introduced, cards done), raw records one click away in a `FoldSection`. `role: 'operator'`. No new page (surface ladder).
5. Tests: `apps/api/server/tests/test_gen_runs_router.py`, `scripts/tests/test_gen_runs.py`, `GenRunsCard.test.tsx`.

## Collisions / must-not-touch

- `apps/web/app/(app)/monitoring/page.tsx` is ALSO touched by in-flight card `20261111_110106` (branch `feat/test-runs`, `TestRunsCard`) — expect a trivial adjacent-import conflict on merge.
- Never: board schema/chain seal, `docs/AGENT_SYNC.md` prose (ride an entry only at close), chat/agent tool registry or `domain/tools`, a new page, raw `fetch()` in web, `tests/core-py/test_results/*.json` (read-only), gh/CI API calls.

## Definition of done

`./scripts/python scripts/gen_runs.py history` prints ≥1 recorded run; `GET /gen-runs` returns the summary envelope; `GenRunsCard` renders it on `/monitoring`. Gates: pytest (new suites), `gen_router_manifest --check`, `gen_ts_contracts --check`, `check_contract --static`, `ruff check` + `format --check`, `tsc --noEmit`, eslint 0 errors, vitest (new card tests) — run with the coordination-workflow verification incantation.
