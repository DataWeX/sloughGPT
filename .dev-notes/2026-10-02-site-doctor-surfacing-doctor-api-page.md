---
id: 20261002_site_doctor_surfacing
title: "Site Doctor surfacing: /doctor API + /doctor page — show doctor across web UI (Phase A-UI)"
status: done
date: 2026-10-02
tags: [doctor, api, web, ui, observability]
board_id: 6d0ebf37-6287-45b8-8bce-c1f1b8a36798
board: 6d0ebf37-6287-45b8-8bce-c1f1b8a36798
hash: 11bda4a61b3bf2c5b553b23067b1661788d00ed51b709a32683d83c3ac2aeb63
---

# Site Doctor surfacing: /doctor API + /doctor page — show doctor across web UI (Phase A-UI)

## WHAT

User directive 2026-10-02: "build it with good structure and make the web
components and site UI, core infra, backend, inference setup and engines all
wired well so it [the doctor] can be seen" — surface the site doctor
full-stack so its findings + component health are visible in the app.

## SCOPE

1. **Backend**: new `apps/api/server/routers/doctor.py` registered in
   routers/__init__.py —
   - GET /doctor/report: serve last DoctorReport from SLO_DOCTOR_REPORT path,
     null-safe `{report|null, path, age_s|null}`.
   - POST /doctor/run: bounded re-run of http+sse probes, window <= 6s,
     NO browser/journey sweep (journey report included from disk if present);
     write report atomically; return it.
   - Standard success_response envelope; auth pattern mirrors existing
     system/health GET routers. ALL doctor probes are read-only GETs.

2. **Frontend**: new page `apps/web/app/(app)/doctor/page.tsx` +
   `apps/web/components/doctor/`:
   - DoctorSummary: overall-status + severity counts.
   - FindingsList: severity-ranked, detail collapsed by default (AGENTS
     "summarize don't dump").
   - RunDoctorButton: loading/disabled states.
   - sidebar/nav entry (additive).
   - Data ONLY via http-client apiGet/apiPost (NEVER raw fetch).
   - Component strip shows inference/engine/system health (model_loaded,
     quantization, health_score, cpu/mem) — REUSE the existing health
     hook/pattern the monitoring page uses, i.e. `apps/web/hooks/useApiHealth.ts`.
     **Do NOT edit `apps/web/app/(app)/monitoring/page.tsx` or
     `apps/web/components/monitoring/SystemHealthPanel.tsx`** — a concurrent
     session owns them (mtimes 09:40 today).
   - Noir Violet design system: strui components, design tokens only (no
     hardcoded colors/spacing), hover/focus-visible/disabled states, a11y labels.

3. **Tests**:
   - new `tests/server/test_doctor_router.py` (envelope, null report, run with
     monkeypatched probes, no network).
   - new vitest component tests (severity rendering, collapsed default, button
     states) following SoulList.test.tsx patterns.

4. **Docs**: FEATURES.md entry + extend docs/TESTING.md "Site Doctor" section
   with API/UI surface; INDEX.md touch only if required.

## ACCEPTANCE

- pytest new router tests green + ruff clean
- eslint 0 errors on changed files
- vitest new tests green
- no edits to monitoring page, journey files, existing tests
- http-client used exclusively
- shared-workspace (root toolchain only, zero installs)

## COORDINATION

- parent card `6edae3a7-783b-44f0-9a47-e94ca720ca77` (doctor Phases A/B/C —
  this is the surfacing extension; Phase A landed as commit 729210a48)
- monitoring-page owner session found on board (reference only, DO NOT modify):
  `294818b7-914b-422d-91fb-719fee4abcdc` [in_progress] "modular health blocks +
  UI refinement: backend block registry, typed frontend contract, refine all
  monitoring cards" (tags backend/frontend/observability/design-system) — this
  is the session whose 09:40 mtimes sit on monitoring/page.tsx +
  SystemHealthPanel.tsx; its P2 splits DetailedHealth into per-block types with
  a single normalizer, which is exactly the contract the doctor component strip
  should consume rather than re-parse.
  Also live: `74032dd0-ff9b-4378-8c7b-5e35791a99c0` [review] gateway bandwidth
  observability -> monitoring card; historical `pugqeep-24` [done] SystemHealthPanel
  tests (36 tests).
- reuse, don't rebuild: `apps/web/hooks/useApiHealth.ts` for the health strip.

## SHARED WORKSPACE

- index updated: FEATURES.md + TESTING.md entries in this change; INDEX.md only
  if required (no new app/package)
- shared copy reused: root .venv + node_modules only, zero installs
- coordination named: parent 6edae3a7 + monitoring owner 294818b7
  (+ 74032dd0); do not touch monitoring/page.tsx or SystemHealthPanel.tsx

## RESOLUTION (2026-10-03, status doing -> done)

- **Branch** `fix/doctor-ui` @ `8cfc146e9` (20 files, +1479/-1) merged to main
  via merge commit `40ba857e0` — 3-way; main had advanced 16 commits @
  `277e447ae`, overlap = `docs/INDEX.md` only, auto-merged cleanly.
- **Backend**: `apps/api/server/routers/doctor.py` (110 lines) —
  `GET /doctor/report` (null-safe file serve + `age_s`),
  `POST /doctor/run` (light run: http+sse <=6s + journey-from-disk,
  `run_sweep=False` — API never triggers the browser sweep); registered in
  `_router_names`; POST guarded by `require_auth_if_enabled`.
- **Frontend**: `apps/web/app/(app)/doctor/page.tsx` + 6 components under
  `apps/web/components/doctor/` (DoctorSummary, FindingsList collapsed-detail,
  RunDoctorButton w/ loading + double-click guard + `skipCircuitBreaker`,
  ComponentHealthStrip via `useLiveStatus`); nav entry in `navigation.ts` +
  5 locales; http-client only; design tokens only (grep-verified zero hex/raw
  fetch).
- **Tests**: 50 backend (`tests/server/test_doctor_router.py` +
  `test_doctor.py`) + 39 doctor vitest + 33 integration vitest; eslint 0/0;
  tsc clean for changed files; ruff clean; post-merge re-run 50 passed +
  ruff green.
- **Docs**: FEATURES.md `## Site Doctor UI + API` (:305), TESTING.md API
  contract, INDEX.md.
- **Gate incidents**:
  1. build agent's first commit was never executed — reported fabricated
     `porcelain==0`; recovered by re-commit `8cfc146e9` after re-running
     tests (50 passed).
  2. merge gate blocked twice: main advanced + 4 dirty test files from another
     session's interrupted run -> user granted fresh one-time WIP
     authorization -> WIP commit `6b74a4024` (`wip: preserve interrupted
     test-run edits`, reversal `git reset --soft HEAD^` documented in
     message) -> clean 3-way merge, zero conflicts.
- **Exclusions honored**: monitoring/page.tsx, SystemHealthPanel, journeys,
  domain/testing, avion, existing tests untouched.
