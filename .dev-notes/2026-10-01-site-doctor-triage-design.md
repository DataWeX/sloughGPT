---
id: 20261001_site_doctor_triage
title: "Site doctor: triage layer over domain/journeys — coordination + design (builds on journey owners, no new harness)"
status: done
date: 2026-10-01
tags: [journeys, triage, design, coordination, doctor]
board_id: e8fd60d4-20c0-4c36-869e-05db22bf83ef
board: e8fd60d4-20c0-4c36-869e-05db22bf83ef
hash: 422b74e6bdf2356791562bda702dded14fc16d11d894bcc7b54239b9aed2e80b
---

# Site doctor: triage layer over domain/journeys — coordination + design (builds on journey owners, no new harness)

## GOAL

ER-style site doctor — probe the running site AS A CONSUMER through the single
point (:5173 web → :8080 gateway → :8000 API), collect symptoms (journey
failures, console/pageerror, HTTP ≥4xx/5xx, SSE drops, API health), then
inference-engine triage → severity-ranked worklist (report only at first;
doctor orders tests, doesn't operate).

## WHAT EXISTS (inventory 2026-10-01)

`domain/journeys` committed on main = the ONE runner (`python -m
domain.journeys`; 13 spec-linked flows tied to docs/UX_FLOWS.md; real browser
via avion+playwright; console/pageerror/HTTP≥400 capture; findings JSON →
~/.cache/slog-journeys/ux-flows-report.json; `--strict-errors`). Server-side
severity vocabulary: domain/infrastructure/_internal/health_flow.py
Diagnosis/Severity + `/health/detailed` (recent_errors, rate_violations).
Error endpoints: routers/errors.py (/recent,/grouped,/trends,/stream).
Crawler: packages/bawl (stdlib, unused by journeys). Test-suite doctor:
scripts/test-doctor.py (different scope — suite failures, not site). Cypress:
22 specs, mock-based, no console/network assertions. Route inventory:
domain/testing/_internal/sloughgpt.py SLOUGHPGPT_PAGES (~110 routes) — data
only.

## WHAT'S MISSING

Sweep orchestrator merging journey+console+network+API+server-errors into ONE
findings doc; SSE probe; severity/score in journey report (reuse
health_flow.Diagnosis); LLM triage (only precedent: avion.ai pluggable models
+ .opencode/plugins/ui-error-watcher.ts regex categorizer); bawl wiring;
scheduling (no *.timer anywhere); findings history/baseline; owning doc entry
(docs/INDEX.md, docs/TESTING.md).

## PROPOSED HOME

Thin layer over domain/journeys — `domain/journeys/triage.py` (or
`domain/doctor/`); port route data from SLOUGHPGPT_PAGES, do NOT keep the
domain/testing package. Findings reuse health_flow.Diagnosis vocabulary.
Triage inputs: journey report + GET /errors/recent|grouped|trends +
/health/detailed + optional bawl crawl. LLM classifier behind avion.ai
pluggable-model pattern. Scheduler later: scripts/systemd/*.timer (confirm
with the units session first).

## COORDINATION (link, do not race)

- owner card `20260927_067_journey_library` (done)
- handoff card `e0844f51-64d1-48a7-9aae-d3ec30f3ae60` (todo, re-target markers/tests)
- card `13f50db7-975d-4c6a-86c4-7d8ea190e259` (in_progress, avion CDP
  console/network capture — note 2026-09-29-avion-cdp-cli.md says DO NOT RACE)
- notes 2026-09-28-journey-library.md (in_progress),
  2026-09-29-card078-journeys-runner-landed-handoff.md (todo)
- journey worktrees in flight: .wt-jrn (feat/headless-journey-sweep),
  .wt-gates (fix/journey-test-gates), .wt-ux

## WARNINGS

(a) worktree sloughgpt-api-std has STAGED adds re-landing domain/testing +
run_journey_tests.py — byte-identical to the pre-consolidation legacy DELETED
at 30fed5792 as fake-green (all steps hardcode passed=True); do NOT build on
it; flagged to user. (b) URL drift: domain/journeys targets SLO_WEB_URL :5175,
legacy targets :3000 — actual live web is :5173 (nothing on :3000; see card
e47e19ee retire-3000-refs); doctor must use configured URLs. (c) Sequencing:
after fix/health-payload-trim merges (held: main worktree dirty) so the doctor
doesn't just re-diagnose the 10 MB health payload we already fixed in that
branch.

## SHARED WORKSPACE

- index updated: no index change required (design card; docs entries come with
  implementation)
- shared copy reused: root .venv/node_modules only, zero installs
- coordination named: the five links above (cards 20260927_067_journey_library,
  e0844f51-64d1-48a7-9aae-d3ec30f3ae60, 13f50db7-975d-4c6a-86c4-7d8ea190e259
  + notes 2026-09-28-journey-library.md,
  2026-09-29-card078-journeys-runner-landed-handoff.md), journey worktrees
  .wt-jrn/.wt-gates/.wt-ux, and cards A (d7492f1a health trim) + C (e47e19ee
  :3000) as sequencing dependencies

## Closed 2026-10-17 (fork-board reconciliation)

All coordination deps done (13f50db7, e47e19ee, d7492f1a) and the build
landed on origin/main: phases 1+2 as f8d4ee873 (PR #136), phase 3 as
29afaa8f0 (PR #138), plus the mole family (9dcee3fed/e96bf53e6,
PRs #135/#137) — all ancestor-verified. The earlier close (31ac8f1b5)
was stranded on a side branch and this note was never flipped, so the
fork board reverted the card to todo; flipped note-first this time.
Residual scope (Phase B LLM triage, Phase C nightly timer — neither on
main as of 10-17) stays tracked on card 6edae3a7.
