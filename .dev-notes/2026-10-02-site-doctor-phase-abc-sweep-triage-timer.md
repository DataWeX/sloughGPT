---
id: 20261002_site_doctor_phases
title: "Site doctor: Phase A sweep orchestrator + SSE probe + severity; Phase B LLM triage; Phase C nightly timer"
status: doing
date: 2026-10-02
tags: [doctor, journeys, triage, sse, observability]
board_id: 6edae3a7-783b-44f0-9a47-e94ca720ca77
board: 6edae3a7-783b-44f0-9a47-e94ca720ca77
hash: bb0565454f1b2d20415fff96145493181a8cfaa5a06d81fa2f6490b4b8600d93
---

# Site doctor: Phase A sweep orchestrator + SSE probe + severity; Phase B LLM triage; Phase C nightly timer

## WHAT

Build the ER-style site doctor as a THIN LAYER over the existing
`domain/journeys` runner — never a second harness. Phases:

**Phase A — sweep orchestrator + SSE probe + severity**: single entry point
that walks the configured routes/journeys against the running stack (web
:5173 → gateway :8080 → API :8000, URLs from env/config, NEVER hardcoded
:3000/:5175) and merges into ONE findings doc: (a) `domain/journeys` report
(~/.cache/slog-journeys/ux-flows-report.json), (b) console/pageerror + failed
network (reuse journey capture; adopt card 13f50db7 avion CDP output when it
lands), (c) `GET /errors/recent|grouped|trends`, (d) `/health/detailed`
recent_errors/rate_violations, (e) **new SSE probe**: subscribe to
/health/stream + chat stream for a bounded window, assert event cadence +
payload sanity (this session's original bug class: 10.5 MB events / stalls).
Severity/score per finding reusing
`domain/infrastructure/_internal/health_flow.py` Diagnosis/Severity vocabulary
— no new vocab. Proposed home: `domain/journeys/triage.py` (or
`domain/doctor/` if it grows); route data ported from legacy
`SLOUGHPGPT_PAGES` (data only — do NOT keep/re-land the `domain/testing`
package).

**Phase B — LLM triage**: inference-engine pass over the Phase A findings doc
→ severity-ranked worklist, following the avion.ai pluggable-model pattern
(categorizer precedent: .opencode/plugins/ui-error-watcher.ts). REPORT ONLY:
doctor orders tests, doesn't operate — no auto-fixes, no auto-created cards
until triage is trusted.

**Phase C — scheduling**: nightly systemd *.timer running the sweep, gated on
coordinating with the units/systemd session first (no timers exist in-repo
today). Findings history/baseline so regressions diff against the previous
run.

## ACCEPTANCE

Unit tests for the merge/severity logic (pure, no browser); SSE probe test
with a fake stream (reuse tests/server test patterns for health stream);
journey-sweep e2e stays browser-gated/skippable like existing avion tests;
docs entries (docs/INDEX.md + docs/TESTING.md or FEATURES.md) added in the
same change; lint + typecheck + tests green before merge; no benchmark (not a
training/inference change).

## SEQUENCING / DEPENDENCIES

After (a) fix/health-payload-trim merges (branch held at 2c460cbaf — main
worktree was dirty; re-probe then merge) so the doctor doesn't just
re-diagnose the already-fixed 10 MB health payload, and (b) card e47e19ee
retire-3000-refs (URL truth: live web is :5173). Phase B after Phase A has
produced a stable findings shape.

## COORDINATION (link, do not race)

- coordination/design card `e8fd60d4-20c0-4c36-869e-05db22bf83ef` (parent —
  inventory lives there)
- journey owner cards `20260927_067_journey_library` (done),
  `e0844f51-64d1-48a7-9aae-d3ec30f3ae60` (handoff, todo),
  `13f50db7-975d-4c6a-86c4-7d8ea190e259` (avion CDP capture, in_progress —
  DO NOT RACE)
- notes 2026-09-28-journey-library.md,
  2026-09-29-card078-journeys-runner-landed-handoff.md,
  2026-09-29-avion-cdp-cli.md
- worktrees .wt-jrn/.wt-gates/.wt-ux
- card `d7492f1a` health trim (sequencing dep)

WARNING: staged domain/testing resurrection in sloughgpt-api-std is fake-green
legacy — never build on it.

## SHARED WORKSPACE

- index updated: docs/INDEX.md (+TESTING/FEATURES) in the implementation change
  (design card — no index change required now)
- shared copy reused: root .venv + node_modules only, zero new
  installs/lockfiles (avion/playwright already shared)
- coordination named: the five links above — parent e8fd60d4, journey owner
  cards 20260927_067 + e0844f51 + 13f50db7, notes 2026-09-28-journey-library.md
  / 2026-09-29-card078-journeys-runner-landed-handoff.md /
  2026-09-29-avion-cdp-cli.md, worktrees .wt-jrn/.wt-gates/.wt-ux, sequencing
  deps d7492f1a + e47e19ee

2026-10-02: Phase A implementation started on branch fix/doctor-phase-a (user directive 'continue building').

## Phase A — DONE 2026-10-02 (merged)
- Final home per user directive: `domain/core/_internal/doctor/` — NO new domain; wiring layer in domain/core ("the body of the system"), probe seams integrating core components. Branch fix/doctor-phase-a, commit 729210a48 "feat(doctor): Phase A — site doctor wiring in domain/core (probe seams + severity triage)" — 12 files, +2081. MERGED to main 2026-10-02 fast-forward (34898f48 → 729210a48), worktree clean, post-merge pytest 41/41 + `from domain.core import run_doctor` import ok, ruff clean.
- Built: Finding→Severity/Diagnosis banding (models.py, reuses health_flow vocabulary); probe seams journey.py (wraps domain.journeys.runner, report-file triage, doctor-side SLO_WEB_URL setdefault :5173 — journey files untouched), http.py (5 read-only GETs: /health, /health/detailed, /errors/{grouped,trends,recent}, injectable fetch), sse.py (/health/stream frame-size/stall/ERROR watchdog + /errors/stream best-effort); report.py (merge → severity-ranked DoctorReport, atomic write, exit codes 0/1/2); CLI `python -m domain.core._internal.doctor [--no-sweep --window --json --skip --strict]`; domain/core lazy exports run_doctor/DoctorReport; docs/INDEX.md:63 + docs/TESTING.md:254.
- LIVE SMOKE VALIDATION: read-only run against the live old-branch API flagged critical `sse/stream.frame_size` max 10,566,958 B (limit 262,144 B) across 4 frames + warn journey/console_errors (2 react-virtuoso), API health 85 healthy — i.e. the doctor caught this session's original bug on first run (exit 2 as designed).
- Phase A residual notes: 41 hermetic tests in packages/core-py/tests/test_doctor.py; probe registry PROBES (http→sse→journey) is the extension seam.
- REMAINING (unchanged): Phase B — LLM triage (avion.ai pluggable-model pattern, report-only) → still todo. Phase C — nightly systemd timer + findings history → still todo (coordinate with units session). Chat-stream probe deferred to Phase B (needs authenticated session/model state).

## Status note 2026-10-17 (verified against origin/main)

Phase A (sweep orchestrator + SSE probe + severity) is DONE — landed as
phases 1+2 in f8d4ee873 (PR #136) + phase 3 in 29afaa8f0 (PR #138), plus
the mole probes family (9dcee3fed/e96bf53e6, PRs #135/#137). The
remaining scope is Phase B (LLM triage — no avion pluggable-model
classifier on main) and Phase C (nightly systemd timer — no doctor/mole
timer on main; coordinate with the units session per the card). Build
the remainder ON mole's probe architecture (domain/core/_internal/mole),
not a parallel module — a 10-17 attempt at a standalone triage layer
turned out to be a 100% duplicate of mole's journey/http/sse probes and
was scrapped before landing.
