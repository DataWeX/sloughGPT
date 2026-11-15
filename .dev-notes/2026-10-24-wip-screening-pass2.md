---
title: 2026-10-24 — wip-column screening pass 2 (6 rows resolved, 2 builds)
status: open
date: 2026-10-24
tags: [kanban, board-hygiene, screening, wip, ports]
---

# Wip screening pass 2 — 17 unowned wip rows triaged

Same discipline as pass 1 (`2026-10-23-review-column-screening-flips.md`):
screen against `origin/main` code first, flip only with sha/content evidence,
port (never re-implement) when a settled fix exists on an unlanded branch.

## Flipped to done (4)

| card | evidence |
|---|---|
| `a9bc8648` moodboard skill | `.opencode/skills/moodboard/SKILL.md` + `docs/moodboards/assets/` on origin/main — the four-stage pipeline the card specced. |
| `278fdeaf` kanban content hashes | `packages/app-planner/src/app_planner/store.py` (chain_hash/chain_prev/chain_index live on every board row) + `tests/test_store_chain.py` (compute/verify/sync) + note provenance (6 provenance/note_id refs) on origin/main. |
| `656023ab` http-client.ts modularization | `apps/web/lib/http-client.ts` on origin/main is 28 lines — a pure facade re-exporting one-module-per-concern from `apps/lib/` (errors · types · corr-id · url · retry · interceptors · resilience · state · request · verbs · api-client · client · auth · sse). 1443→28. |
| `9fc4b4cf` fault containment Tier-1 | On origin/main: `infrastructure/staged_loader.py` (critical/ready/background staged startup), `infrastructure/startup_rollback.py`, health capability/degraded in `routers/health.py` + `controllers/health.py`, pinned by `tests/server/test_startup_routers.py` + `test_startup_profiler_feed.py`. Planned `test_containment.py` name changed; behavior pinned. |

## Ported + moved to review (2) — fixes existed on unlanded branches only

| card | port |
|---|---|
| `5bfa16e5` tie-sort TypeError (LIVE MAIN BUG) | `origin/main:domain/infrastructure/_internal/context_core.py:274` `scored.sort(reverse=True)` on (score, episode_dict) still compares dicts on tie → TypeError → 500 + zero streamed tokens. Ported fix+test verbatim from `70b021b6a` (fixes B/C of that card already on main by other means — verified by content). RED with exact production error, GREEN 113/113. Commit `6f3238f00`, note `2026-10-24-tie-sort-port.md`. |
| `8e9ee65d` compaction-tight.ts | Single-file exact port of `aed61d144` (.opencode/plugins/compaction-tight.ts, 141 lines) — SOP-field compaction prompt override, fail-open, prior-summary merge. Commit `d2f14aa65`. Both ride the stack; both cards close at landing. |

## Left open (deliberate)

- `6edae3a7` site doctor — only `scripts/site_doctor_probe.py` + contract test on
  main; the orchestrator ("doctor spawns it", `5fe9d970a`) and rehomed
  `tests/ux_journeys` ride unlanded branches (`chore/ast-reachability`,
  `feat/app-ready-signal`, `feat/ast-union-report` — active lane).
- `4a1e62fa` DOMAIN_CONSOLIDATION design record — branch
  `feat/generate-consolidation` (tip 2026-10-20, ACTIVE) already updated
  PRODUCT_ENGINEERING/ROADMAP/USER_PERSONA citing this card (`9980f56e5`);
  the doc itself is that lane's business. Do not port.
- `284aa214` "Synced Task" — empty shell (no description), 2026-09-24. Row
  deletion is user-gated (board-line rule) — flagged, not touched.
- `294818b7`, `ecea49c8` (updated 2026-10-20 — possibly active), `bb8212a8`
  (strui merge pending), `a92c4a31` (rides stack landing `2cec0811`),
  `89c58cfb` (user-gated dedupe), `ca647de2` (CI lane), `6109e26e`/`65ff978f`
  (strui twins) — pre-classified, untouched.
- Legacy priority-1 rows "Product docs truth-up + guard" / "QUICKSTART truth-up
  + guard (seam 6)" (date-style ids, 2026-09-29): no guard for those docs on
  main (`doc-guard.ts` is an area→docs reminder, not a freshness gate). Truth-up
  is a real editorial job — deferred; closing needs store-level surgery
  (CLI can't address legacy ids).

Board: wip 24→19, done 910→914, review 10→12 before other sessions' moves.
