---
id: fb2805c0
title: "Flag surface for web+mobile: boundary discussion, then a projected route"
status: review
board_id: Main
date: 2026-10-14
branch: feat/flag-surface
tags: [gene, flags, endpoint-transport-rule, apps-api, domain-shared, web]
---

# Card fb2805c0 — flag snapshot surface (batch of 2026-10-14)

Worktree `.wt-flags`, branch `feat/flag-surface` (base c6d95a9e2).
Parent card: **d94aea1c** (roadmap decision half — same worktree/branch).

## Boundary (settled on the card — do not re-litigate)

- web/mobile run in the browser (different process) → a transport crossing is
  justified; it must be ONE descriptor-projected route, not an
  endpoint-per-feature, and never a generic flag switchyard.
- Mechanism (agreed in review d1b7a0f60): sync in-memory evaluator in
  domain/shared; refresh fetch at the app-host async seam in
  apps/api/server/infrastructure/startup.py (never domain/ fetching over
  HTTP; core stays sync); offline fallback to config/feature_flags.json;
  clients evaluate a local snapshot and never fetch per read.
- Practical payoff: a flag flip stops needing a process restart (measured
  ~130s under load).
- Endpoint & Transport Rule: ToolSpec-shaped descriptor emitted via
  create_router (apps/api/server/infrastructure/contract.py); registration is
  the side effect of projection; regenerate TS contracts if the descriptor
  changes (scripts/gen_ts_contracts.py, freshness gate
  tests/api-server/test_ts_contracts.py).
- Frontend consumes via http-client.ts only (never raw fetch).

## Scope guards

- No flag deletions (d94aea1c scope limit — needs user sign-off).
- Keep UI minimal: this card ships the route + client snapshot helper, no
  page/component churn.
- Gateway/mobile consume the same route — their wiring is follow-up.

## Status log

- 2026-10-14: claimed by Gene batch 2026-10-14 (coordinator).

## LANDED (2026-10-15) — sha a65ab6f2c on feat/flag-surface

Boundary as settled; scope kept (route + client helper, no page churn):

- `GET /flags` — descriptor-projected (`FLAGS_SPEC` -> `create_router`),
  tier (b) utility tooling, `flags/snapshot` in the TS contract registry
  (regenerated, freshness gate green).
- `FeatureFlags.status_of(name, app)` — one evaluation home for the
  surface; unknown flags fail closed like `is_enabled`.
- 30 s refresh loop at the async seam (`startup.py`,
  `load_config` via `to_thread`, cancelled on shutdown) — a config flip
  lands in <=30 s instead of the measured ~130 s restart.
- `apps/web/lib/protocol/flags.ts` — local snapshot store;
  `isFlagEnabled()` is an in-memory fail-closed read (http-client only).
- Ports the stranded per-app family (d1b7a0f60, 2054d15b6 [was e855a3f4a],
  87c86b16e, 0ff32c259, 89e7ab607): per-app scoping, KNOWN_APPS vocabulary,
  shipped-config guard test, config actually loads at import (hf_finetune
  aligned to effective value).
- Doc: per-app section now describes what ships; roadmap items 2+4 marked
  landed / populate-next.

Gates: 109 passed (flags + per-app + 4 route tests + TS freshness + router
manifest + shared indexes); ruff clean; apps/web `tsc --noEmit` clean
(worktree reuses shared `apps/web/node_modules` via symlink — no install).

Follow-ups (not this card): gateway reader (roadmap item 3); web/mobile
runtime wiring consumes flags.ts (their own cards); flag deletions await
user sign-off on d94aea1c's memo.
