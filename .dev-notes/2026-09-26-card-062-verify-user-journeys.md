# Card 062 — Build Order #7: verify 12 UX_FLOWS journeys in browser

- **Status**: done
- **Branch**: `feat/journey-verification` (worktree /home/mana/Documents/sloughgpt-journeys, base cd6638a8c + fix 15bc0a3da)
- **Date**: 2026-09-26 (started 2026-09-25, card record recreated — earlier board entry lost to parallel-session board rewrite)

## Mandate

Boot the stack, walk all 12 features + Training 3-click flow from `docs/UX_FLOWS.md` in a real browser, fix what blocks them. Non-technical user must be able to complete each journey.

## Environment (working)

- uvicorn `:8000` + vite `:5175` (worktree apps/web, hardlink-farm node_modules)
- Firefox via playwright-core (persistent `~/.local/share/ms-playwright/firefox-1543`, symlinked from `.cache`) — Chromium 151 has cookie-init nav deadlock (ERR_BLOCKED_BY_LOCAL_NETWORK_ACCESS_CHECKS first; every http nav NEVER commits)
- Scratch: `~/.cache/slog-journeys/` (other session wipes /tmp/opencode)

## Blockers found + status

| Blocker | Status |
|---|---|
| Chromium nav deadlock (cookie store init) | pivoted to Firefox |
| `setup_providers` cfg.autoload_model AttributeError | FIXED 15bc0a3da |
| No model weights anywhere | pending card 061 → SLO_BOOTSTRAP_HF boot path |
| Router download passes int as url | suspect, unverified |
| load_model missing slnc conversion | documented, unverified |
| tiny catalog model unloadable | card 061 regression target |


## Outcome (closed 2026-09-27)

- **13/13 journeys pass** (`docs/UX_FLOWS.md` flows 1-12 + training 3-click), 0 network errors, report at `~/.cache/slog-journeys/ux-flows-report.json`.
- Runner: `scripts/run_ux_flows.py` (avion engine) — persists report after every flow, waits for StartupOverlay unmount, polling element waits, `--flow/--list/--headed/--strict-errors`.
- Blockers fixed on `feat/journey-verification` (commits `91fa1a145`, `08a8c8304`, `9c30fa601` → main `73ab55d8f`):
  - model load failure (card 061 merge): dim_ff ceil-to-64 broadcast error → all chats 503.
  - `domain/learner` `enrich_with_knowledge` export — "Knowledge enrichment failed" on every chat.
  - `db.ts` `toStored`/`fromStored` + session inspector partial-session guards — recurring `session.messages is undefined` pageerror.
  - message-notes docstore routes (list/put/search/delete) + `COLLECTIONS` registration.
  - CORS defaults include 5173/5175.
  - avion: arken→avion rename completion, run_task failure-screenshot guard (killed whole runs), test import fixes (avion suite 228 passed / 6 pre-existing failures).
  - Runner robustness: ENOSPC recovery (disk full killed servers), overlay wait, polling `s_exists` (7-talk voice opens async).
- 120s whole-server wedge: **root-caused + fixed 2026-09-28** (commits 6f6e73f7f, 7660e4d82 — stream consumer leaked a default-pool thread per 30s timeout, starving every endpoint; plus prompt-echo strip hardened to startswith + 5 stale tests refreshed). Post-merge run after fix: 13/13.
- Remaining polish (documented, not blockers): boot-window SSE 503 noise, gpt2-provider UI load failures.
