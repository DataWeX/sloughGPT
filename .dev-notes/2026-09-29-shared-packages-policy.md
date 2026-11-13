---
id: 20260929_080_shared_packages_policy
status: done
title: "Card 080 — Tell every agent: packages are shared, not rebuilt; keep the indexes; cooperate"
tags: [policy, agents, coordination, packages]
sprint: current
gh: null
---
# Card 080 — Shared-packages policy for all agents

## User directive (verbatim intent)

"Tell every agent in our session that the packages we use are shared
instead of rebuilding — we can have an index for newly built apps and
cooperate on features and when we fix them."

## What was written

- **`AGENTS.md` Core Rules** (the file every agent reads at session
  start; subagents inherit it):
  1. *Shared, never rebuilt* — ONE `.venv`, ONE root `node_modules`,
     ONE `packages/*`, ONE `apps/` for every session/worktree/agent.
     Search-and-reuse before any install/copy/scaffold; fix shared
     bugs in the shared copy, never fork a private rebuild. Cites the
     7.5G `apps/mobile/node_modules` duplicate as the cautionary cost.
  2. *Keep the shared indexes current* — new apps → `apps/README.md`,
     libs → `packages/README.md`, docs → `docs/INDEX.md`, features →
     `docs/FEATURES.md` / `PRODUCT_ENGINEERING.md`; check them first,
     update them in the same change; coordinate shared features/fixes
     through the kanban card.
- **`apps/README.md`** — completed the app index: added the missing
  `gateway/` (Rust edge, built once, shared binary), `mobile/`
  (regenerable 7.5G install — do not rebuild), `data/` (user stores —
  never delete) entries.

## Visibility caveat

Running sessions read `AGENTS.md` at start — already-running parallel
agents pick the rule up on restart/re-read; new sessions and subagents
get it immediately. The note itself is the durable journal record.
