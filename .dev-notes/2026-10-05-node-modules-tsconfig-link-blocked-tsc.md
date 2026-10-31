---
id: 20261005_node_modules_tsconfig_link_tsc_verdict
title: "node_modules tsconfig link blocked tsc — verdict: link OK, artifacts gone (504f9a1c)"
status: done
date: 2026-10-05
tags: [frontend, infra, typescript, tsc, kanban]
---

CARD REF 504f9a1c — user directive 2026-10-05: "confirm the linked-path
hypothesis, then update the card."

The referenced card does not exist anywhere: 25+ `board.jsonl` copies
(root, every `.wt-*`, 9 sibling repos, `.cache/slog-*`), every dev-note
(content AND filename), git (246 branches, stash, `log -S`), the opencode
DB (messages, parts, events, todos, kv, session titles) — zero hits. The
original user message was purged by the 03:47 compaction; only my own
responses quoting fragments survive. User approved filing the verdict as
a fresh card (question, 2026-10-05).

## Verdict — linked-path hypothesis: mechanism CONFIRMED, "blocks tsc" REFUTED

### 1. Mechanism confirmed (traceResolution)

`tsc --traceResolution` run from `.wt-orb/apps/web` resolves straight
through the symlinked node_modules into the ROOT checkout:

    Type reference directive 'next' → /home/mana/Documents/Default Project/sloughGPT/node_modules/next/index.d.ts

`.wt-orb/apps/web/node_modules` and `.wt-orb/node_modules` are symlinks
to the root copies; TypeScript realpaths through them
(`preserveSymlinks: false`, its default). Worktree paths DO escape to the
root tree — the `<repo>/...` symbolic-resolution behavior the card
describes. This is also exactly what makes the one-shared-node_modules
policy work: content is identical by construction.

### 2. It does not block tsc

- `.wt-orb/apps/web`, FULL cold check `tsc --noEmit --incremental false`:
  **exit 0, 0 errors** (12m39s wall under load 20, 3m53s user).
- root `apps/web`, same command: **exit 0** (2m04s).
- warm/incremental runs in both trees: exit 0.

### 3. The original 44-error artifacts are gone (not reproducible)

- `@workspace/ui`: zero hits in any working tree, any of the 246 git
  branches (`git log -S`), any tsconfig. Root `workspaces` =
  `web / strui / sdk-ts / eslint-config` only — the package never
  existed in this repo's history.
- no `.next/` directory exists anywhere now (16 of the reported
  breakdown).
- no `.venv` anywhere under `Documents` (AGENTS.md still documents one —
  it has been deleted from disk); a full-system find for
  `site-packages/ui` returns nothing, so the reported
  `find … → .venv/lib/python3.12/site-packages/ui/button.js` target is
  gone too.

## Conclusion

The symlink behaves as designed. The 44 errors came from since-deleted
artifacts (stale `.next/cache` + a removed `.venv`, consistent with the
card's VSCode inferred-project quote) leaking into a type context — not a
live defect in the link. If it ever recurs: re-capture the raw error list
AND the triggering command before touching tsconfig.

Follow-up observation (separate scope, not filed): AGENTS.md still
documents a repo `.venv` (stdlib 3.12.3) that no longer exists on disk —
doc drift worth a cleanup pass.
