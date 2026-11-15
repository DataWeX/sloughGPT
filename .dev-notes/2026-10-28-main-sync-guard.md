---
title: "Main sync guard — local main must track origin/main (mechanical pre-commit/pre-push gate)"
status: done
date: 2026-10-28
tags: [infra, git, sync, hooks, agent-sync]
---

# Main sync guard

## What

Prevent the 2026-10-27/28 failure mode recurring: local `main` forked at
`9002f1a0` (last ff from origin), accumulated **149 local-only commits**
(kanban/docs/real web code, incl. a "local only, not pushed" merge) while
`origin/main` moved **35** the other way. Naive reconciliation = 8 conflicts
incl. `Providers.tsx` modify/delete (Next-era local tree vs post-#217 main)
and `package-lock.json` next-vs-no-next — a push of local main would have
reverted the Next removal.

## Mechanism

- `scripts/check-main-sync.sh` — HEAD=main: best-effort `git fetch origin main`,
  then `git rev-list --count origin/main..HEAD`; **>5 ahead → exit 1** with
  the landing-flow message; behind → warning; `SLO_MAIN_SYNC_SKIP=1` escape;
  non-main HEAD → no-op (lane `HEAD:main` pushes unaffected).
- `.husky/pre-commit` (guard before lint-staged), `.husky/pre-push` (blocks
  pushing a >5-ahead main), `.husky/post-merge` + `.husky/post-checkout`
  (warn-only status).
- Policy text: AGENTS.md Core Rules + Agent-sync clause; AGENT_SYNC.md
  purpose-block hard rule (docs alone don't hold — hence the hook).

## Landed (evidence)

- `3b6fe1c93` — script + post-merge/post-checkout warn hooks
- `212568adc` — pre-commit/pre-push wiring + AGENTS.md/AGENT_SYNC.md policy text
- Verified on origin/main: all four greps present; scratch sim 6/6; husky chain via lint-staged

## Follow-up (separate card)

Salvage triage of the 149 local commits: port genuinely-unpushed code onto
post-removal main via PR, union-merge kanban/notes jsonl, then repoint both
main-owning worktrees (quick-engine, sloughgpt-api-std) at origin/main.

## Notes

- Board sync intentionally NOT run: `.kanban/board.jsonl` dirty from
  concurrent lanes (would stomp uncommitted edits).
