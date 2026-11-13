---
title: "Salvage the 149 local-main commits — merge onto post-removal origin/main via PR, union jsonl, repoint worktrees"
status: wip
date: 2026-10-28
tags: [infra, git, sync, salvage, kanban, agent-sync]
---

# Salvage local main's 149 commits

## What

Local `main` (a88816c5d, forked at 9002f1a0) holds 149 unpushed commits
(~25 web fixes/features, planner, readiness-verdict merge, 48 kanban/notes
bookkeeping) against a Next-free origin/main. Reconcile by MERGING local main
into a branch off origin/main (merge-tree: only **8 true conflicts**; 291
local-side files auto-merge), then PR.

## Decisions

- 8 conflicts: tokenizer + training/runs page tests, `.eslintignore`,
  `notes.journal.jsonl`, `AGENT_SYNC.md`, `board.jsonl`, `package-lock.json`,
  `Providers.tsx` (modify/delete — origin deleted it in #217).
- jsonl files → union merge (both sides' records kept).
- package.json/lock → origin (no-next) + re-add only genuinely-local deps;
  regen lock via `npm install --package-lock-only`.
- Next-removal invariants after merge: grep `from 'next*` = 0, no next deps.
- Codemod re-run; gates tsc/eslint/vitest/build + CI python suites.

## Follow-through

- Repoint `quick-engine` + `sloughgpt-api-std` (both hold main) at origin/main
  only if their worktrees are clean.
- AGENT_SYNC entry rides the PR push; board sync after board.jsonl lands.
