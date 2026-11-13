---
id: 0c23913d
title: "Worktree hygiene: reclaim .wt-* space by merge/remove, never rm"
status: done
board_id: Main
date: 2026-10-19
branch: chore/wt-hygiene-20261019
tags: [infra, git, coordination]
---

# Card 0c23913d — safe .wt-* reclaim (provably-safe subset only)

Full card id: `0c23913d-26c3-4ad5-95ac-d0cd6c93fe4f` (board column: in_progress).

## LANDED 2026-10-19 (Gene batch of 3, done inline — subagent nesting blocked at depth 1)

- **Commit**: `6f21a1a69` on branch `chore/wt-hygiene-20261019` (worktree
  `.wt-wthygiene`), parent = `23fb2ba3b` (= feat/generation-merge tip, the
  B landing point / this batch's "main"). Pathspec commit, 1 file,
  208 insertions — `git show --stat` exact, no foreign hunks.
- **Removals executed: 0 — the safe subset is provably empty** (not a
  non-result: measured, twice). Final classification vs `23fb2ba3b`:
  all 8 pre-existing merged-ancestor worktrees are DIRTY with human work
  (decode 10 files, strui 13, gstage 5, pguard cluster 2-3 each, b1-fsrv
  `.splitwork/`); every one of their branches carries 0 unique commits
  (rev-list = 0, tips cat-file-alive). Clean merged candidates
  (.wt-astland, .wt-chatstate, .wt-land) are ACTIVE → untouchable.
  Second classification vs the alternate trunk `f53190f7f`: also 0.
- **Split-brain caveat for verification**: local `main` ref = `f53190f7f`
  is a DIFFERENT trunk without B; verify against `feat/generation-merge`
  (= 23fb2ba3b), not `git rev-parse main`.
- **Merge queue** (report-only, 72 unique + 4 detached = 76 rows,
  merge-tree conflict risk vs 23fb2ba3b): 10 zero-conflict tier-0
  candidates; sweep-singles cluster tier-last (~1,350 conflicts each);
  claimed lanes (planner-delete 8c4b7927, tokenizer f47069b0) flagged.
- **Space**: removals reclaimed 0 bytes by design; `.wt-*` du 8.6G → 8.7G
  (concurrent growth), df / 4.6G → 2.2G free over the batch — ALL of the
  disk pressure is other sessions; .wt-fbdrv untouched, no rm, no --force.
- Gates: worktree count invariance + no primary-tree mutation + doc
  commit verified. Status: done pending foreman landing the branch.

## Dispatch (2026-10-19, Gene batch of 3)

- **Worktree**: `.wt-wthygiene`, branch `chore/wt-hygiene-20261019`
  (artifact/report commit only; `git worktree` ops are repo-wide metadata,
  executed from this worktree against the shared admin)
- **Routed to**: general
- **Foreman measurements (verified)**: 151 registered worktrees; 80 `.wt-*`
  scanned; 9 branches already ancestors of HEAD (removal candidates);
  `git worktree prune -n` → 0 dangling; 1 unregistered husk `.wt-fbdrv`
  (~408K USER DATA — OUT OF SCOPE, human decision); `.wt-*` ~12G after
  earlier artifact sweep (card af346301 — do not redo).
- **DO**: classify all 80 (already-merged / stacked-duplicate / truly-unique /
  husk / ACTIVE); ACTIVE = mtime < 24h OR process cwd inside (scan
  /proc/*/cwd) — untouchable. For already-merged AND clean AND not-active:
  `git worktree remove` WITHOUT `--force`, after re-verifying every
  branch-only sha alive via `git cat-file -e`. Produce ranked merge queue
  (REPORT ONLY) for truly-unique branches with per-branch conflict risk.
- **MUST NOT**: `rm -rf`; `git worktree remove --force`; `.wt-fbdrv`;
  active worktrees; touching tracked source (report artifact is a NEW doc file,
  docs/INDEX.md stays untouched — it is foreign-staged elsewhere).
- **Report artifact**: `docs/misc/worktree-hygiene-2026-10-19.md`
- Status: wip → review with before/after evidence.

Logs: `~/.cache/slog-gates/gene-batch-20261019/`
