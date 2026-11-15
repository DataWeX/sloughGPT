---
id: 433ccdb9
title: "JOURNEY HARNESS DELETED (92af1aa0): no uxjourneys branch may land — packages/ and tests/ both superseded"
status: review
board_id: Main
date: 2026-10-18
branch: read-only (no branch — verification only)
tags: [coordination, journeys, ux-flows, mole, deleted, git]
---

# Card 433ccdb9 — guard: no uxjourneys branch may land (harness DELETED)

Full card id: `433ccdb9-07c9-4dcc-8d13-c576fefa9f7a`.
Parent context: card 92af1aa0 (in_progress), its note
`2026-10-16-card-92af1aa0-rehome-uxjourneys.md`.

## Dispatch (2026-10-18, Gene batch of 3)

- **Worktree/branch**: read-only verification (git object reads only) —
  `.wt-journeyguard` only if a clean checkout is genuinely needed
- **Routed to**: git
- **No destructive ops**: do NOT delete branches, do NOT merge, do NOT
  touch `.cache/wt-uxjourneys` (foreign dirty worktree).

## Clauses to verify with evidence

1. Forbidden branches: `feat/uxjourneys-rehome` @ 44b3a6754 (rejected
   packages/ design), `refactor/uxjourneys-rehome` (superseded rehome
   2beb966b0 — but tip also carries the approved CUT 8b77470d8),
   `feat/rehome-uxjourneys` (must be absent). Plus the unlisted 4th:
   `feat/uxjourneys-rehome-land` @ 8781496d3 (worktree gene-wt-a) —
   assess as at-risk.
2. What still stands at main: cut 8b77470d8 is NOT merged (harness paths
   still present at main) — distinguish "guard: rejected design never
   landed" from "cut still unlanded" (that landing is 92af1aa0's scope).
3. Merge-order clause vs `test/journey-content-gate` @ 79b788b95: its
   must_see/must_not_see edits target `domain/journeys/flows.py` +
   `tests/test_ux_flows_library.py`; under the cut both are deleted
   either order → rework must be re-expressed against docs/UX_FLOWS.md.
4. Remote refs: no forbidden branch present on origin.

## Evidence (2026-10-18, Gene — read-only verification, no commits)

### 1. Nothing forbidden ever reached main — SATISFIED
- `packages/uxjourneys`: absent from main tree AND history (`git log main
  -- packages/uxjourneys` empty). `tests/ux_journeys`: never in main
  history (empty). The one rehome that briefly landed per note a7bb38fd
  (`5fe9d970a`) and its revert `9074fad93` are BOTH absent from current
  main history (split-brain hard-reset aftermath, card 240da352).
- `git grep -i "uxjourneys|ux_journeys" main` → only historical log/docs
  rows for the still-present harness (AGENT_SYNC:882 other-branch entry,
  FEATURES:302 legacy-runner row); no dangling code refs.
- Remotes: no forbidden name on origin (only unrelated
  `origin/feat/goal20-consciousness-rehome`).

### 2. Branch state (all four)
| branch | tip | vs main | merge-tree conflicts | worktree | last commit |
|---|---|---|---|---|---|
| feat/uxjourneys-rehome (REJECTED packages/) | 44b3a6754 | +142/-359 | 48 | none | 10-16 08:13 |
| refactor/uxjourneys-rehome (superseded rehome + APPROVED cut) | 2af46a361 | +7/-56 | **0** | .cache/wt-uxjourneys (foreign; dirty ONLY .kanban/board.jsonl + 4 test_results JSONs — no source edits) | 10-16 14:34 |
| feat/rehome-uxjourneys (delete) | — | ABSENT ✓ | — | — | — |
| **feat/uxjourneys-rehome-land (UNLISTED 4th hazard)** | 8781496d3 | +150/-359 | 46 | .cache/gene-wt-a (clean) | 10-16 08:26 |
No uxjourney branch has any commit since 10-16 (inactive). `.wt-gpuops`
stood down ✓ (now `test/planner-delete-finality`).

NET of `main...refactor/uxjourneys-rehome` = 33 files, +140/-1736 = the
approved cut ONLY (D domain/journeys/* + mole probe + run_ux_flows + their
tests; the superseded rehome is net-zeroed by the cut). So its landable
"any form" is a DECISION: merge (0 conflicts today) or cherry-pick
8b77470d8+a2d494fe0+729319ee2 (a7bb38fd note chose cherry-pick for the
REHOME era; the cut era supersedes it). Not landed here — no merges done.

### 3. Merge-order clause vs test/journey-content-gate (79b788b95)
Branch unchanged @79b788b95, worktree clean; `main...` diff = +29
`domain/journeys/flows.py` +67 `tests/test_ux_flows_library.py`, ZERO
`docs/UX_FLOWS.md` work → the mandated re-express has NOT started. Both
target files are deleted by the cut (and the cut strips UX_FLOWS.md's
registry paragraph), so whichever lands second hits modify/delete — per
card, its additions are moot and must be re-expressed against
docs/UX_FLOWS.md. merge-tree(content-gate, main) = 0 conflicts TODAY.

### 4. What remains at risk (guard NOT fully satisfied → keep open)
1. The approved CUT is still unlanded: main still ships the whole harness
   (domain/journeys/, scripts/run_ux_flows.py, mole probe,
   tests/test_ux_flows_library.py). Landing = card 92af1aa0 scope, but it is
   the user-visible deviation ("we don't need it").
2. Two rejected branches + the unlisted 4th still exist; recommend stand-down
   and (with user approval) deletion — nothing deleted here.
3. content-gate rework (UX_FLOWS.md re-express) not started.
4. Landing vehicle for the cut (merge vs cherry-pick) undecided.

Verdict: guard clause "no uxjourneys design ever landed, none active, none
on the remote" = VERIFIED. Card moves to review with recommendation KEEP
OPEN until the cut lands and the stale branches are retired.
