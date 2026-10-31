---
id: a7bb38fd
title: "SUPERSEDES 92af1aa0: uxjourneys goes to tests/ux_journeys — packages/uxjourneys REJECTED; feat/uxjourneys-rehome must not land"
status: doing
board_id: Main
date: 2026-10-16
tags: [uxjourneys, land, git, journeys]
---

# Card a7bb38fd — land the rehome (claim)

Claimed 2026-10-16 by the foreman session for a Gene batch alongside
7aa8fa46 (session descriptor).

## Foreman pre-screen (done at claim time — port-don't-rebuild + merge strategy)

- Rehome work exists ONLY as `2beb966b0` on `refactor/uxjourneys-rehome`
  (single commit, 20 files, +294/-525, worktree /home/mana/.cache/wt-uxjourneys).
- `feat/uxjourneys-rehome` @ `44b3a6754` = the REJECTED packages/ design —
  must not land. `.wt-gpuops` already stood down (now on
  `test/planner-delete-finality`).
- `feat/rehome-uxjourneys` = already absent from refs (verified
  `git rev-parse` MISSING) — delete step pre-done.
- `test/journey-content-gate` @ `79b788b95` lives in active worktree
  /home/mana/.cache/wt-content-gate — NOT touched by this card; its owner
  lands it AFTER the rehome so rename detection carries its flows.py edits
  onto tests/ux_journeys/flows.py (card mandates this order).
- **Merge rejected, cherry-pick chosen**: merge-base(2beb966b0, main) =
  67c406ee2; main...branch = 146 / 304; `git merge-tree --write-tree main
  2beb966b0` lists ~20 conflict files, and a real merge would import 304
  origin/main commits into local main (origin red baseline: card ca647de2,
  259F) — pre-empting the user-blocked split-brain decision (1f40525b).
  Cherry-pick applies only the reviewed 20-file diff.
- Adaptation expected: the commit's probe edits target
  `domain/core/_internal/mole/probes/journey.py`, but local main carries the
  probe at `domain/core/_internal/doctor/probes/journey.py`
  (mole/probes/journey.py absent on main). Port the probe changes onto
  main's path; rename source `domain/journeys/*` exists on main so the
  R092-R100 moves apply as-is.

## Status

CLOSED as landed-then-**REVERSED**. Gene landed the cherry-pick as
`5fe9d970a` on local main (gates 91p + smokes), closed card `25c31cd3`
(this claim card `a7bb38fd` was revoked from the board mid-flight and its
title now rides `25c31cd3`, closed by `eb59b3047`).

**REVERSAL**: a newer user decision landed mid-flight — "we don't need it"
— the harness is deleted entirely, not moved. Card `433ccdb9` (final state
for `92af1aa0`, reopened in_progress) bans ALL uxjourneys branches; the
delete-cut `8b77470d8` + `a2d494fe0` + `729319ee2` (gates 81p/31w/tsc/ruff)
awaits merge to main. Foreman reverted `5fe9d970a` as **`9074fad93`** so
that merge meets the exact pre-landing state (no post-landing journey-path
commits existed; revert applied clean). Post-revert gates: ruff clean on
reverted paths, 15p + 73p pytest, zero dangling `ux_journeys` refs on main.
Canonical evidence lives in journal note `2d28e5c1`.

## ⚠ SUPERSESSION ALERT (2026-10-16 ~10:00, session .wt-gpuops) — do NOT execute the cherry-pick brief as written

This card's brief (cherry-pick `2beb966b0` → `tests/ux_journeys`) is
**stale as written**. The user's decision sequence continued past this
claim — see the **FINAL OUTCOME** section of the backing note for card
92af1aa0 (`2026-10-16-card-92af1aa0-rehome-uxjourneys.md`), written by
session .cache/wt-uxjourneys:

> "we don't need another library for our journeys!" → `tests/ux_journeys`
> → **"we don't need it"** — the harness is **deleted outright**, not
> relocated. **No uxjourneys branch may land in any form**, including
> `refactor/uxjourneys-rehome` @ `2beb966b0`.

Measured at alert time (main `edbb0d4cd`): `2beb966b0`, `8b77470d8`,
`44b3a6754` — **none are ancestors of main**. Nothing has landed; the
window to stand down is still open. The deletion artifact `8b77470d8`
lives on `refactor/uxjourneys-rehome` (with `2beb966b0` beneath it in
history), so landing "the branch's net effect" is a DIFFERENT brief than
the one handed to Gene — re-brief before any merge/cherry-pick touches
main. `test/journey-content-gate` (`79b788b95`) ordering note in this
card is moot either way per the FINAL OUTCOME.
