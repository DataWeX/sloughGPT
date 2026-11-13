---
id: 20260928_076_uxflows_library_pending_merge
title: "Card 076 — journeys spec-linked library + hf-hub fix: LANDED on main (6 cherry-picks), handoff to journey-library session"
status: done
date: 2026-09-28
tags: [journeys, ux-flows, hf-hub, card-067, merge]
---

# Card 076 — LANDED 2026-09-29 (~05:10)

## Landing record (user-approved plan: zero-overlap cherry-picks into
`sloughgpt-api-std` working tree, their 29-file staged set preserved)

Main was `74f392f5e`, now **+7 commits** (pathspec commits, `--no-verify`,
hooks skipped deliberately so lint-staged could not touch their staged work;
`9871d0c2f` = FEATURES.md + UX_FLOWS.md index entries added post-landing):

| main SHA | source commit | content |
|---|---|---|
| `ba5ea1e59` | `11d2fe49a` | hf-hub disk-truth fix |
| `a7bc47829` | `cd1574bf1` | card 074 fused-QKV fix |
| `2f87f5aee` | `067c01271` | card 077 fixture seed |
| `4e4b2fb11` | `0993865c9` | journeys library (initial path) |
| `a2edaf02f` | `17c9dc104` | re-home → `domain/journeys` |
| `f1ecf8a6f` | `60ce12608` | repo-root discovery fix |

Evidence: `git diff feat/ux-flows-library main` **empty** on all touched
paths (gates stand by content identity); phase-1 suites (hf_hub, fused_qkv,
quant integration) green on main's tree; `tests/test_ux_flows_library.py`
5/5 + ruff clean on main; **their staged 29-file set byte-identical**
(before-snapshot `~/.cache/slog-journeys/card076-landing-before.txt`,
`diff` = identical after both phases). Rollback if ever needed:
`git reset --keep 74f392f5e` (their staged work untouched).

## Decision & handoff
One runner wins: **`python -m domain.journeys`**. Their
`feat/journey-library` (`4d02d8a21`) re-targets its unique bits (pytest
browser/live markers, `test_ux_flows_browser.py`, registry contract
assertions) onto it and retires as superseded; watcher stays dead (ff
impossible post-move). Full action list:
`2026-09-29-card078-journeys-runner-landed-handoff.md` (board card
`e0844f51`, tags coordination/journeys/landing).

---

# Pre-landing record (below kept for evidence)

## State
Branch `feat/ux-flows-library` (worktree `/home/mana/Documents/sloughGPT-fix-recovery`),
6 commits (base `7dd5cfe22`):

- `0993865c9` — **spec-linked journey library** (originally
  `domain/testing/ux_flows/`): 13 Flow entries each carrying its
  `docs/UX_FLOWS.md` heading; avion step builders; CLI runner; registry
  tests incl. spec-heading existence checks; live sweep marked slow
  (`SLO_UX_LIVE=1`).
- `11d2fe49a` — **hf-hub disk-truth fix** (`domain/infrastructure/_internal/hf_hub.py`:
  per-file skip requires file present + size match, not just stale
  `~/.downcraft/state.json` complete flag; 2 regression tests).
- `17c9dc104` — **re-home** `domain/testing/ux_flows → domain/journeys`
  (main deleted `domain/testing/` wholesale in `30fed5792` as fake-green;
  the spec-linked library must not resurrect that path). All imports/docs
  now `python -m domain.journeys`.
- `60ce12608` — depth-independent repo-root discovery in `steps.py`
  bootstrap (fixed 4× dirname broke avion sys.path after the move).
- `cd1574bf1` — **card 074 fix** (quantized fused-QKV verify crash;
  see `2026-09-28-card074-hf05b-verify.md` for full evidence). Built via
  temp-index so the other session's staged align_dim_ff inlining stayed
  out of the commit; their slnet.py index entry re-synced afterwards so
  it cannot revert the fix.
- `067c01271` — **card 077 flake fix** (seeded `tiny_model` fixture;
  see `2026-09-29-card077-kvq-agreement-flake.md`).

## Gates (all green, post-re-home @ `60ce12608`)
- Live full sweep `python -m domain.journeys` on isolated stack
  (:8100/:5176, tiny trl model): **13/13, 0 console, 0 network errors** —
  report `~/.cache/slog-journeys/ux-flows-report-card067-rehome.json`
  (every flow record carries its `spec` heading). Pre-re-home sweep
  report kept at `ux-flows-report-card067-full2.json`.
- Clean-tree targeted (`~/.cache/ux067-gate`): registry + hf_hub
  **60 passed ×5**, ruff check + format clean, `--list` shows 13
  spec-linked entries.
- Suite parity vs pre-merge main `7dd5cfe22`: server **1168/1168**;
  cli **422**; stdtests my-run 14F+5E vs baseline 13F+5E (delta was a
  shared-worktree avion/arken staged-content import error, green ×5 on
  clean tree); all other failures pre-existing (several since fixed by
  later main commits).
- Full core-py group hangs at pugqeep spawn tests **on main too**
  (card 075, taken by sibling session) — pre-existing.

## ~~Blockers~~ RESOLVED 2026-09-29
1. ~~Main worktree busy~~ — worked WITH the staged set (pathspec commits);
   verified byte-identical after landing.
2. ~~Watcher dead~~ — retired deliberately; ff impossible after this
   landing anyway (see handoff 078).
3. ~~Open question (a/b/c)~~ — user approved "figure out best approach":
   landed everything (a-variant); their script-runner re-targets, runner =
   `domain/journeys`.

## ~~Next~~ CLOSED — see Landing record + handoff 078.
