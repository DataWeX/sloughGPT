---
id: 20261023_review_column_screening_flips
title: "Review-column screening: 9 cards flipped review->done with origin/main sha evidence; 6 correctly left"
status: done
date: 2026-10-23
tags: [kanban, board-hygiene, sync, review]
---

# Review-column screening: 9 flipped, 6 left (2026-10-23)

## Why

origin/main AGENT_SYNC (entry `0dcdefda1`) warned: board `review`-column state
is stale vs code — do not re-implement, flip with sha evidence. A prior session
could not edit the main checkout's board (dirty with another session's work), so
the flips landed here on the fork board instead, screened code-first against
`origin/main` (`4a25b8385`).

## Flipped to done (9) — sha evidence, all ancestor/content-verified

| card | evidence |
|---|---|
| 9d8319fb startup overlay de-stack | `StartupOverlay.tsx:160` "Stuck is its own variant" comment on origin/main |
| 83f01763 one banner surface | `AppLayout.tsx:123` "Backend health lives on ONE banner surface"; `sl-app-banner` exists only in sanctioned `GlobalBanner.tsx:23` + `globals.css:251` |
| d484f48a static hosting | `4e37b70df gateway: serve dist-vite as the document root (card d484f48d)` ancestor of origin/main |
| e26dc68c avion unify drivers | landed `b777b4384` (per closure commit `d62873e8a`, ancestor-verified); `wait_for_function` present in origin/main `packages/avion` backends |
| 24676ff2 avion event logger | landed `68aeec952`; `avion/events/logger.py` on origin/main |
| 553da7a7 avion agent loop | landed `fc56789e3`; `step_timeout` enforced in origin/main `ai/agent.py` (TimeoutError at :41, bounded awaits :521) |
| a101ee3a Doctor core probes | PR#135 `9dcee3fed` "feat(mole): system-health probe (thermal/load/uptime)" — re-homed as mole (`apps/api/server/routers/mole.py` + mole UI) |
| b2136a14 lifecycle StartupHook tests | ported `9791f0df9` "real async StartupHook handlers at the 4 test sites (card b2136a14)", ancestor of origin/main (62 passed) |
| 96d06ef3 voice orb | PR #167 `8ec274820` merged `d9761588d` "voice: hero orb as the single visual surface"; VoiceOrb in origin/main `apps/web/features/chat/components/input/{VoiceChatMode,VoiceWaveform}.tsx`. The "awaiting land call" was answered — it landed. (`2a4132c81` was the pre-merge head; the PR tip was re-rebased, sha changed.) |

## Left in review (6) — work NOT on origin/main, branches exist

| card | why it stays |
|---|---|
| 752ca443 general search | `domain/search/` absent on main; rides the window-blocked stack landing (card 2cec0811: feat/workspace-search-q + feat/core-general-search) |
| 74032dd0 gateway bandwidth | no "bandwidth" in origin/main gateway src; branch `feat/gateway-bandwidth-observability` local, unmerged |
| f2836f6c Tree->Pool rename | no `GuardPool` in origin/main pugqeep; no branch found (lives in session worktrees) |
| 8fe718f5 Pipe bounded execution | no `pipe.py` in origin/main pugqeep; branch `feat/pipe-bounded-execution` local + origin, unmerged |
| 88fe100b native Linux build port | `25cd0a369` exists but NOT ancestor of origin/main; main still ships only `libtransformer_forward.dylib` (macOS) |
| 117429ed native GEMV perf | no `gemv` in any origin/main C code (only journal/board mentions); rides the same native branch |

## Board after

todo 15, wip 32, review 6, done 906 (was done 897 / review 15).

## WIP screening (same run) — 4 wip rows closed, 28 left correctly

| card | evidence |
|---|---|
| 7d5e5a7c hygiene sweep | ratchet `bb7162041` + merge `a189d78eb` ancestors of origin/main; PEP563 future-import present in 626 origin/main domain files; origin/main `tests/test_code_hygiene.py` carries both scope halves (`test_print_only_in_exempted_stdout_channels` with "2026-10-04 sweep: 92/92" marker + the domain future-import check #5). The cited 9a1b4c492/4e7594a67 never merged — same content reached main by another path. |
| b8c37898 slot_chain crash | crash vector extinct: no `slot_chain`/`SlotChainError`/`_chain_apply` anywhere in app-planner src (the missing import cannot recur when the subsystem is gone); was a transient main-checkout breakage (2026-10-02 "present 07:45, gone 07:55"). Linked card 4b89419d not on board. |
| 7f879f56 chargectl | `packages/chargectl/` complete on origin/main (src layout, tests, systemd unit, console script); `domain/shell/_internal/charge_manager.py` gone; `system.py:70-72` GET /battery + /battery/limit + /battery/policy; `systemController.getBattery()` wired into monitoring page.tsx:177/195. |
| cb089b43 stall watchdog | origin/main `StartupOverlay.tsx:56` progress-based stall clock ("mere health updates do NOT count"), `overlay_timeout` event :96-98, test "force-dismisses a stalled known stage and raises the degraded banner". |

**False-positive avoided**: mechanical note-status sweep matched
`294818b7` in `2026-10-02-site-doctor-surfacing-doctor-api-page.md`
(status done) — but that note is card `6d0ebf37`'s and explicitly lists
`294818b7` as its in-progress owner session. wip column correct, not closed.

Left wip (correct state / owned / in-flight): training-router delegation
cluster (50675952 part 1 + c48aaaad part 2 + af19fb40 recovery — plans in
progress, no done markers), 243016c3 (no CONSCIOUSNESS_PROC on main),
8e9ee65d (no compaction code on main), 9fc4b4cf, 5bfa16e5 (no SIGTERM fix
in api), 4a1e62fa (no design record on main), 278fdeaf/6edae3a7/656023ab/
a9bc8648/ecea49c8 (notes doing/review/wip), ca647de2 (OPENCODE owns,
do-not-duplicate + verification run pending), 89c58cfb (user-gated
dedupe), 284aa214/bb8212a8/a92c4a31 (lane-owned or ride stack landing).

Board after: todo 15, wip 28, review 6, done 910.
