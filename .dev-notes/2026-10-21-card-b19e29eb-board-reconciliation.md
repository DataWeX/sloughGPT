# Card b19e29eb — worktree board reconciliation (stale-row graveyard cleanup)

date: 2026-10-21
branch: feat/npu-caps (stack tail)
card: b19e29eb-e684-4db1-b6e4-10f06133053f (created+claimed+closed this session)

## Why

The 10th consecutive "continue building" round spent re-pickaxing the SAME
stale rows (npu-01, 90839119, f74b9e75 ×3, f5c4cf80, e0844f51, a25e490b,
20260924_031, pugqeep-06, f01852a3, b83a5788, 37325860, ... — every one
already delivered on origin/main with a close commit). The fork board was
~50% rot: unowned-todo went 75 rows → 38 after this sweep, and all 38
remaining are intentional (cleanup/lint other-lane ×16, ux-×7, My-Note ×3,
user-parked, window-blocked, design-lane lows).

## What was done (one pass, PlannerStore-only, board-only)

- **32 cards closed** with sha-cited evidence lines in their descriptions.
  Evidence classes, ancestry-tested against origin/main:
  - IN-main close or delivery: npu-01 (3fe27bea7), f74b9e75 (cfd1539f6 +
    delivery 30161ebef), a25e490b (9c846d3e8), 20260924_031_sync
    (1d95ba665), f01852a3 (fix d50dafa15), b83a5788 (93286c6fe), 37325860
    (dd83baf9a), 122109a9, 56cf354c (b53487b3b), 33773fce (5c5721b04),
    6e826226, 60652af2 (3f73ff956), 2656f858, b538ecbd (91e81243b),
    5473eb90, 13aed9ad, b22e878a (0d75f84d5), ec2ae5c1 (cb5f88622),
    7e2af9fe (73c030a39), e8fd60d4 (f8d4ee873), 8182563a (06ce18b10),
    e157e4f1 (33616ea94), f3663079.
  - config-file proof: ca167dbd — origin/main's pytest.ini literally carries
    `; ca167dbd: default per-test timeout` + timeout=120/signal.
  - close-sha + origin/main-board-agrees: 90839119 (stale-close 0321c78ec +
    base_url in config), pugqeep-06 (ed6334e17, integration-sync d195b5956,
    board not-todo), d2803a3c (7a8ffe77d self-attested ancestor-verified),
    257d310b (7433f3919 + a46ff19e3).
  - close exists despite origin/main board revert (external-restore disease,
    1eed05ad1): f5c4cf80 (9717f050c + b17dfec2d delivered-class), e0844f51
    (4fa20e313 stale-claim verdict), 08baf13f (1eed05ad1 re-applied, net-zero
    proven cf89ac6a4).
- **10 byte-identical dup-id rows collapsed** (assert-equal guard, keep
  first): 8182563a ×4→1, f74b9e75 ×3→1, 90839119/a25e490b/13aed9ad/
  ec2ae5c1/7e2af9fe ×2→1 each.
- **ccad389e note-only, STAYS OPEN**: close 140402a4e claims "landed on main
  356d73b83" but that sha is contained ONLY in chore/* branches, absent from
  origin/main, and origin/main has no shm/resource_tracker fix since 10-12 —
  genuinely unlanded. Evidence note appended instead of a false close.
- **44c63b91 untouched** (frontend lane + close 1eed05ad1 conflicts with
  origin/main still showing todo — other lane's call).
- **NO `app_planner.cli sync` run**: this fork predates cf89ac6a4 (the
  "2 moved permanent" oscillation fix), which is on a planner branch, not in
  origin/main either. Board-only moves keep notes untouched.

## Incident avoided / mini-mistake caught

- Prefix collision caught pre-commit: `20260924` matches 53 date-stamped
  ids — resolved to exact `20260924_031_sync-session-feedback-batch` by
  id-prefix + title check.
- First sweep dropped the `20260924` key from the EV dict — trait-weights
  card missed while a sibling (`..._style-mode-tone-drift`) sat already-done.
  Caught in the post-sweep todo audit, closed in a follow-up store call.

## Gates

- Post-sweep audits: unowned-todo = 38 = intended leave-set exactly;
  closed-id dup rows = 0; board loads via PlannerStore (list_cards OK);
  backup at /tmp/opencode/board.backup-b19e29eb.jsonl.
