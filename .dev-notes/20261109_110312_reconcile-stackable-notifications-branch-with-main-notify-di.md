---
id: 20261109_110312_reconcile-stackable-notifications-branch-with-main-notify-di
status: todo
board: 20261109_110312_reconcile-stackable-notifications-branch-with-main-notify-di
title: "RECONCILE stackable-notifications branch with main notify dispatcher (f96411252) — ruling HELD: Path A 618 conflicts vs Path B proven clean"
tags: ["web", "notifications", "reconcile", "area:web"]
sprint: ""
gh: null
---

# RECONCILE stackable-notifications branch with main notify dispatcher (f96411252) — ruling HELD: Path A 618 conflicts vs Path B proven clean

DECISION HELD by user 2026-11-09 — do not start implementation without a fresh ruling. feat/stackable-notifications (pushed, tip df1d85396) and origin/main built the same notify feature two different ways from divergent bases.

COLLISION: our stack forked main at 11007f58e (2026-11-03; now 695+ commits behind) and never saw f96411252 (landed main 2026-11-07, card 20261107_064201): lib/notify.ts thin 6-intent dispatcher over ALIVE toast/banner stores, 143-file acquisition migration, notify-ratchet zero-baseline — verified on origin/main: 0 old-API (.addToast/showBanner) call sites outside stores. Our branch instead has: a6c8b6dd3 (stackable sticky/ttl/bounded-stack engine written INTO toast-store/banner-store + Toast.tsx pin rendering) and 741c5da2e (apps/web/lib/notify/ full library, BOTH stores DELETED, 274 files migrated to object-shape notify.toast(msg, {type,...}), old signatures gone) + docs fa2e1fabc + AGENT_SYNC df1d85396. 189 files migrated by BOTH lanes into different call shapes (main: positional forwarding; ours: options object).

PROBES (read-only, 2026-11-09):
- Path A (our library wins, reconcile onto main): git merge-tree origin/main HEAD = 618 CONFLICTED FILES, plus rewrite main 143 already-migrated call sites to object-shape, delete/rewrite notify.ts + notify-ratchet.test.ts, port the process-dock tier into lib/notify/, rebase 695 commits (incl. unlanded downcraft base 07a5de74c). Days of work, high regression risk.
- Path B (main wins, port only the stackable feature): git cherry-pick -n a6c8b6dd3 onto origin/main = 0 CONFLICTS (main drifted 0 commits on all 5 files since fork), tsc 0 errors, notification closure 51/51 green incl. main ratchet; addToast signature change is purely additive (6th opts?: ToastOptions param, positional params untouched). Only gap: main lib/notify.ts toast() tuple is 5-wide, does not forward opts — sticky/ttl unreachable through the handle until a ~5-line extension + test.

AGENT RECOMMENDATION: Path B. USER RULING: hold — record findings only, decide later. Branch stays pushed, NOT merged; local main untouched; probe worktrees removed.

WHEN THE RULING COMES — Path B recipe: new branch off origin/main → cherry-pick a6c8b6dd3 (proven clean) → extend lib/notify.ts toast() labeled tuple with opts?: ToastOptions + forward + extend notify.test.ts → gates (tsc 0 / eslint --max-warnings 900 / vitest closure incl. ratchet) → AGENT_SYNC entry. The 741c5da2e object-shape migration is then SUPERSEDED (its goal, zero old-API call sites, already holds on main) — keep feat/stackable-notifications for reference until the ruling, then close/delete. Path A recipe only if the user specifically wants the one-module engine architecture to replace dispatcher+stores.
