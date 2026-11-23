---
id: 20261107_064201_unified-notify-dispatch-handle-one-intent-api-over-bannertoa
status: done
board: 20261107_064201_unified-notify-dispatch-handle-one-intent-api-over-bannertoa
title: "Unified notify() dispatch handle — one intent API over banner/toast/process-dock tiers"
tags: ["frontend", "notifications", "seam"]
sprint: ""
gh: null
---

# Unified notify() dispatch handle — one intent API over banner/toast/process-dock tiers

User-directed (2026-11-07, garbled message resolved via clarify: 'Unified notify() dispatch'). GOAL: one notify() handle so producers raise notification intents instead of writing the three stores directly (banner-store single-slot, toast-store, process-notifications dock). MEASURED on main (2026-11-07): banner writes = 4 sites (error-store, AppLayout x2, StartupOverlay); direct .addToast() store calls = 9 sites / 8 files; grandfathered bare addToast() selections = ~724 calls / 160 files; showToast context funnels = 80 sites; useToastStore mock coupling in 130 test files. REFERENCE LANES (port the pattern, never merge): feat/display-layer + feat/cognition-callback — same concept landed 2026-10-14..17, now 1554 commits behind origin/main (unmergeable; intent-only actions + no-direct-store-imports ratchet pattern to borrow). DESIGN: apps/web/lib/notify.ts thin pure dispatcher (no state, stores stay owners); banner tier fully migrated (0-baseline allowed); toast tier shrink-only per-file baseline (grandfather clause — migrate when next touched); dock tier = ingestProcessEvents passthrough (SSE projector untouched; dispatchAction is commands, NOT notifications — stays out). RATCHET: new gate test scanning for direct write sites, baseline may only shrink (handbuilt-routes / page-baseline precedent). MUST NOT: change GlobalBanner one-banner invariant, toast rendering, dock lifecycle, or merge the stale lanes. Branch: feat/unified-notify. Gates: RED watched, tsc 0, eslint 0, test:changed, ruff n/a (TS-only), no new page.
