---
id: 20261114_103000_fix-chat-render-defects-gutter-fab-header-status
status: done
board: 20261114_103000_fix-chat-render-defects-gutter-fab-header-status
title: "fix: chat render defects — gutter gap, global FAB over toast, header/session split-source, Slow+green Live"
tags: ["web", "chat", "ui", "fix"]
sprint: ""
gh: null
---

# fix: chat render defects — gutter gap, global FAB over toast, header/session split-source, Slow+green Live

Root-cause verified against LIVE Vite source (curl of :3000 serves current code) + code read 2026-11-14. Four UI defects, file:line:

1. Gutter turn-number jammed into message text ('24Hello!', '26This is...'). MessageBubble.tsx:157-161 grid sm:grid-cols-[2.5rem_minmax(0,1fr)_auto] has ZERO column gap; gutter span (:164) has no pr-*. Gutter landed (da6a2209c), spacing never did.

2. ConsciousnessQuickActions violet FAB (ConsciousnessQuickActions.tsx:110 fixed bottom-6 right-6 z-[9999], mounted globally vite-entry.tsx:42) renders on /chat ON TOP of chat toast viewport (feedback/Toast.tsx:267 bottom-0 right-0 z-[300]) - screenshot: FAB covers 'Message deleted'.

3. Header/split-source: toolbar 'Untitled' + 29 msgs vs conv card derived name + 30. ChatToolbar.tsx:25 resolves by sessionIdRef snapshot vs useChatSessions.ts:242 sessions array; diverge during async sessionController.create (:115).

4. Status truth: health_flow.py:103 'Slow/Very slow' (throughput) + useLiveStatus green Live dot (connection) shown side by side in StatusBar.

Also: TrainingIndicator.tsx:55 'Training failed' sidebar-only label -> should route through GlobalBanner (banner rule).

OUT OF SCOPE (existing cards): over-refusal + duplicate replies = consciousness narrative injection (card 20261114_092948). Pink nav bookmark tab + right-margin metadata = intentional design.

VERIFY: chat vitest suites green; header name == card name; visible gutter gap; FAB not covering toast on /chat; Slow renders amber dot.

Landed 2026-11-15 — pushed `HEAD:main` as `62e889c3b` (merge `3cc5e9ff5` on `merge/notify-reconcile`, Path A ruling: our `lib/notify/` directory facade wins, 6-intent contract ported, shadow `lib/notify.ts` git rm'd + ratchet-pinned). All five defects land with it. Gates: tsc 0; notify ratchet 2/2 + facade lock 7/7; eslint 785/900 warnings 0 errors; vitest 789 files / 8018 tests; python closure 342; router manifest + TS contracts fresh; domain baseline no drift; benchmark_slonet 7/7. Main moved twice mid-session — absorbed via merges `97d72b45d` + `62e889c3b`.
