---
title: "Chat transcript visual fixes: duplicate streaming indicator, premature margin metadata, double stop button"
status: done
date: 2026-10-26
tags: [frontend, chat, ui, design-system]
---

# Chat transcript visual fixes

Screenshot audit of the chat page (2026-10-26) found three real defects in the
Log Entry transcript design:

1. **Duplicate streaming indicator** — ChatInput renders a centered
   "••• Generating" band above the composer (`ChatInput.tsx` ~line 289) while
   the empty assistant turn already shows bouncing dots inline
   (`MessageContent.tsx` empty-state). Two indicators for one stream; the
   composer band also breaks the gutter/content/margin grid alignment.
   Fix: composer band renders only for `tool_call` status (unique info the
   inline dots don't carry); generating/thinking/context rely on the in-turn dots.

2. **Premature margin metadata** — MessageBubble's margin shows
   "Assistant / model / timestamp" before the first token lands, so the log
   claims a finished entry that doesn't exist yet.
   Fix: gate model + timestamp on `hasContent` for assistant turns.

3. **Double stop button in composer** — ChatInputRow renders a standalone
   destructive cancel button (`loading && onCancel`) AND ChatSendButton which
   itself flips to a destructive stop button while loading. Two red buttons.
   Fix: drop the standalone button; route the fuller server-side cancel
   (`onCancel` = chatController.cancelStream + abort) into ChatSendButton's
   onStop slot (`onCancel ?? onStop` — handleCancel is a superset of handleStop).

## Evidence (2026-10-26, landed on `feat/chat-transcript-visual-fixes` @ `6509a9015`)

All three fixes implemented + 8 new regression tests:
- `ChatInput.test.tsx` — tool_call band renders; `generating` status does NOT
  (inline turn dots own it): 8P
- `ChatInputRow.test.tsx` — exactly ONE stop control with loading+onCancel;
  `onCancel` owns the stop slot (onStop not called), `onStop` fallback when no
  onCancel; mocked ChatSendButton corrected to mirror real enabled-while-loading
  behavior: 10P
- `MessageBubble.test.tsx` — empty streaming assistant turn hides model +
  timestamp, shows both once content lands: 60P (3 files)
- Full `features/chat` suite: **138 files / 1552 tests green**; `tsc --noEmit`
  clean; eslint **0 errors** on touched files (warnings = pre-existing mock `any`).
- Re-ran 3 test files on committed bytes after lint-staged reformat: 60P.
- Not pushed/merged: worktree lineage is 145 commits ahead of origin/main
  (AGENT_SYNC stale-main trap) — landing call is the user's.

## Follow-ups (2026-10-26, `0da807028` on same branch)

- **Item 4 — stale header metadata**: ChatToolbar now renders the active
  behavior-protocol chip (e.g. "Rewrite") next to the conversation identity;
  session title stays historical (first-message naming), chip = now.
  `MODE_LABELS` derived from ModeBar's `MODES` (no drift); threaded via
  optional `ctx.mode` (ChatToolbarContext → useChatToolbarValue →
  useChatPageController). 3 new ChatToolbar tests (chip present / tracks mode
  / absent for legacy context).
- **Item 6 — image fallback**: `MessageImages` failed thumbnails → named
  "Image unavailable: <file>" `role="img"` fallback instead of bare
  broken-image glyph; 2 new tests (error swaps to fallback, healthy untouched).
- **Item 5 — GlobalBanner**: NO CODE NEEDED — verified `error-store.ts`
  `fanOut` already raises the banner for journey blockers (severity=error,
  surface page/global, dedupe key), StatusBar `errs` badge = quiet summary
  for non-blockers. Rule 16 already satisfied (duplicate-check before build).
- Gates: features/chat **138 files / 1557 tests green** (+5), tsc clean,
  eslint 0 errors; 2 affected files re-run green on committed bytes.
  `next-env.d.ts` modified by next dev (generated, left unstaged).

## Landing (2026-10-27)

- Lineage worktree branch was 246 commits diverged from origin/main
  (features/chat differed in 97 files) → cherry-picked onto fresh origin/main
  per the PR #206 precedent; conflicts resolved by keeping main's wrapper
  classes (`w-full`, plain flex) and taking only the fix hunks.
- **PR #211** (`pr/chat-transcript-visual`, commits `88c6c1460` + `b5ba7b12b`
  + docs `bde5dc5cd`) → **MERGED `daabcc33a`** 2026-10-27T07:57Z.
- PR-branch gates: features/chat **137 files / 1567 tests green** (7 new), tsc
  0 after deleting lineage stale `.next` (its generated validator referenced
  pages absent from main), eslint 0 errors.
- CI: all 16 required checks SUCCESS on both workflows (CI 12m26s, CI/CD
  32m8s); benchmark/build/deploy skipped (main-only lanes).
- AGENT_SYNC updated: entry flipped OPEN → MERGED `daabcc33a` (+CI note),
  pushed direct to main as `21a432509`.
- Lane branch `feat/chat-transcript-visual-fixes` kept locally for history;
  stale lineage not merged.
