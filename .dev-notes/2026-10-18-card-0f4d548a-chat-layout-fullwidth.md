---
id: 0f4d548a
title: "chat layout: nav bookmark tab floats over conv sidebar as sticker (z-40 + --nav-tab-protrusion gutter) + conversation column full-width"
status: done
board_id: Main
date: 2026-10-18
branch: fix/chat-layout-fullwidth
tags: [chat, layout, ui, nav, fullwidth]
---

# Card 0f4d548a — chat layout: nav bookmark tab sticker + full-width conversation column

## WHAT

Two coupled chat-layout fixes on branch `fix/chat-layout-fullwidth`:

1. **Nav bookmark tab as sticker** — the nav bookmark tab floats over the
   conversation sidebar as a sticker (z-40 + `--nav-tab-protrusion` gutter)
   instead of colliding with sidebar content.
2. **Full-width conversation column** — the chat conversation column renders
   full-width.

Touched files (worktree /tmp/opencode/land-layout): `apps/web/app/globals.css`,
`apps/web/components/AppLayout.tsx`,
`apps/web/features/chat/components/layout/ChatScreen.tsx`,
`apps/web/features/chat/components/input/ChatInput.tsx`,
`apps/web/features/chat/components/messages/ReasoningPanel.tsx`.

## LINEAGE

- **Ports 8aec74365** — stranded gutter fix from card 88e4d9fb (done-but-not-on-main);
  cherry-picked with `-x` attribution.
- Relates to **card bb8212a8-resp** (chat responsive, in_progress).
- Relates to **card 2d10d992** (chat UI follow-ups).

## EVIDENCE (2026-10-19)

- PR #163 merged — merge sha `858ee59d30acca3247464727c89fcbf13b172612`, branch `fix/chat-layout-fullwidth`.
- Ports 8aec74365 (cherry-pick 741012232).
- Gates: eslint 0; tsc 2 pre-existing (byte-identical main); vitest 89/89 targeted; CI Frontend unit tests / TypeScript / Next.js build / lint / E2E / test-web / test-strui all pass. test-core-py failure pre-existing on main (runs 39143377985, 39139690771).
- Headless verification: hitTest z-40 true; hover 18.39→22.36px; gutter 18.4px / 274.39px; composer 1147.6 / 1197.6px; max-w-4xl count 0.
