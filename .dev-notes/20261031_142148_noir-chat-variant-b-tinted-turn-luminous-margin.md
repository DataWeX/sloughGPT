---
id: 20261031_142148_noir-chat-variant-b-tinted-turn-luminous-margin
status: done
board: 20261031_142148_noir-chat-variant-b-tinted-turn-luminous-margin
title: "Noir chat Variant B: tinted turn + luminous margin"
tags: ["design", "noir-violet", "chat"]
sprint: ""
gh: null
---

# Noir chat Variant B: tinted turn + luminous margin

Moodboard Variant B (Current vs Modern, board approved): MessageBubble user turn solid bg-primary bubble -> tinted card per DESIGN_SYSTEM.md:258; luminous margin gains breathing live StatusDot; gutter/margin stay mono 10px per Log Entry spec (DESIGN_SYSTEM:255/261). Type scale enforced (section 16 / body 14 / caption 12 / label 12 / badge 10). Evidence: /tmp/opencode/moodboard-modern-noir.html tab B; MessageBubble.tsx:179; 43 existing tests must stay green. Gates: audit greps, tsc, eslint, vitest.
