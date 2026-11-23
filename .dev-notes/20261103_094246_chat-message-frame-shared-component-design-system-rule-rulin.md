---
id: 20261103_094246_chat-message-frame-shared-component-design-system-rule-rulin
status: done
board: 20261103_094246_chat-message-frame-shared-component-design-system-rule-rulin
title: "Chat message frame — shared component + design-system rule (ruling B, 2026-11-03)"
tags: ["frontend", "design-system", "chat"]
sprint: ""
gh: null
---

# Chat message frame — shared component + design-system rule (ruling B, 2026-11-03)

User ruling 2026-11-03 after visual review of chat: 'visual consistency please' → Direction B (shared frame both sides). PROBLEM: user messages render as rounded-full pills while assistant messages are naked text — two visual languages, no shared shape; assistant metadata (model/tok/s/timestamp) sits in a detached right gutter; message index numbers (11,13,15...) add noise. FIX: one MessageFrame component with a role prop (user|assistant) composed by BOTH MessageRenderer.tsx and MessageContent.tsx (the two renderers that must not drift). User = right-aligned primary-tinted surface, rounded-2xl (NO pills, no new hues/fonts/spacing — Noir Violet tokens only); assistant = left-aligned card surface, same radius/shape; metadata = footer inside the frame. Design-system rule goes into the frontend-design skill so review can check it mechanically; add a component test asserting both roles render through the frame (enforcement, not vibes). OPEN SUB-DECISION (resolve at card time): message index numbers — remove entirely vs hover-only; pick one, apply everywhere. OUT OF SCOPE: composer chrome (3 stacked bands — separate card), toasts/banners (own system). Gates: tsc, eslint warning-delta, vitest chat closure. Evidence: screenshot review 2026-11-03 (chat bubbles awkward).
