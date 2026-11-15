---
id: 20261031_173714_noir-chat-variant-a-quiet-composer-mode-rail-subtype-chips
status: todo
board: 20261031_173714_noir-chat-variant-a-quiet-composer-mode-rail-subtype-chips
title: "Noir chat Variant A: quiet composer (mode rail + subtype chips)"
tags: ["area:web"]
sprint: ""
gh: null
---

# Noir chat Variant A: quiet composer (mode rail + subtype chips)

Moodboard tab A (Current vs Modern toggle, board approved; standing next pick after Variant B landed ea69e3cec). Target: apps/web/features/chat/components/input/ChatInput.tsx (+ ModeBar MODE_ACCENT reuse). DESIGN: mode rail on top — segmented control, label row (labels text-sm/12, selected uses MODE_ACCENT bg-primary/10 text-primary border-primary/20); subtype chips row — badge-row chips (text-[10px] uppercase tracking, selected chip tinted); floating composer — card bg with border, body placeholder (text-sm Outfit), mono kbd hints (Enter / Shift+Enter text-[10px] font-mono text-muted-foreground), docked send button (icon-only needs aria-label). Rules: Noir Violet tokens only (rgb(var(--token)) / token classes), no new hues/fonts/spacing, type scale (section 16 / body 14 / caption 12 / label 12 / badge 10), reuse MODE_ACCENT from ModeBar — no duplicate accent literal. A11y: focus-visible ring on rail/chips/textarea/send, aria-pressed on segmented + chips, kbd hints aria-hidden (visual only). Evidence: /tmp/opencode/moodboard-modern-noir.html tab A. Gates: tsc, eslint (0 errors), vitest chat closure (ChatInput tests + chat-barrel), audit greps (no violet-*/purple-* etc.). SOP: branch feat/*, benchmark n/a (UI), push HEAD:main with AGENT_SYNC entry, card done with evidence.
