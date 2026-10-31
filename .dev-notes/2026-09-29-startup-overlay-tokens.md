---
id: 20260929_079_startup_overlay_tokens
status: doing
title: "Card 079 — Track D1: StartupOverlay hex → design tokens"
tags: [design-system, tokens, startup, frontend]
sprint: current
gh: null
---
# Card 079 — StartupOverlay token conversion (Track D1)

## Scope

Convert the Apple-black boot overlay to Noir Violet tokens so the
palette/aura/mode switcher is continuous from boot (it currently paints
`#0a0a0a` iOS-blue while the rest of the app is themed).

- `components/startup/StartupOverlay.tsx` (18 hex):
  `bg-[#0a0a0a]`→`bg-background`; gradient `from-[#0a7aff] to-[#5856d6]`
  →`from-primary to-primary/60`; `#c7c7cc`→`text-foreground`;
  `#8e8e93`→`text-muted-foreground`; `#1c1c1e`/`#2c2c2e`→`bg-muted`;
  `#febc2e`→`text-warning`; `#28c840`→`bg-success`; `#0a7aff`→`bg-primary`.
- `components/startup/StartupHistoryChart.tsx` (26 hex): chart series →
  `var(--chart-N)` map (skill rule: color maps are token refs, never hex).
- `StartupOverlay.test.tsx:81-83` asserts `bg-[#28c840]` → `bg-success`.

## Branch note

Planned base was `main`, but this is a shared checkout carrying the
parallel session's 202-file staged batch — switching to main is blocked
(overwrite refusal) and would hijack their session. D1 lands on
`feat/gateway-sse-compression` instead; token conversion is mechanical
and rebase-safe.

## Gates

vitest StartupOverlay (+ history chart), eslint via lint-staged hook,
design-system-enforcement scanner: 0 hex in components/startup/.
