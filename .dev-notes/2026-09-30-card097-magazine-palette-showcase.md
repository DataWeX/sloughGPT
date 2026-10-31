---
id: 20260930_097
title: Magazine palette showcase -> generated data
status: done
sprint: 2026-09-30
tags: [design-system, palette, magazine]
gh: ""
created: 2026-09-30T17:52:00Z
---

# Magazine palette showcase -> generated data

Follow-up of card **20260930_096** (structured palette, landed `3cff76b55`).

## Problem
`apps/web/app/(app)/magazine/page.tsx` (in-app design-system showcase) hand-lists
the palette values it displays. Those are projections of the same values that
`packages/strui/tokens/palette.json` now owns — until this card, they are a 6th
hand-synced copy that drifts silently.

## Do
- Import the display data from a generated module (extend
  `packages/strui/scripts/gen-tokens.mjs`, e.g. emit
  `apps/web/lib/palette-showcase.generated.ts`) instead of literal arrays/objects.
- Keep `gen-tokens.mjs --check` covering the new artifact; `tokens.test.ts` gets
  the file registered in the artifact list.

## Acceptance
- [x] zero literal palette hex/values in magazine/page.tsx (62 sites extracted: 43 RGB
      triples in lightColors/darkColors/accentThemes, 6-entry dark demo array, 13
      comma-form rgb() style literals)
- [x] `--check` verifies the new generated artifact (6th: palette-showcase.generated.ts)
- [x] web tests green (tsc + magazine 36; strui 934 incl. 3 new contract tests)

## Outcome (2026-09-30)
- Commits `06c943547` + `8ec057f28` (render fix: lint-staged prettier had corrected the
  committed aura rows to 2-space unquoted, so the generator now emits that form directly).
- Editorial labels/descriptions stay hand-written in page.tsx; values resolve from
  `SHOWCASE_TOKENS`/`SHOWCASE_AURAS`. Verified all 62 sites resolve to their original
  values (behavior identical; pre-change drift = zero).
