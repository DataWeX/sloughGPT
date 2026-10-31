---
id: 20260930_098
title: tamagui.config.ts -> generated from palette.json
status: done
sprint: 2026-09-30
tags: [design-system, palette, mobile]
gh: ""
created: 2026-09-30T17:52:00Z
---

# tamagui.config.ts -> generated from palette.json

Follow-up of card **20260930_096** (structured palette, landed `3cff76b55`).

## Problem
`apps/mobile/tamagui.config.ts` hardcodes palette hex — e.g.
`borderColorHover/Focus/Press: '#7C52C4'` — a runtime copy of the primary that
`packages/strui/tokens/palette.json` owns (mobile colors.ts already converged to
the generated `palette.generated.ts` in 3cff76b55; this config did not).

## Do
- Emit the color fragment of `tamagui.config.ts` from `palette.json` via
  `gen-tokens.mjs` (marker region or a generated import), same pattern as
  `palette.generated.ts`.
- Prefer importing `MOBILE_COLORS` where the config shape allows; otherwise a
  generated constants module.

## Acceptance
- [x] zero literal palette hex in tamagui.config.ts (128 literals extracted; config is
      now 53 lines of glue spreading `TAMAGUI_THEMES`)
- [x] `--check` covers the new artifact (5th: tamagui-themes.generated.ts); tokens test
      extended (23 tests: no-hex, family totality, light/dark cross-projection parity)
- [x] mobile tsc + jest green; behavior identical vs HEAD (all 128 values byte-equal)

## Outcome (2026-09-30)
- light/dark (31 fields × 2) resolve from `MOBILE_COLORS` (name-match > value-exact;
  every roundtrip asserted); accents purple/pink/ocean/sunset/mint (13 fields each)
  extracted raw as `palette.json.tamagui.accents` — hand-authored mobile-only hues.
