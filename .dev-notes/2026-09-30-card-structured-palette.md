---
id: 20260930_096
title: Structured palette: one source of truth + generated projections
status: done
sprint: 2026-09-30
tags: [design-system, strui, theme, palette]
gh: ""
created: 2026-09-30T15:05:00Z
---

# Structured palette: one source of truth + generated projections

## Ask (user, 2026-09-30)
"we need a new palette, our palettes are just the theme, not structured" + "rebuild our
website and make it so modular it is just okay to have a grid, then a palette to apply,
and not break a strict structure". Approved (expert review -> "continue"): rebuild the
**token layer**, not the website; complete all 9 themes (2 palettes + 7 auras) via a
generator.

## Evidence found during scoping
- **5 hand-synced copies** of the same values: `apps/web/app/globals.css` (runtime),
  `packages/strui/src/styles/globals.css` (storybook mirror, header says "keep in
  lockstep"), `apps/web/components/ThemeProvider.tsx` (THEMES swatch hex),
  `apps/web/lib/theme-storage.ts` (PALETTE_COLORS hex),
  `apps/mobile/src/theme/colors.ts` (mobile mirror, comment: "All values match the web
  CSS variables").
- **strui mirror is stale (pre-fix structure)**: its `.theme-*` aura blocks set surface
  tokens (secondary/muted/border/input) — the exact "palette x aura cascade bug" that
  web fixed and documented at globals.css:1078-1085; aura primary is also lost in dark
  mode (selector `.theme-blue` loses to `html.dark`), and palettes are declared AFTER
  auras so the palette wins over the accent. Storybook has no palette/aura toolbar, so
  the blocks are currently dead-but-wrong.
- **One real value drift**: mobile `info` dark = `#78AFF0`, web `--info` dark =
  `#64A5F0` (everything else matches exactly; verified by script).
- `docs/design/DESIGN_SYSTEM.md` (LOCKED) tabulates the same values again + magazine
  page showcases them — projections, not runtime; follow-up cards.

## Plan
1. `packages/strui/tokens/palette.json` — THE source: base palette (noir-violet),
   neural-precision, 7 auras, per-target token inventories/orders, mobile mapping.
2. `packages/strui/scripts/gen-tokens.mjs` (zero deps, node) — emits, into
   marker-delimited regions of both globals.css files + `apps/web/lib/theme-tokens.generated.ts`
   + `apps/mobile/src/theme/palette.generated.ts`; `--check` = byte-verify.
3. Web CSS stays behavior-identical (extracted FROM it); strui CSS is fixed to the
   documented tier order base -> palette -> aura with auras scoped to --primary/--ring.
4. Mobile info dark converges to web `#64A5F0` (documented contract).
5. TDD: `packages/strui/src/tokens/tokens.test.ts` — byte-sync, tier order, aura
   scoping, totality, mobile/swatch parity.

## Acceptance (shared-workspace)
- [x] index updated: docs/INDEX.md gains the Design section + design/DESIGN_SYSTEM.md
      entry (shared-index resolver extended to subpath refs); packages/README.md registers
      the palette source; DESIGN_SYSTEM.md carries the source-of-truth pointer
- [x] shared copy reused: zero installs; plain node script + existing vitest/tsc
- [x] coordination named: all touched runtime files verified git-clean first; no other
      session owns them; magazine page + tamagui.config.ts + DESIGN_SYSTEM tables deferred
      to follow-up cards 20260930_097/098/099 (filed — drift risk contained there until then)

## Outcome (2026-09-30)
- Landed as **3cff76b55** — 15 files (palette.json, gen-tokens.mjs, tokens.test.ts,
  marker regions in both globals.css, theme-tokens.generated.ts, palette.generated.ts,
  colors.ts/ThemeProvider/theme-storage rewired, docs + shared-index resolver fix).
- Verification: `--check` byte-syncs 4 artifacts; strui 928 / web 8153 / mobile 14+tsc;
  pytest at known-foreign baseline (12 foreign failures unchanged).
- Web CSS behavior-identical (semantic diff vs HEAD: only 2 allowlisted no-op
  declarations); strui diff is exactly the intended tier/scoping fix.
- Coordination: `packages/strui/package.json` NOT committed by this card (foreign staged
  eslint-config change in the same file) — its `tokens:build`/`tokens:check` scripts stay
  in the working tree and land with whoever commits that file. Post-commit: lint-staged
  phantom index entries cleaned (foreign staged set restored to 201).
- Known containment: syntax-highlighter `.token.*` keeps 4 palette-coincident hexes
  (bespoke code theme); tripwire test pins them at 2+2 so they cannot grow.

## Follow-up cards — FILED as
- 20260930_097 magazine in-app palette showcase -> generated data
- 20260930_098 tamagui.config.ts -> generated from palette.json
- 20260930_099 DESIGN_SYSTEM.md tables -> generated section
