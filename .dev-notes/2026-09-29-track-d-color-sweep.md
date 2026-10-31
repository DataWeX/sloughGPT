---
id: 20260929_088_track_d_color_sweep
status: done
title: "Card 088 — Track D2/D3 color-token sweep DONE (hex 187 + rainbow 163)"
tags: [design-system, tokens, frontend]
sprint: current
gh: null
---
# Card 088 — Track D color sweep — DONE

## Commits

- **`f20105909`** — hex phase: 187 literals / 13 files. Arbitrary
  `X-[#hex]` → semantic utilities (alpha preserved); SVG+recharts →
  `rgb(var(--chart-1..5))`; phoneme 7-cat map (chart-1..5+accent+info,
  no collisions); healthColor ternary → success/warning/destructive;
  magazine brand literals → vars; invalid `hsl(var(--background))` →
  rgb(). Scanner 0 excl. exception.
- **`5df22a816`** — rainbow phase: 163 instances / 39 files. Hue-family
  map: orange→accent, violet/purple/indigo→primary, red/rose→
  destructive, green/emerald/lime→success, yellow/amber→warning,
  blue/sky/cyan/teal→info, pink/fuchsia→chart-5. Gradient from/via/to
  structure preserved; ring-offset/divide/decoration/selection/
  content/columns covered; workspace-search test assertions converted
  with sources. Scanner 0 excl. lib/utils.test + lib/cn.test examples.
- Also this card: cascade fix **`96de3239b`** (auras → primary/ring
  only + PALETTE_COLORS swatches) + committed theme tests 46/46,
  build green.

## Exceptions / notes

- ThemeProvider aura swatches (7 hex): literal per-aura preview hues
  required (mirror html.theme-* --primary) — documented exception.
- magazine `rgb(${...})` templates: data-driven previews, not hardcoded.
- 2 ignored workspace/data pages converted on disk (dir gitignored,
  unversioned) — track if that dir becomes versioned.
- Track D remaining: rgb() = data templates only (clean); `border-white/
  [0.06]`-style alpha-white utilities NOT in scanner scope (optional
  future pass to border-border/white-token).

## Gates

hex scanner 0 · rainbow scanner 0 · vitest 572 files/4787 tests
(1 flaky compare-page i18n test passes isolated — global-state leak,
known) · tsc clean · eslint 0 errors.

## Operational lessons

- Parallel sessions clobber uncommitted edits (2×): atomic
  `script && add && commit -- <paths>` chains; commit fast.
- lint-staged leaves `MM`/` M` residue: `git restore --staged
  --worktree -- <files>` (HEAD/prettier wins).
- `FILES=$(…)` newline lists break `eval`/git add — `tr '\n' ' '`.
- My commits survive their branch switches (`git branch --contains`).

## Status

done — D2+D3 complete; Track B (Slo brand copy) + Track C (DeskMode)
next in queue.
