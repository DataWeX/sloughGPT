---
id: 20260930_099
title: DESIGN_SYSTEM.md color tables -> generated
status: done
sprint: 2026-09-30
tags: [design-system, palette, docs]
gh: ""
created: 2026-09-30T17:52:00Z
---

# DESIGN_SYSTEM.md color tables -> generated

Follow-up of card **20260930_096** (structured palette, landed `3cff76b55`).

## Problem
`docs/design/DESIGN_SYSTEM.md` (LOCKED) tabulates every token's light/dark value
and the 7 aura swatches — the same data `packages/strui/tokens/palette.json` owns.
Card 096 added the source-of-truth pointer; the tables themselves are still a
hand-maintained projection.

## Do
- Generate the value tables into a marker-delimited section of DESIGN_SYSTEM.md
  from `palette.json` (extend `gen-tokens.mjs`; `--check` verifies the doc).
- Keep surrounding prose hand-written (the LOCKED rules, usage columns).

## Acceptance
- [x] table values generated; `--check` fails if doc drifts (doc is the 7th artifact;
      four `<!-- @tokens:* BEGIN/END -->` regions — light/dark/charts/auras)
- [x] docs/index pytest green (`tests/test_shared_indexes.py` = 6 passed)

## Outcome (2026-09-30)
- Commit `42743f498`. Generator owns table structure + value cells; Usage/Aura/Mood
  prose cells are parsed from the current doc and carried verbatim (kept hand-written
  as this note requires). `mdTable` emits prettier-stable padding — generator, file,
  and prettier agree byte-for-byte (round-trip verified); rendered tables were
  byte-identical to the committed ones (zero drift at extraction).
- Tokens contract now 28 tests: markers declared + doc values parsed INDEPENDENTLY of
  the generator and compared against palette.json (tripwire both ways).
