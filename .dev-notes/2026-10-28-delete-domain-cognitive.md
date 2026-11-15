---
title: "Delete domain/cognitive/ — dead tree (monstrosity of the cognition domain)"
status: todo
board: 58527152-03df-4b8d-bd60-aa2fcc4a54b6
date: 2026-10-28
tags: [backend, cognition, cleanup]


# Delete `domain/cognitive/`

## What

Remove `domain/cognitive/` (10 Python files) and its two self-only test
files: `packages/core-py/tests/test_cognitive_base.py`,
`test_cognitive_base2.py`.

## Why

- User ruling (2026-10-27, context-contract audit): "we only have cognition,
  not cognitive — cognitive is a monstrosity of the cognition domain."
- **Dead tree, verified 2026-10-28:** zero production importers outside the
  tree itself; the ONLY external references are its own two test files.
  The real domain is `domain/cognition/` (canonical home of consciousness
  at `domain/cognition/_internal/consciousness/engine.py`).

## Scope / risks

- Grep for `domain.cognitive` (and `domain import cognitive`) repo-wide
  INCLUDING scripts/, docs/, config before deleting — re-verify at card
  time, the audit number above is a snapshot.
- No facade, no re-exports: nothing outside imports it, so there is nothing
  to grandfather.
- Separate from this card: any *content* worth salvaging (grounding.py,
  reasoning/deep.py are cognition-lane duplicates) — check for
  divergence vs `domain/cognition/_internal/{grounding,reasoning/deep}.py`
  FIRST; delete-only unless a diff proves unique logic worth porting.

## Evidence

(to fill)
