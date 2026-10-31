---
id: 20260930_101
title: "strui: Checkbox + Radio components to unify theme"
status: done
date: 2026-09-30
tags: [strui, components, design-system, frontend]
board_id: 44c63b91-363c-42fe-9a4d-eaa14de4e223
board: 44c63b91-363c-42fe-9a4d-eaa14de4e223
hash: ad942dd53da3643857b4a9cb85284eaf5d857bedbcc18d7b698c56678c889e33
---

# strui: Checkbox + Radio components to unify theme

## Background

Board card `44c63b91` (title-only, no body, no note → unowned): the strui
component library has no canonical Checkbox/Radio, so callers roll their own
(likely `type="checkbox"` / `type="radio"` inputs with ad-hoc styling or
per-page classes). Unify: first-class strui components driven by the token
layer, matching existing strui patterns.

Coordination checked: structured-palette session (card 096, `20260930_096`)
is `done` and scoped to the token layer (palette.json + gen-tokens.mjs +
globals.css marker regions) — components are a different layer. No session
owns `packages/strui/src` components right now.

## Plan

1. Survey existing strui components (`packages/strui/src/*`) for pattern:
   prop conventions (controlled/uncontrolled, size, disabled), styling source
   (CSS modules / globals / tokens), storybook stories, test shape.
2. Find existing ad-hoc checkbox/radio usages in `apps/web` (+ storybook) to
   unify — grep `type="checkbox"`, `type="radio"`, `accent-`, custom check.
3. Implement `Checkbox` + `Radio` (+ `RadioGroup` if pattern demands) with
   states per component-state-completeness: hover, focus-visible,
   active/pressed, disabled, checked/indeterminate (checkbox), success/error
   if the library's form pattern includes them.
4. Barrel export + stories + tests (library's existing test runner).
5. Adopt in the ad-hoc call sites found in step 2 (only if diffs are safe
   and small — record any skipped sites and why).
6. Gates: strui test suite, `tsc`, design-system checks (no hardcoded colors
   — tokens only), `tests/test_shared_indexes.py` if any index touched.

## Open questions (recommendation-taken per directive)

- Adopt in apps/web call sites vs component-only: decide after survey —
  adopt where the diff is mechanical; skip sites owned by active work.
  **Resolved**: adopt — all 8 sites were mechanical swaps, 0 skipped.

## Gates

- [x] survey + pattern conformance documented
- [x] Checkbox + Radio ship with full state matrix
- [x] strui tests green + tsc clean (strui tsc caveat: pre-existing TS5101,
      proven base/toolchain issue — see Evidence; 0 errors in changed files)
- [x] design-system check: tokens only
- [x] adoption sites (or explicit skip list) recorded — 8/8, 0 skipped

## Evidence

### Survey + pattern conformance (gate 1)

Check-if-exists found the card already half-done on main: `Checkbox`,
`Radio`, `RadioGroup` exist at `packages/strui/src/components/ui/{checkbox,
radio,radio-group}.tsx`, barreled in `src/index.ts:54-59`, token-driven,
with tests. Adopted usage (reference: `app/(app)/experiments/page.tsx`):
keep the site's own `<label>` wrapper, swap `<input type="checkbox">` →
`<Checkbox>` (no `label` prop → className lands on the input), keep the
site's handlers — `onChange` still fires (component calls both `onChange`
and `onCheckedChange`), `onClick`/`disabled` pass through via spread.
So the real work = the 8 remaining ad-hoc sites, not new components.

API: `checked` / `onCheckedChange(bool)` / `onChange` passthrough /
`size: sm|default|lg` / `label` / `description` / `indeterminate` /
`disabled`; rest spread onto `<input>` (native `name`/`value` preserved
for radios).

### State matrix (gate 2)

- `checkbox.test.tsx` — 13 tests: role=checkbox, unchecked default,
  defaultChecked, click→toggles+fires onCheckedChange, onChange alongside,
  controlled by checked, disabled attribute, label+description, indeterminate
  property, size classes, checkmark SVG when checked, indeterminate dash.
- `radio.test.tsx` — 13 tests: role=radio, default/defaultChecked,
  onCheckedChange on click, controlled, disabled, label+description,
  size, radiogroup role, multiple radios + onValueChange, vertical/horizontal.
- CSS states in source: `focus-visible:outline-none` + ring, hover borders,
  `disabled:cursor-not-allowed`, `disabled:hover:border-input`, checked via
  data-state — full component-state-completeness matrix present.

### Adoption (gate 5) — 8 sites, 0 skipped

All in `.wt-strui`, branch `feat/strui-checkbox-radio`:

1. `app/(app)/workspace-settings/page.tsx` — Allow data sharing
2. `app/(app)/vm/page.tsx` — Debug toggle
3. `app/(app)/consciousness/test-runner/page.tsx` — suite enable checkboxes
   (disabled while running)
4. `app/(app)/consciousness/history/page.tsx` — selection checkboxes
   (both stopPropagation handlers preserved: onClick via spread, onChange native)
5. `components/workspace-settings/WorkspaceLimitsCard.tsx` — allow sharing
6. `components/vm/VMControlBar.tsx` — Debug toggle
7. `components/export/ExportTemplateCard.tsx` — Tokenizer (`size="sm"`)
8. `app/(app)/personality/PersonalityQuiz.tsx` — answer radios
   (`Radio`, name/value kept via spread; no test file, not imported elsewhere)

Sweep: `grep -rn 'type="checkbox"|"radio"' apps/web --include=*.tsx
| grep -v .test` → **NONE**. Test-mock files legitimately render raw
inputs inside their strui mocks.

### Gate results (gate 3 + extras)

- **strui suite**: 89 files / 931 tests pass — exact baseline (strui source
  untouched: `git status packages/strui` empty).
- **strui tsc**: fails TS5101 (`baseUrl` deprecated) — **pre-existing
  toolchain drift**: shared `node_modules/typescript` = 6.0.3; main only
  "passes" because `packages/strui/node_modules` holds a stale local tsc
  5.9.3 that a fresh checkout doesn't reproduce. Identical tsconfig in both
  trees. Not card-owned (no tsconfig/strui-source change here).
- **apps/web affected tests**: 7 files / 117 tests pass; the 3 component
  tests needed `Checkbox` added to their `vi.mock('@sloughgpt/strui')`
  factories (rendered as a real `<input type="checkbox">` so
  `getByRole('checkbox')` queries keep working). 29/29 re-run after
  `_size`/`_onCheckedChange` unused-var cleanup.
- **apps/web tsc --noEmit**: 7 errors — all in files this card never
  touched (`training/page.tsx` ×2, `v86-controller.ts` ×4,
  `vite.config.ts` ×1). Pristine stash control → **identical 7 errors**.
  0 errors in the 12 changed files.
- **eslint** (11 changed tsx): 3 errors, all pre-existing lines
  (`vm/page.tsx:1503` unescaped `'`; `/// <reference>` line 1 of
  VMControlBar/WorkspaceLimitsCard tests). 0 new errors; the 6 new
  unused-destructure warnings fixed with `_`-prefix.
- **design-system**: `git diff | grep '^+'` for hex/rgb/hsla → none.
  Diff removes exactly 8 raw inputs, adds exactly 8 strui components.
- **tests/test_shared_indexes.py**: 6/6 (no index touched).
- **full suite context** (828 files): 798 pass / 30 fail files — failure set
  is outside this card's blast radius: static import-graph analysis shows
  only `app/(app)/export/page.tsx` renders a changed component, and its
  failure is `useNavigate() outside <Router>` from `page.tsx:43` —
  base's committed test mocks `next/navigation` while the alias points at
  `@/vite/next-compat/navigation`; another session already has the
  uncommitted fix (` M` on main: mock target switched). `date-render-guard`
  (6 raw `new Date().toLocale*`) and Sidebar/StatusBar fail pristine via
  stash control; `vm/page.test.tsx` passes standalone ×3 (full-run flake
  under 4-fork load).

### Worktree infrastructure notes (for future sessions)

- Fresh worktrees have no `node_modules`: symlink `.wt-<x>/node_modules →
  ../node_modules` (`.wt-audit` precedent) — required for react resolution.
- Committed `apps/web/vitest.config.ts` aliases react to
  `__dirname/node_modules/react` (doesn't exist; react is hoisted to root).
  Main has the fix uncommitted (` M`): `../../node_modules/react`.
  Mirrored that fix in this worktree — gates cannot run without it.
- `npx vitest` from `.wt-<x>/apps/web` crashes with silent exit 216; invoke
  `node <root>/node_modules/vitest/vitest.mjs run …` instead (from worktree
  root npx works).
