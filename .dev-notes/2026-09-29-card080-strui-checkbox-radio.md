---
id: 20260929_080
board: 44c63b91-363c-42fe-9a4d-eaa14de4e223
title: "Card 080 — strui: Checkbox + Radio components to unify theme"
status: done
date: 2026-09-29
tags: [strui, design-system]
---

# strui: Checkbox + Radio components to unify theme

## Background

Add Checkbox and Radio (+group) components to packages/strui following switch.tsx patterns, Noir Violet tokens, full states (hover/focus/active/disabled/loading), tests. Ask 2026-08-27: 'do our checkbox and radio buttons to unify our theme of design and style'. Currently only switch.tsx exists.

## Source
Mined from stale opencode sessions (Aug 26-31 2026) at user's request: ~/.cache/slog-journeys/session-exports/.

## Landed (2026-09-29, branch `feat/strui-adopt-checks`, worktree `.wt-adopt`)
- **Components already existed** (landed earlier on main): `packages/strui/src/components/ui/{checkbox,radio,radio-group}.tsx`
  exported from `src/index.ts` with tests — verified green (checkbox 13, radio 13, switch 12 = 38 passed).
  The card's "only switch.tsx exists" premise was stale.
- **Adoption done** — 8 raw `<input type="checkbox|radio">` sites migrated to strui `<Checkbox>`/`<Radio>`
  (drop-in: kept `checked`/`onChange`/`className` and the wrapping `<label>`, so the real input node and
  every existing test selector survive): workspace/settings/page, vm/page, consciousness test-runner,
  consciousness history, personality PersonalityQuiz (radio), WorkspaceLimitsCard, VMControlBar,
  ExportTemplateCard. Zero raw checkbox/radio inputs left in source files.
- **Deliberately not migrated**: ThemeSwitcher + DatasetImportDialog (`role="radio"`/`role="checkbox"`
  on button toolbars — color swatches and extension chips are correct ARIA-on-button patterns, not ad-hoc inputs).
- **Test mocks**: added the repo's canonical `Checkbox` stub to the 3 `vi.mock('@sloughgpt/strui')` blocks
  that covered migrated components (stub falls through to the component's own `onChange` via `...props`).
- **Gates**: `tsc --noEmit` exit 0; eslint on the 8 source files exit 0 (2 remaining errors are pre-existing
  `triple-slash-reference` on line 1 of untouched test files); targeted tests 29/29; **full apps/web suite
  8031 passed / 8032** — the one failure (`TokenTreeCodecCard`) is a pre-existing order-dependent flake,
  green standalone both with and without this diff.
- Scope note: worktree reuses the shared `node_modules`/`apps/web/node_modules` via symlinks (no install).
