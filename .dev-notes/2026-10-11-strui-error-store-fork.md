---
id: 20261011_strui_error_store_fork
title: "strui error-store fork: AppLayout <ErrorPanel /> reads a store nothing writes to"
status: todo
date: 2026-10-11
tags: [strui, frontend, error-store, split-brain, dead-code]
priority: medium
---

# strui error-store fork — dead `<ErrorPanel />` + decoy `useErrorStore` export

**Status:** reported 2026-10-11, code deliberately NOT changed (user ruling:
"card it, don't change code"). Needs a product call before any fix.

## What's wrong

There are **two independent zustand stores exporting the same name**:

| | app | strui |
|---|---|---|
| file | `apps/web/lib/error-store.ts` | `packages/strui/src/lib/error-store.ts` |
| features | dedup (30s window + `count`), `recentActivity`, `totalErrorCount`, `stateEvents`, `logStateEvent` | errors only, no dedup, own inline `extractErrorMessage` |
| factory | `createStore` (vanilla) + `useStore` | `create` |
| writers | **16 files** call `addGlobalError`; `http-client.ts:783,864` write it | **ZERO** in production |
| readers | `StatusBar`, `ActivityTicker`, `ErrorDiagnosticsPanel`, `ErrorLifecycle`, `DebugOverlay` | `error-panel.tsx:138-140` |

A zustand store's identity **is** its module identity — same exported name
across two modules is two separate stores, not aliases. So:

- `apps/web/components/AppLayout.tsx:12` imports `ErrorPanel` from strui,
  `:242` renders `<ErrorPanel />`
- that panel reads **strui's** store, which nothing writes → **it renders
  empty forever** (dead mounted DOM)
- `packages/strui/src/index.ts:264` exports that store from the barrel, so the
  natural `import { useErrorStore } from '@sloughgpt/strui'` silently writes
  into a store no app surface reads — a live footgun for the next dev

## What is NOT broken

Errors are not invisible to users. The app's rich store is healthy and
displayed by `StatusBar`, `ActivityTicker`, `ErrorDiagnosticsPanel`,
`ErrorLifecycle` and `DebugOverlay`, and `GlobalBanner` correctly follows the
AGENTS.md `useBannerStore` path. Only `<ErrorPanel />` is inert.

## Evidence (measured 2026-10-11, not inferred)

- grep over `packages/strui/src` for `addError(`/`useErrorStore` outside the
  store module and its own test: **zero writers**. Only
  `packages/strui/src/components/ui/error-panel.test.tsx` calls `addError`.
- `error-panel.tsx` lines 138-140 only *read* (`s.errors`, `dismissError`,
  `clearErrors`).
- `apps/web/lib/http-client.ts:783` and `:864` do `import('./error-store')`
  (relative → the **app's** store).
- `<ErrorPanel />` appears at exactly one production call site:
  `AppLayout.tsx:242`. Its other 9 references are strui's own test.
- Duplicate detector: 10 export names overlap between `apps/web` and
  `packages/strui/src`; `addGlobalError` was **byte-identical**, the rest are
  separate components/helpers (`cn` was a second byte-identical dup — fixed in
  `8f96badc1` by re-exporting).

## Options considered (decision pending)

1. **Wire it** — make `ErrorPanel` presentational (take `errors`/`dismiss`/
   `clear` props, defaulting to its own store for standalone use) and have
   `AppLayout` feed the app store. Non-destructive, all 9 strui tests keep
   passing. Does not create a second banner — an error *list* is a different
   role from AGENTS.md's one `GlobalBanner`.
2. **Unmount** — delete the one-line `<ErrorPanel />` mount. Smallest change,
   no dead DOM, keeps the component + tests intact, trivially reversible.
3. **Unify** — move the rich store into `packages/strui`, have `apps/web`
   re-export, delete the fork. Removes the decoy export entirely, but the rich
   store imports app-only modules (`@/lib/error-utils`, `@/lib/event-kinds`)
   that would have to move too — largest blast radius.
4. **Card only** ← chosen for now.

## Constraints for whoever picks this up

- Dependency direction is `apps/web → strui`; strui must **not** import
  `apps/web` (option 1 above keeps this).
- Must not disturb the five working app-store consumers, and must not
  introduce a second banner alongside `GlobalBanner`.
- Any fix touching `packages/strui` runs strui's `error-panel.test.tsx`
  (9 tests) — option 1 keeps them green by preserving the default store path.

## Board note

Written while `.kanban/board.jsonl` held another session's uncommitted hunks.
A board edit is a whole-file pathspec commit, so it would sweep their work —
**card not materialized yet**. Title verified collision-free against all 1112
board lines, so the next `app-planner sync` will ADD this card rather than
move an existing one.
