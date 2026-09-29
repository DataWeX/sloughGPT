---
name: shared-workspace
description: Use when installing packages, adding a new app/package, creating node_modules, .venv, or a package-lock.json, or running rm/npm install/pip install in sloughGPT — enforces the one-shared-copy policy and index freshness.
---

# Shared workspace policy (AGENTS.md core rules)

sloughGPT has **one** toolchain for every session, worktree, and agent:

- ONE root `node_modules/`, ONE root `.venv/`, ONE `packages/*` tree, ONE `apps/` tree.
- Reuse them. Never rebuild, never fork a private copy. A rebuild has cost us
  hours and gigabytes (e.g. a 7.5G duplicate `apps/mobile/node_modules`).
- When you fix a shared package, fix it **in the shared copy** so every
  session benefits.

## When this applies

- Before `npm install`, `pip install`, `pnpm install`, `yarn`
- Before creating any `node_modules/`, `.venv/`, or `package-lock.json`
- Before `rm -rf node_modules` (deleting shared toolchains needs approval — ask twice for user data)
- When adding a new directory under `apps/` or `packages/`

## Required actions when adding an app or package

1. Register it in the index **in the same change**:
   - app → `apps/README.md`
   - library → `packages/README.md`
   - doc → `docs/INDEX.md` (plus `docs/FEATURES.md` for features)
2. If two sessions would touch it, coordinate via a kanban card —
   build on the shared copy, never duplicate it.

## Mechanical gate

`tests/test_shared_indexes.py` (6 checks) enforces both invariants and runs:

- in CI (pytest `tests/`), and
- via `scripts/verify.sh`.

Run it yourself after structural changes:

```bash
.venv/bin/python -m pytest tests/test_shared_indexes.py -q
```

If it fails: add the missing index entry (or remove the stray install
tree) — do **not** extend the allowlists casually; each allowlist entry
needs a reason comment and ideally a card.

## Legitimate existing install trees (allowlist)

`apps/web`, `apps/mobile`, `packages/strui`, `packages/eslint-config`,
`packages/sdk-ts/typescript-sdk`, `.opencode`, repo root — plus the
single root `.venv`. `.wt-*` directories are other sessions' worktree
checkouts and are out of scope.
