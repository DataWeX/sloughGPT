---
id: 20260924_vite-phase6-cutover-delete-next
status: done
priority: high
sprint: vite-migration
title: Vite Phase 6: cutover + delete Next
gh: —
tags: vite, next, cutover, migration
---

## Task
ROADMAP: cut compose/CI/docs over to Vite as default; remove Next runtime/build
surface (next.config, Dockerfile Next path, Next CI jobs). Keep `app/**/page.tsx`
as Vite route sources.

## Scope
1. Delete `next.config.*`, `next-env.d.ts`, Next `Dockerfile`, `proxy.ts`, app/layout.
2. Add `lib/document-meta.ts` + `vite/index-metadata.test.ts`.
3. Point package/tsconfig/eslint/CI/compose/turbo/docs at Vite only.
4. Global Link mock in `vitest-setup.ts` (`@/vite/next-compat/link` → `<a>`).
5. Harden flaky title-before-content tests (users, workspaces, audit, benchmark,
   companion, shared-data, workspace-settings, PageSkeleton, datasets, training/runs,
   multimodal, vm, usage, notifications, profile).

## Gate (green)
- `tsc --noEmit` = 0
- `eslint . --quiet` = 0
- `vitest run` = 7922/7922, 825/825 files
- `build:vite` = 0
- `e2e:vite` = 0 (20/20)

## Status
done. Do not commit unless asked.
