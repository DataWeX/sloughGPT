---
id: 20261002_web_dev_script_next_vs_vite
title: "apps/web: dev script still next dev (:3000) while live slough-web runs vite :5173 (deferred from card e47e19ee)"
status: done
date: 2026-10-02
tags: [web, migration, ports, followup]
board_id: e157e4f1-b770-4897-a06d-392bc8b0850a
board: e157e4f1-b770-4897-a06d-392bc8b0850a
hash: 3f781594d42503b983866eca2b3e05c27d4b1922cb88935b858a9db1b04a6460
---
# apps/web: `dev` script still `next dev` (:3000) while live slough-web runs vite :5173

## CONTEXT

Card e47e19ee fixed labels/docs but apps/web/package.json `dev` = `next dev`
(Next's default :3000) while the live slough-web.service runs vite on :5173;
`.next/` leftover and NEXTAUTH_* env vars remain (their labels were retargeted to
:5173 as part of e47e19ee — verify that didn't create a lie if Next paths are
actually dead). Root checkout also noted `next lint` removal in Next v16 breaking
`npm run lint` on some branches.

## TASK

Investigate the actual state (is apps/web mid-migration Next→Vite with both
wired? which one does every runner/service/scripts assume — vite) and then:

- (a) point `dev`/related scripts at the vite workflow, OR
- (b) if Next is still intentionally supported, document the two-server split so
  :3000 claims are labeled as the Next path explicitly.

Do not break Cypress (baseUrl now :5173 after e47e19ee) or slough-web.service.

## ACCEPTANCE

- `npm run dev` matches what the live stack runs OR the Next path is documented
  as an explicit alternate
- `npm run lint` story resolved or a follow-up filed
- vitest/eslint/tsc green for touched files
- no new installs (root node_modules shared)

## SHARED WORKSPACE

- index updated = only if a doc entry is genuinely needed
- shared copy reused = root node_modules/.venv only, zero installs
- coordination named = card e47e19ee (done, source), and coordinate with any
  active web-migration session via .kanban if one owns apps/web right now
  (grep board for vite/next migration cards and link them)

## COORDINATION (cards found on the board at creation time)

- `e47e19ee-9be7-4e60-84a0-f7b3a3443fdc` — retire-3000-refs (done, source of deferral)
- Vite migration series, all done: `20260923_024_vite-phase1-shim-routes`,
  `20260923_025_vite-phase2-planner-api`, `20260923_026_vite-phase3-route-codemod`,
  `20260923_028_vite-phase5-deploy-ci`, `20260923_029_vite-phase6-cutover-delete-next`
  (Phase 6 = cutover + delete Next — check whether that landed before assuming
  Next is still wired)
- LIVE cards touching apps/web: `bb8212a8` (in_progress — Tailwind content globs),
  `ux-010-react19-migration` (todo — "Align frontend to React 19 (next@16)",
  implies Next still in play), `968e7094` (todo — palettes/themes), `6109e26e`
  (magazine palette showcase)

## RESOLUTION (closed 1167ef4c0)

Both concerns were fixed on branch `feat/frontend` by `1167ef4c0` ("feat(web): sever UI
from web engine, remove Next"), verified in the `feat/display-layer` worktree
2026-10-16:

- **dev script** — apps/web `dev` = `vite`, `build` = `vite build`,
  `start` = `vite preview`. Next dep + next.config + Dockerfile deleted;
  zero `next/` imports remain (33 files codemodded onto
  `@/vite/next-compat/*`). Guarded by
  `apps/web/vite/next-compat/no-next-imports.test.ts` (1/1 passing).
- **worker.format** — `apps/web/vite.config.ts` sets `worker: { format: 'es' }`,
  the one-line fix this note predicted. `vite build` exits 0 (dist-vite
  produced) on both the root vite 7 and apps/web-pinned vite 8 toolchains.

The forward-risk this note flagged (the fix being a merge-blocker for the
Next→Vite migration) is resolved: the migration IS this branch.
