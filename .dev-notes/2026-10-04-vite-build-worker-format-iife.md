---
id: 20261004_vite_build_worker_format_iife
title: "vite build broken: worker.format iife vs code-splitting (soulnet-webgpu module worker) — production build path never exercised"
status: done
date: 2026-10-04
tags: [web, build, vite, bug]
board_id: 60652af2-be26-432c-8a9e-972434671aa0
board: 60652af2-be26-432c-8a9e-972434671aa0
hash: e67af4bd618be62575ecbaed7381b44dc816f93703e3d54d35a5a0c3f9140ff3
---

# vite build broken: worker.format iife vs code-splitting (soulnet-webgpu module worker) — production build path never exercised

FINDING (2026-10-04, verification sweep ordered by user): the web app's REAL production build path is broken on main. Evidence (log kept at ~/.cache/sweep-logs/vite-build3.log): `vite build` from apps/web transforms 3294 modules, then fails:
```
[vite:worker-import-meta-url] Invalid value "iife" for option "worker.format" — UMD and IIFE output formats are not supported for code-splitting builds.
file: apps/web/lib/soulnet-webgpu/index.ts
```
ROOT CAUSE: `apps/web/lib/soulnet-webgpu/index.ts:62` creates `new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })` while `apps/web/vite.config.ts` sets NO `worker` config → vite default `worker.format: 'iife'` → incompatible with code-splitting for module workers. Probable fix (one line): `worker: { format: 'es' }` in vite.config.ts — verify no regressions for other workers (index.ts:62 is the ONLY `new Worker(` site in apps/web, verified 2026-10-04).

WHY NOBODY SAW IT: two-layer blind spot — (1) the live stack runs `vite dev` only (slough-web.service ExecStart = vite), which never exercises the production bundling path; (2) on main `npm run build` in apps/web = `next build` (stale next-vs-vite path, card e157e4f1), so the npm CI path doesn't run vite build either. Verified 2026-10-04 against main (5bfbb8e08): main's ci_cd.yml has NO vite-build step; main's ci.yml runs `npm run build` = `next build`; cloudflare-pages.yml runs `npm run build` but triggers only on branch session/agent_4563443b — so no workflow on main actually runs `vite build`.

ALSO OBSERVED (same sweep): `npm run build` (= next build) fails under the worktree symlink bridge with Turbopack "Symlink [project]/node_modules is invalid, it points out of the filesystem root" — likely environment/symlink-specific, but the script it represents is the stale path (e157e4f1) anyway.

FORWARD RISK (verified 2026-10-04, read-only, subagent): the root checkout's branch (feat/pipe-bounded-execution) already carries 870ab1bad "web: finish the Next.js -> Vite migration" — it flips apps/web `build`/`build:vite` to `vite build` AND ci_cd.yml gains a "Vite build" step (`npm run build:vite`). Once that branch merges, CI WILL run `vite build` and hit this exact failure. The fix is therefore a merge-blocker for 870ab1bad, not a cosmetic one. (Sweep log's error path points at /home/mana/Documents/sloughgpt-api-std — the main checkout — with node_modules resolving through the shared root copy: the symlink bridge already in place.)

ACCEPTANCE: `vite build` exits 0 from apps/web with dist/ produced; worker behavior unchanged (WebGPU worker still loads); decide and document which build is canonical (coordinate with card e157e4f1 next-vs-vite — reference it, do not duplicate its scope); if a build step should enter CI, say so in the resolution.

SHARED WORKSPACE: root node_modules only (worktree access via symlinks already in place), zero installs; index updated = only if docs need a build-command correction (coordinate e47e19ee precedent).

COORDINATION: reference e157e4f1 (next-vs-vite, todo), e47e19ee (done, :3000 retire — cypress/dev-stack already on :5173), and the sweep context (user-ordered verification sweep 2026-10-03/04). Read-only grep for other build cards; don't modify any.

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
