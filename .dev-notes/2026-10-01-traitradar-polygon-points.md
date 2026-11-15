---
id: 20261001_traitradar_polygon_points
title: "TraitRadar: SVG path syntax passed to <polygon points>"
status: done
date: 2026-10-01
tags: [frontend, bug, souls, svg]
board_id: 7c1bfccb-9a35-49e9-b2a2-0ede8a39559c
board: 7c1bfccb-9a35-49e9-b2a2-0ede8a39559c
hash: 970ce4000be32b3b610a84118a9522e0a20b04f0cdaa05bf4eefac7b3b238293
---

# TraitRadar: SVG path syntax passed to `<polygon points>`

apps/web/app/(app)/souls/page.tsx:128 and apps/web/components/souls/SoulList.tsx:49
assign pathD ("M24,14 L31.07…Z") to points={pathD}; console error
`<polygon> attribute points: Expected number, "M24,14 L…"` and radar fill never
draws. Fix: points={points.map(p => `${p.x},${p.y}`).join(' ')}. Acceptance:
vitest test asserting no rendered polygon points value starts with 'M'; no
console error on /souls. Branch: fix/health-payload-trim.

SHARED WORKSPACE
- index updated: no index change required (fix only — no new app/package/doc)
- shared copy reused: root .venv + node_modules, no new deps
- coordination named: card A "Health payload: drop per_tensor from /health,
  /health/detailed, /health/stream" (d7492f1a) — same branch fix/health-payload-trim

## Resolution

MERGED to main 2026-10-02.

- Commit: `2c460cbaf` "fix(health): drop per_tensor from health payloads
  (10 MB -> ~300 B); radar polygon points" (branch `fix/health-payload-trim`).
- Merge commit `e1fbcee08` (parents `952e59571` + `2c460cbaf`). Main advanced
  `a1ebb7f2d` -> `952e59571` (WIP: preserves another session's 30-file staged
  work; reversal = `git reset --soft HEAD^` per its commit message) ->
  `e1fbcee08`.
- Verification on merged tree: byte-identical to the tested branch for all
  touched paths; pytest `tests/server/test_health_controller.py` +
  `packages/core-py/tests/test_slonet_provider.py` = 138 passed; ruff clean;
  worktree clean.
- Radar fix verified by vitest 41/41 on the branch; picked up with the next
  vite reload.
