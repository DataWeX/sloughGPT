---
id: 20261111_110106_featdeps-engineering-library-landscape-build-time-generated
status: done
board: 20261111_110106_featdeps-engineering-library-landscape-build-time-generated
title: "feat(deps): engineering library landscape — build-time generated artifact from both trees manifests (no endpoint earned)"
tags: ["viz", "deps", "codegen"]
sprint: ""
gh: null
---

# feat(deps): engineering library landscape — build-time generated artifact from both trees manifests (no endpoint earned)

USER RULING 2026-11-11 ("visualizing... those libraries we use in the engineering" — second subject of the "1 and 5" ruling; companion to the test-runs card).

QUESTION THIS SURFACE SERVES (intake must state it): which libraries does sloughGPT depend on across BOTH trees (python: conda env/pyproject/requirements; js: root + workspace package.json manifests), at what versions, where duplicated/drifted. Today the answer is hand-grepping manifests.

DESIGN (lean-first, boundary rule):
- Data: BUILD-TIME generated static artifact (deps.gen.ts-style), like the contracts.gen pattern — manifests are read in the same process as the build, so NO HTTP endpoint is earned (boundary test: same process → function call; generation script under scripts/ + freshness --check gate, mirroring gen_ts_contracts).
- Surface: TBD at intake between inline card on the existing /developer surface vs operator page tagged role:'operator' — decide at card time per the surface ladder; NO default new page (Page Rule: needs ux/ journey citation or operator persona task from USER_PERSONA.md).
- v1 scope: inventory + versions + duplicate-across-trees highlights; NOT a vulnerability feed (no network at build).

OWNED PATHS: scripts/gen_deps_artifact.py (or equivalent), generated artifact file(s), one UI card, tests + freshness gate.

GATES: generator --check freshness gate, vitest card tests, tsc --noEmit, eslint, change-scoped tests.

MUST NOT TOUCH: package-lock.json semantics (read-only), conda env files, existing contracts.gen outputs (unless the chosen pattern extends gen_ts_contracts — then that script only), notification surfaces.

DoD: one view answers "which libraries do we use, which versions, where do the two trees disagree"; gate fails when a manifest changes and the artifact is stale.

CLOSED 2026-11-11 (OPENCODE viz/deps-landscape lane) — pushed branch `feat/deps-engineering-library` (4350fe47d impl 8 files, 02078ff39 AGENT_SYNC; remote tip 02078ff39, 0 ahead; lane-only, NOT merged). Boundary test first: manifests are read in the same process as the build → NO HTTP endpoint earned; the tab renders a static import and a fetch-spy test pins zero runtime fetches. scripts/gen_deps_artifact.py (stdlib only: tomllib/json) reads 19 manifests — root pyproject (project+optional), requirements.txt, packages|apps pyprojects, root + every apps/** and packages/** package.json (node_modules excluded; setup.py has no install_requires → skipped) — PEP 503-style normalization on BOTH trees (React==react) for grouping; kinds runtime/optional/dev/peer; groups: duplicates (≥2 manifests), drift (≥2 distinct specs), cross_tree (both trees). One output apps/web/lib/deps.gen.ts, byte-deterministic (sort_keys + explicit entry sorts) so --check can never flap; generator owns the bytes → .prettierignore entry (lint-staged requote would flap the gate). ANSWER: 196 entries · 39 duplicates · 14 drift · 0 cross-tree — typescript 5 specs, pytest 4, zustand 3, react 2, eslint 2. Surface ladder: 8th tab on the EXISTING /developer (no new page.tsx; page.test asserts tabs by name not count); wiring assertion added pinning the Dependencies tab. TDD: RED both sides → GREEN python 13/13 (subprocess --check freshness, determinism, artifact shape, prettierignore coverage, gate-discriminates via synthetic tree, dup/drift/cross-tree fixture detection, spec-parse parametrize incl markers/extras, PEP 503 normalize) + frontend 5/5 + developer page 13/13 (incl new wiring assertion). GATES: generator --check ✓ 196/39/14/0; tsc 0; eslint 0; npm run test:changed 10 files/129 exit 0. Full-record evidence shared with the sibling test-runs card (same worktree, one run_affected run: 48,360 passed / 13 failed, all triaged, zero attributable).
