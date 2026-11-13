---
title: "Port selected pieces of feat/remove-core-py (do not merge the branch)"
status: doing
date: 2026-10-24
tags: [python, tests, consolidation, remove-core-py]
---

# Port selected pieces of feat/remove-core-py

## Verdict (2026-10-24, subagent analysis)

(B) partially landed — do NOT merge. Merge-base f27654e54 (2026-09-25),
884 main commits of drift, 370 unique branch commits. Core intent
`packages/core-py/domains/` -> `domain/` already landed via other PRs
(518/2571 changed files byte-identical). Branch stays local, no PR.

## Headline task — IN FLIGHT: PR #202 (`feat/test-tree-consolidation`)

`packages/core-py/tests/` -> unified `tests/` root — done as a **fresh
consolidation on main** (not a rebase): 832 files `git mv`'d to `tests/core-py/`
(branch's final name was `tests/domain/`; chose `core-py` to match the
`test-core-py` job). Zero collisions (main's `tests/core-py/` held only
untracked `__pycache__` debris from a reverted attempt).

- 127 `parents[N]` depth expressions rewritten to resolve to identical dirs;
  14 touchpoints total (ci_cd.yml test-core-py job ID kept as required check +
  test-root `--ignore=tests/core-py/`, pytest.ini testpaths + collect_ignore
  5-file set folded into addopts, Makefile 5 targets, run_training_tests.py
  12 paths, test-doctor, test_all.sh, dead ruff ignore, 6 live docs).
- Gates so far: collection 42,475/42,952 zero errors (old-layout parity),
  compileall 806/806, ruff clean. CI is the merge gate (user call: push early,
  don't pre-run every suite locally).
- **Phase 2, separate card**: `apps/cli/tests` + `apps/api/server/tests` moves
  (`test_support` sibling imports, test-server coverage job wrinkles).

## Small cherry-picks (each independent)

- [x] `apps/api/server/vm_builtins.py` (+1,350 lines vs main subset) — **PR #199**
- [x] `domain/shell/_internal/vm_programs.py` (+619 lines) — **PR #199**
- [x] `docs/DOMAIN_CONSOLIDATION.md` — **PR #199** (+ INDEX.md row)
- [ ] `scripts/check_docs_api_parity.py` — **SKIP, do not port**: main has its
  own evolved line (`fcffd8918` → `c38c4e04b` create_router adaptation →
  `dac7b55be` "port to main's line"); branch copy is wired to the branch's
  `tests/api-server/` layout (413 branch-only vs 179 main-only lines).
- [ ] `tests/test_soul_path.py` — **BLOCKED, own decision first**: depends on
  the branch-only soul-identity subsystem (`classify_soul`, `soul_path`,
  sidecar, `SOUL_TIER_POLICY`, `SoulIdentity`) inside an invasive
  `slo_format.py` evolution (also touches `save_soul`/`load_soul` signatures).
  Porting the test = porting the feature; needs an evaluate-and-port call,
  not a cherry-pick.
- [ ] selective web/mobile fixes — only after diffing vs main's evolved versions

## Discard

kanban-data (`board.jsonl`, `slot_history.jsonl`) churn, `package-lock.json`
churn, vendored `apps/web/public/v86/libv86.js`, stale duplicates of files
main has since evolved (`pugqeep/*.py`, pages).
