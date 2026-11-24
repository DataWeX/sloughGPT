---
id: 20261107_112103_cixdist-parallelize-the-full-pytest-suite-test-core-py-31m52
status: wip
board: 20261107_112103_cixdist-parallelize-the-full-pytest-suite-test-core-py-31m52
title: "ci(xdist): parallelize the full pytest suite — test-core-py 31m52s is the PR queue's bottleneck"
tags: ["ci", "performance", "testing", "area:tooling"]
sprint: ""
gh: null
---

# ci(xdist): parallelize the full pytest suite — test-core-py 31m52s is the PR queue's bottleneck

# ci(xdist): parallelize the full pytest suite

**Goal:** test-core-py takes ~31m52s single-process (measured 2026-11-04) and is the long pole of every PR's matrix; E2E 26m27s, test-web 10m20s. pytest-xdist 3.8.0 is installed in the sloughgpt env but UNUSED — no `-n` flag anywhere in CI or pytest.ini. Parallelize the full suite so the merge-to-main backstop (Test Rule clause 3 — full suite stays, never weakened) gets cheaper.

**Scope:**
- Enable xdist for test-core-py (start `-n auto` or a fixed `-n 4` on the runner's cores) and measure wall clock vs the 31m52s baseline.
- Hazards to handle explicitly (why this is a card, not a one-liner):
  - **Order-dependent tests** — we've paid for these already (self_train stop order-dependence #296, learner fixture join #295, response_tracker global order). xdist distributes tests across workers and WILL surface new ones. `-p no:randomly` stays; each xdist run's distribution is deterministic per worker count only if a grouping dist mode (`--dist loadscope` / `loadfile`) is considered.
  - **Shared ports / sockets** — journey tests hit localhost:5173; parallel workers collide (the X-Slo-Web-Tree foreign-tree problem gets worse with N servers).
  - **Subprocess tests** (self_train stop fixture, benchmark e2e CLI runs) — CPU contention + timeouts.
  - **Disk**: each worker imports the tree; ~/.cache is at ~90%.
- Decide placement: pytest.ini addopts vs CI workflow flag vs scripts/ wrapper (the repo's Command Rule says flags are declared once).

**Not in scope:** the affected-closure work (card 20261104_085623 — run LESS on PRs); E2E parallelization (separate runner class); weakening the merge-to-main full run.

**Evidence/baseline:** test-core-py 31m52s (2026-11-04), 42823 passed/116 skipped typical run (2026-11-07 logs). Target: 2x+ wall-clock reduction with zero flake introduction; gate = full suite green x3 consecutive runs on the parallel config before it becomes the default.

## Pickup findings (2026-11-24)

- **Premise partly stale:** `-n auto` already landed (card 20261112_062436). Measured **1.45x** (50m19s serial → 34m32s), NOT 2x — the CI log says **"created: 2/2 workers"**: the public runner has **2 CPUs**, so `-n auto` inside one job is ceilinged. The lever that adds hardware is **N shards on N runners** (public repo = free minutes).
- Remaining scope taken: 4-way sharding + flake hazards + the ×3-consecutive-green gate.

## Landed design (pushed 79b12de96, run 45355582144)

- **Owned split, no new deps** — `tests/shard_split.py` (round-robin over the collected list: disjoint + complete by construction; spreads sibling files so one pathological file can't strand a quarter on one runner) + `tests/conftest.py` hook. `SLO_TEST_SHARD=k/n`; unset/empty = exact no-op; malformed spec = pytest exit 4 (fail loud, never silently run whole suite in one shard). Build-from-scratch rule applied to CI plumbing (vs a floating unpinned split plugin in CI's install line).
- **Matrix on `test-core-py`** — push → `[1,2,3,4]`, PR → `[1]` (PR keeps the single unsharded affected-closure run unchanged). `fail-fast: false` (a red shard must not cancel siblings — keep the evidence). Each shard runs **BOTH passes** over its quarter (kept in one job so `test_workflow_testmon_split.py`'s job-scoped pin keeps exact teeth; the ~7m-faster two-job overlap variant is an evidence-gated follow-up).
- **Coverage** — `COVERAGE_FILE=.coverage.shard{k}` per shard; pass 1 gets `--cov-report=` (4 partial percentages would read as red) + `--cov-fail-under=0` (pyproject `fail_under=50` would be inherited per shard — pytest_cov/plugin.py:270) + `--durations=25` (straggler profile; local box can't instrument). New `test-core-py-coverage` job: `coverage combine` + `coverage report` — the 50% floor is enforced on the COMBINED data; pyproject stays the single declaration.
- **Testmon single-writer preserved** — shards upload `.testmondata*` artifacts (include-hidden-files: dotfiles; glob catches WAL sidecars). New `test-core-py-record-merge`: `scripts/merge_testmon.py` (stdlib sqlite only; target = byte-copy of shard 1 so schema comes from a real recording, never a hand DDL) merges shards → ONE `.testmondata` PRs restore. pytest-testmon 2.3.1 has no merge path (verified vs wheel); `suite_execution_file_fsha` is dead DDL (zero readers/writers). **Strict verification**: row counts must equal sum (proves no loss AND no test in two shards), fingerprint union, `PRAGMA foreign_key_check`, schema `user_version` equality — output **unlinked on any violation** (an under-selecting DB would silently skip affected tests on PRs; strictly worse than no DB where run_affected over-selects — the safe direction). Cache save stays verbatim: main only, continue-on-error.
- **Needs wiring** — `build` + `benchmark` now need the two new jobs (both are main-only, matching their consumers' own `if:` — no skipped-needs trap on PRs).

**Measurements:** shard split over real collection = 5+5+4+4=18 disjoint; merge E2E green (bogus spec → exit 4; `coverage combine` "Combined 4 files", TOTAL 72% exit 0); 36 pins+new tests green; test-root closure **1944 passed** (2 pre-existing foreign-server 404s on :8000 — reproduce on pristine origin/main; they SKIP in CI, no server there). Expected wall 54–66m → **~21–26m (2.5–3x)**.

**Gate remaining:** ×3 consecutive green CI runs on main.
