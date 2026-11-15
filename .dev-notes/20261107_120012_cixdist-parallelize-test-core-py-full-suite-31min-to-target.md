---
id: 20261107_120012_cixdist-parallelize-test-core-py-full-suite-31min-to-target
status: todo
board: 20261107_120012_cixdist-parallelize-test-core-py-full-suite-31min-to-target
title: "ci(xdist): parallelize the full pytest suite — test-core-py 31m52s is the PR queue's bottleneck"
tags: [ci, performance, testing]
sprint: ""
gh: null
---

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
