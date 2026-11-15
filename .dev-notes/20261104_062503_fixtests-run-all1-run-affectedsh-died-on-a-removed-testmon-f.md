---
id: 20261104_062503_fixtests-run-all1-run-affectedsh-died-on-a-removed-testmon-f
status: done
board: 20261104_062503_fixtests-run-all1-run-affectedsh-died-on-a-removed-testmon-f
title: "fix(tests): RUN_ALL=1 run_affected.sh died on a removed testmon flag"
tags: ["tests", "ci"]
sprint: ""
gh: null
---

# fix(tests): RUN_ALL=1 run_affected.sh died on a removed testmon flag

`scripts/run_affected.sh` used `--testmon-no-changes` for the run-everything-and-re-record path, but pytest-testmon 2.2.0 has no such option, so the documented full local run (AGENTS.md Test Rule clause 3) died with 'unrecognized arguments' before collecting a test. Switched to `--testmon-noselect` (deselects nothing, still collects + writes .testmondata). Verified in a fresh worktree with no .testmondata: RUN_ALL=1 --collect-only collected 2106/2174 (68 slow deselected), and a real RUN_ALL=1 run of tests/server/test_startup_routers.py passed 60/60 and wrote .testmondata. Landed on main as 883f43433.
