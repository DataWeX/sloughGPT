---
id: 20261102_025758_cigithub-ci-on-main-never-concludes-24-consecutive-cancellat
status: done
board: 20261102_025758_cigithub-ci-on-main-never-concludes-24-consecutive-cancellat
title: "ci(github): CI on main never concludes — 24 consecutive cancellations, cancel-in-progress kills every run"
tags: ["ci", "github", "infra", "flake"]
sprint: ""
gh: null
---

# ci(github): CI on main never concludes — 24 consecutive cancellations, cancel-in-progress kills every run

FOUND 2026-11-02 ~02:55 by querying the Actions API directly (gh auth is VALID: account iamtowbee, token scopes gist/read:org/read:packages/repo — the older "gh token invalid" note is stale).

THE PROBLEM: CI on main never reaches a verdict. In the ~2.5 hours before this card, main produced ONE success (00:26, which cleared only because there was a ~20 minute gap between pushes) and 24 consecutive `cancelled` runs. The 02:40 run is the clearest specimen: 7 of 8 jobs COMPLETED SUCCESSFULLY (TypeScript, Frontend unit tests, Vite build, Board integrity, Python 3.11, Python 3.12, Feature Tags) while E2E tests sat in `npm run e2e:vite` for 15+ minutes; then a newer push cancelled the whole run ("Canceling since a higher priority waiting request for CI-refs/heads/main exists"). So the per-job gates are fine and fast — Python finishes in ~1-2 min — but main is structurally unable to conclude.

ROOT CAUSE (two independent multipliers):
  1. .github/workflows/ci.yml has `on.push.branches: [main]` with NO paths filter, so bookkeeping commits ("kanban: sync ...", docs/sync) start the full suite.
  2. `concurrency: group: CI-${{ github.ref }}, cancel-in-progress: true` — correct for PRs (a new push should supersede the old one) and wrong for main (every run is a distinct commit that needs its own verdict). Combined with the repo's 1-5 minute push cadence from parallel sessions, every run is superseded before E2E finishes.

FIX (cheapest first):
  1. `cancel-in-progress: ${{ github.event_name == 'pull_request' }}` — main serializes, PRs still cancel-and-replace. This is the one-line fix that makes main verdicts exist.
  2. Add `paths-ignore: ['.kanban/**', 'docs/**', '**.md', '*.txt']` to the push trigger. A kanban sync commit does not need E2E, and right now the bookkeeping traffic is what kills the real runs.
  3. Split `frontend-e2e` into its own workflow so a slow E2E cannot hold the unit/type/python verdicts hostage; E2E status then becomes a separate required check instead of a shared cancellation domain.
  4. While in there: every actions/*@v4 in these workflows logs "Node.js 20 is deprecated ... forced to run on Node.js 24", and ubuntu-latest migrates to Ubuntu 26 (Oct 19 2026 notice). Bump action versions deliberately, in their own commit, and re-pin node-version where the toolchain actually depends on it.

NOT DONE THIS SESSION (user was wrapping up): no workflow edits, because a CI config change needs a run to validate and there was no quiet window.

HOW TO CHECK IT LANDED: after the change, `gh run list --workflow=ci.yml -L 10` on main should show `success`/`failure` per commit with no `cancelled` beyond the newest; and a docs-only commit should not start a run at all.
