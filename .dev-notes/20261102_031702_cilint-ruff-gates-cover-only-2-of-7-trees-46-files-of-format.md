---
id: 20261102_031702_cilint-ruff-gates-cover-only-2-of-7-trees-46-files-of-format
status: todo
board: 20261102_031702_cilint-ruff-gates-cover-only-2-of-7-trees-46-files-of-format
title: "ci(lint): ruff gates cover only 2 of 7 trees — 46 files of format drift are ungated"
tags: ["ci", "lint", "ruff", "cleanup", "area:tooling"]
sprint: ""
gh: null
---

# ci(lint): ruff gates cover only 2 of 7 trees — 46 files of format drift are ungated

FOUND 2026-11-02 by a cleanup sweep (ratchet green, F401 clean, full rule set clean).

THE GAP: .github/workflows/ci_cd.yml lints only two trees —
    ruff check packages/core-py/ apps/api/server/
    ruff format --check packages/core-py/ apps/api/server/
Everything else (domain/, journeys/, packages/* except core-py, apps/cli, apps/web server code, scripts/, tests/) is ungated. That is how a genuine I001 in scripts/finetune_gpt2.py survived: it was invisible to CI and only surfaced when someone ran ruff over the whole tree by hand.

MEASURED 2026-11-02 with the full tree (domain journeys packages apps scripts, 1383 files):
  ruff check  -> CLEAN after pinning known-third-party = ["datasets"] (b9cde163e)
  ruff format --check -> 46 files would be reformatted

THE 46 ARE REAL DRIFT, NOT NEW BREAKAGE: pre-commit runs `ruff format` on staged *.py, so every file anyone touches is auto-reformatted — the untouched remainder simply fell behind over months. Distribution: domain/infrastructure/_internal (10), scripts (11), apps/cli/src (9), packages/* (6), domain/shell (5), domain/training (2), root benchmark_results.py, etc.

WHY IT WAS NOT DONE IN THE SWEEP: a 46-file mechanical reformat is exactly the mass-edit AGENTS.md forbids doing opportunistically — it touches 10+ worktree-live areas while parallel sessions push every few minutes, and it would produce a large diff with zero behaviour change at the same moment as real work is in flight.

THE FIX (one deliberate pass, quiet window):
  1. `ruff format domain journeys packages apps scripts` in a commit that contains NOTHING else, so the large diff is its own reviewable unit.
  2. Widen the CI step to the same tree in the same PR, so the drift cannot recur.
  3. Verify with `ruff format --check` over the full tree -> 0, and confirm `ruff check` stays clean.
  4. Then the gate is honest: what CI checks equals what the repo is.

NOT IN SCOPE: format-on-save churn, prettier for TS (separate toolchain), and the Node 20 / Ubuntu 26 action bumps (card 20261102_025758 follow-up).
