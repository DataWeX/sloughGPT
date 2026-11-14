---
id: 20261004_playwright_user_site_pythonnousersite_journey_errors
title: "Playwright lives in user-site but PYTHONNOUSERSITE=1 hides it — 175 journey ERRORs; install into conda env + capability marker"
status: done
date: 2026-10-04
tags: [tests, env, playwright]
board_id: 6e826226-a646-4943-b9f4-2fa0e87f7c7b
board: 6e826226-a646-4943-b9f4-2fa0e87f7c7b
hash: 9b0379e14994812598b8d2c7e4bfe429b6a362deb644ebe4d6d57ea9d9c9d979
---

Evidence: verification sweep 2026-10-04 on worktree /home/mana/Documents/sloughgpt-api-std @ 5bfbb8e08 (full suite 619F/45211P/180E in 1h56m; logs in ~/.cache/sweep-logs/: backend5.log full run, detail.log+detail2b.log exception detail, full-summary.txt, failed-by-file.txt).

ROOT CAUSE (verified): `~/.local/lib/python3.12/site-packages/playwright` (1.62.0) is the ONLY copy; canonical runs (AGENTS-mandated PYTHONNOUSERSITE=1, needed because user-site holds an orphaned `tests` package shadowing the repo tests/) get sys.path without user-site → `from playwright.sync_api import sync_playwright` raises ModuleNotFoundError at fixture setup → 175E: tests/test_user_journeys.py 103E, comprehensive_training_journeys 41E, computer_use_training_integration 21E, e2e_training_trigger 10E. Journeys passed 95 as recently as de2fe6cf4 (2026-10-03 06:21) when run WITHOUT the flag.

FIX: (1) install playwright into the conda env `/home/mana/miniconda3/envs/sloughgpt` (pip install playwright + verify browser binaries in ~/.cache/ms-playwright) — INSTALL REQUIRES USER APPROVAL per shared-workspace rules, ask first; (2) add an autouse capability marker (pytest.importorskip or custom marker needs_playwright) so missing capability = SKIP with reason, never 103-error red noise; (3) long-term: purge the orphaned user-site `tests` package so PYTHONNOUSERSITE=1 can be dropped.

ACCEPTANCE: journey suites collect+run under PYTHONNOUSERSITE=1 (pass or real failures, zero collection Errors); marker skips cleanly where capability absent.

COORDINATION: reference existing journey card work (journey residual 11→1, card from de2fe6cf4's note) — do not duplicate; reference AGENTS.md PYTHONNOUSERSITE rule.

RESULT 2026-10-05 — BOTH acceptance criteria met.

ITEM 1 (install into shared conda env) — done earlier this session, now
measured. The four suites this card names, under PYTHONNOUSERSITE=1:

| suite | was | now |
|---|---|---|
| test_user_journeys | 103E | 75 collected |
| test_comprehensive_training_journeys | 41E | 41 collected |
| test_computer_use_training_integration | 21E | 21 collected |
| test_e2e_training_trigger | 10E | 10 collected |
| **total** | **175 ERRORs** | **147 collected, 0 errors** |

ITEM 2 (capability marker) — committed f7835aaf7. Root cause: only 2 of the
4 files import playwright, at scopes that turn absence into an ERROR —
module level (test_user_journeys L19) and inside the `browser` fixture
(comprehensive_training L137). The other 2 use plain urllib against
localhost and never imported playwright (their original 21E/10E were not
playwright-driven; they collect clean now regardless).

Fix uses the repo's EXISTING convention — `pytest.importorskip` is already
used ~15x (fastapi/psutil/safetensors/gguf/cv2/PIL) — so no new custom
marker was introduced. Both guards sit in the `browser` fixture, giving
per-test SKIP rather than whole-module. test_user_journeys additionally
moved `Page` (annotation-only) to `if TYPE_CHECKING` +
`from __future__ import annotations`, exactly the shape
comprehensive_training already used, with `sync_playwright` importing
inside the fixture.

Guarding the SUBMODULE `playwright.sync_api` specifically matters: the
repo's existing module-level `importorskip("fastapi")` pattern still
hard-errors when it is a submodule that is missing (verified —
"Interrupted: 1 error during collection").

Measured both paths:
- capability present: 147 collected, 0 errors (unchanged)
- capability absent (stub `playwright` package shadowing conda's):
  exit 0 + `SKIPPED [...] playwright not installed` per test — previously
  exit 2 / collection ERROR. exit code went 2 → 0.

GATES: ruff check + ruff format --check clean on both files;
tests/test_shared_indexes 6 passed; post-lint-staged re-verified.

CAUTION (cost me a near-miss): running these suites rewrites the tracked
artifact `packages/core-py/tests/test_results/user_journey_results.json`
— an intermediate run emptied it (4290 → 2 bytes). Restored to committed
state before staging. Use `--collect-only` when probing; do not full-run
that suite casually.

ITEM 3 NOT DONE — needs explicit user approval (deletes user data):
purge the orphaned `~/.local/lib/python3.12/site-packages/tests` package
so PYTHONNOUSERSITE=1 can finally be dropped. Confirmed still present
2026-10-05. Long-term item, not part of this card's acceptance.

