---
id: 20260929_journey_gates_conda_migration
title: "Journey test gates + conda-not-venv migration (branch fix/journey-test-gates)"
status: done
date: 2026-09-29
tags: [journeys, conda, test-gates, setup, branch]
---

# Journey gates + conda-first migration

Worktree `/home/mana/Documents/Default Project/sloughGPT/.wt-gates`,
branch `fix/journey-test-gates` (started from `045ed9c21`, fast-forwarded
onto `main` @ `9871d0c2f` after the journeys library landed). 47 files
changed, **uncommitted — waiting on the user's commit go-ahead**.

## A. Journey test gates

- Root `conftest.py`: repo paths `insert(0, …)` instead of `append` —
  the conda-installed `downcraft` was shadowing `packages/downcraft`.
- `tests/__init__.py` (new): makes repo `tests/` a regular package so it
  beats the stray user-site `tests` package (252K in
  `~/.local/lib/python3.12/site-packages`).
- `packages/core-py/tests/test_user_journeys.py`: module `pytestmark`
  skips the whole file unless web (:3000) **and** api (:8000) answer —
  offline suite stays green instead of 157 errors, and it flips on the
  moment a stack comes up.

## B. conda-not-venv migration (one resolver, used everywhere)

- **`scripts/python`** (new): `SLO_PYTHON` → conda env `sloughgpt` →
  `.venv` → `python3`; `--resolve` prints the path; always prepends the
  repo root to `PYTHONPATH` (needed: `domain` is not installed).
- Rewired to it: `run.sh` (conda activation first), `scripts/dev-stack.sh`
  (module-form `-m apps.api.server.main` — file-path launch dies on
  `domain.shared` import before the path bootstrap), gateway/e2e/7 journey
  shell scripts, `.test-driver.py`, lint-staged in `package.json`.
- `scripts/setup.sh`: `USE_CONDA=auto` (conda when present, `--venv DIR`
  opts out), `conda create … python=3.12`, generated `start.sh`/`dev.sh`/
  `test.sh` exec through the resolver.
- `apps/cli` `setup` command: conda-first (create/reuse `sloughgpt`,
  `conda run -n … pip`), `--venv` keeps the old path.
- `Makefile`: `PY := $(CURDIR)/scripts/python` + `ENVBIN` for console
  scripts (chargectl units point at the env bin).
- Docs truth-up: `AGENTS.md`, `QUICKSTART.md`, `DEVELOPER_GUIDE.md`
  (both setup blocks), `STRUCTURE.md`, `USER_JOURNEYS.md` + 20 test/
  script docstrings (`scripts/python -m pytest …`).

## Gates

- ruff clean on every touched file; `bash -n` on every touched shell.
- `--collect-only`: **45771/46481, 0 errors** (post-merge).
- `apps/cli` suite **422 passed**; `test_shared.py` 76 passed;
  `tests/test_ux_flows_library.py` 5 passed.
- Journey slice offline → 58 pass / 99 skip; live `scripts/python
  -m domain.journeys --list` → all 13 spec-linked flows; live sweep
  **13/13, 0 network errors** (pre-merge).
- `make test-repo-root` etc. **not executed** — `make` is not installed
  on this box (Makefile verified statically only).

## Notes / decisions taken

- Dropped my own `docs/UX_FLOWS.md` runner note on merge: main's
  `9871d0c2f` already documents `python -m domain.journeys` (one runner,
  per the card-078 handoff) — no second doc for the same thing.
- Card 077 (kvq flake) was fixed in parallel by the sibling session
  (`2f87f5aee`, landed on main); my duplicate fixture edit was reverted.

## Open

1. ~~Commit decision for `fix/journey-test-gates`~~ — **RESOLVED: user signed off 2026-10-03; landed (see Landed below).**
2. Obsolete worktree `.wt-ux` / branch `feat/journey-consolidation`
   (based on `11d2fe49a`, 13 behind) — delete on approval.
3. `make` availability — install into the conda env?

## Coordination (2026-10-03, gate session)

main repointed 3 suites `:3000` → `:5173`
(`BASE = os.environ.get("SLO_WEB_URL") or "http://localhost:5173"`,
commit `de2fe6cf4` lineage — code preserved by `6b74a4024`): gate9's 118F
route-smoke class was a **stale port**, not a missing server (vite = `:5173`,
live stack up).

- **Rebase conflict heads-up**: your branch changes the same BASE lines —
  take main's value (`:5173`, keep your `SLO_WEB_URL` env support).
- Your module-level server `skipif` + `test_hf_*` rework is **still wanted**:
  post-port-fix live census = 95P/11F, and 7 of the 11 (datasets-import
  dialog + `Kaggle: Download from Kaggle` radio) are exactly your `test_hf_*`
  rework. The offline-skip machinery covers the no-server case (58 pass /
  99 skip slice already validated on your side).
- Residual non-yours: 2 `TestToolsFlows` selectors, 1 Vite-proxy API-URL
  assert, 1 training-config-elements assert — see card `52fb7e5b`.

## Landed (2026-10-03, user sign-off)

Merged to main as `e081ea9f4` (46 files, 511+/213−). Conflict resolutions:
- `test_user_journeys.py`: this branch won wholesale — restored base's `hf-*`
  names (main's `952e59571` preserve-commit had regressed them to `kaggle-*`),
  kept this branch's module `skipif` server gate, `_web_is_ready`,
  dialog-scoped import waits and `SMOKE_ROUTES`; kept main's `:5173` port
  default + 4 training sub-routes. Page-level import trigger stays
  **"Add file"** (committed `datasets/page.tsx:83` — main's transient passing
  "Import" state came from another session's since-reverted dirty tree).
- `.kanban/board.jsonl`: main's chain-sealed board (this branch carried only
  planner test-artifact cards + a duplicate TooltipTrigger card).

Validation post-merge: ruff clean (25 changed py), mypy = exact baseline
(854/102), `bash -n` clean (13 shells), `scripts/python` resolves to the
conda env (`.../envs/sloughgpt/bin/python`), TestDatasetsImport 7/7 live.

**Insight for the next pass (root cause, not symptom):** the import cluster
failed twice today from the same root cause — human-readable button labels
used as test hooks (`Import` ↔ `Add file` churned twice in a day across a
dirty tree and a revert). Candidate next-feature: stable `data-testid` hooks
on journey-critical controls (datasets import trigger, dialog, radio group)
so label churn can never fail the gate again — feed to the journey/drift
owner when picked up.
