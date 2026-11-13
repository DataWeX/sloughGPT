---
id: 20261001_journeys_out_of_domain
title: "Proposal — move journey library out of domain/ (verifier tooling, not application code): coordination for ux-flows-library / journey-library / journey-test-gates"
status: todo
date: 2026-10-01
tags: [coordination, journeys, architecture, placement, proposal]
---

# Proposal: get the journey library out of `domain/`

**Status: proposal — no code written. Waiting on owner + user sign-off.**

Raised by the user 2026-10-01: *"user journeys is for product people and team
members — what's it doing in the implementation folder of domain?"* and *"the
library is used for testing, not for building structured code — it's like
pytest."*

## Why this is a real category error

`docs/REPO_STRUCTURE_MIGRATION.md` declares `domain` = *"canonical for almost
all **application code**"*, under **Canonical paths** beside the API server,
CLI, and SDK. Journeys is the opposite: it drives a live stack and asserts on
it — a **verifier**. Test tooling nested inside the shipped implementation
package is backwards (pytest is a dependency; its tests do not live inside your
product package).

### Import mechanics (verified from `cwd=/tmp`, `.venv` python)

| module | importable outside repo root? | why |
|---|---|---|
| `domain` | **NO** | not installed at all — resolves only because cwd = repo root |
| `apps` | **YES** | editable finder maps `apps*` |
| `tests`, `scripts` | **NO** | namespace packages, no `__init__.py` |
| `avion` | **NO** | never installed → `steps.py` sys.path bootstrap is load-bearing |

Two corrections I made to my own earlier claims during this audit:

1. **The `steps.py` sys.path walk is NOT evidence of bad placement.** It exists
   because avion is not installed. It would survive any move.
2. **`python -m domain.journeys` already only works with repo root on the
   path** — the editable finder does not map `domain`. So a move to `tests/`
   adds **zero** new constraint; it is the existing status quo.

The conda resolver **`scripts/python`** (from the `fix/journey-test-gates`
work) already normalises this: it prepends the repo root to `PYTHONPATH`,
noting *"(needed: `domain` is not installed)"*. Both old and new import paths
are covered by it.

## Proposed destination (user-chosen)

`domain/journeys/` → **`tests/journeys/`**

User picked `tests` over `apps/journeys/` on semantics: definitions belong with
tests (the true pytest analogy), accepting the root-path constraint that
already exists. Runner stays exactly one: `python -m tests.journeys`
(`scripts/python -m tests.journeys` under the conda resolver).

## ⚠️ This revisits a user-approved decision

`2026-09-29-card078-journeys-runner-landed-handoff.md` records the approved
call: **"One spec-linked runner: `python -m domain.journeys`"**, with
`scripts/run_ux_flows.py` explicitly superseded and `feat/journey-library`
explicitly *not* to be merged as-is (it would reintroduce a second runner).

This proposal does **not** add a second runner — but it **renames the one
approved runner**, so it needs the same owners to re-sign. It must not be
executed unilaterally.

## Hard dependency: land `.wt-gates` first

`import tests` currently resolves to a **stray user-site package** (252K in
`~/.local/lib/python3.12/site-packages`). `.wt-gates` adds
`tests/__init__.py` (present in that worktree, **absent on main**) precisely to
make `tests/` a regular package that wins.

`.wt-gates` / `fix/journey-test-gates` is now **committed — 3 commits, clean,
fully validated** (it was 48 uncommitted files awaiting go-ahead):

- `d4e999b78` format debt (8 files) · `9f0f1eac8` resolver + journey gates
  (47 files) · `0ffba2f14` resolver delegates to `find_server_python`

Validation 2026-10-01: root-suite failure list **byte-identical to `main`**
(13 pre-existing, `diff` clean), ruff `All checks passed`, mypy `Success`
(identical to main), shell syntax clean, and a live
`scripts/e2e/journey.sh validate 4570 next` → **ALL PASS** (9 assertions,
ports freed, no orphan processes, worktree clean afterwards).

**Ordering: gates → then move.** Moving first risks `tests.journeys` binding
to the wrong `tests`. That dependency is now satisfied — what remains is
sign-off, not code.

## Scope

**Must change — 16 sites:**
- 7 internal `domain.journeys.*` imports + `python -m` examples in docstrings
- `docs/FEATURES.md` ×6, `docs/UX_FLOWS.md` ×2, `tests/test_ux_flows_library.py` ×4
- index checks per shared-workspace rule (`docs/INDEX.md`, `docs/FEATURES.md`)

**Must NOT change:**
- any of the 36 `domain/*` business packages (zero touches)
- `domain/testing` — api-std owns it; main deleted that path as fake-green
  (`30fed5792`), and the spec-linked registry must not resurrect it

**Explicitly NOT bundled (separate decisions):**
- duplicate `scripts/run_ux_flows.py` (card-078 already marked superseded)
- stale `docs/USER_JOURNEYS.md` (Sep 4, product-people audience)
- zero live-journey coverage *in CI* (offline gates exist on `.wt-gates`;
  live sweep already ran **13/13, 0 network errors** on that branch)
- **the `scripts/e2e/journey*.sh` harness is orphaned (7 scripts)** — verified
  2026-10-01: *nothing* in the repo invokes them. No Makefile target, no CI
  workflow, no pytest, no `.test-driver.py`, no doc reference — a repo-wide
  grep for `scripts/e2e` / `e2e/journey` returns only unrelated
  `e2e_gateway_downcraft.sh` hits. Introduced by `5db762ce1`, whose message
  says *"All four: ALL PASS"* — i.e. run **manually** by its author, with no
  runner ever added. They work (`journey.sh validate 4570 next` → **ALL PASS**,
  9 assertions, clean teardown) but are structurally incapable of catching a
  regression, which is the same root cause as "zero live coverage in CI".
  Wiring them to a runner is a separate decision from this move.
- avion `Backend` contract (3 members) — deferred
- **broken installed `packages/voyager`** — editable-installed into the shared
  venv, sits on `sys.path`, and fails: `from arken import …` →
  `ModuleNotFoundError`. A stale duplicate of avion; violates the
  one-shared-copy rule. Needs its own cleanup.

## Coordination — required before any `git mv`

| Session | Note | Why it must be consulted |
|---|---|---|
| `feat/ux-flows-library` | author of card-078 handoff | 6 commits ahead inside `flows.py` / `steps.py` / `runner.py` |
| `feat/journey-library` | `2026-09-28-journey-library.md` | owns the handoff; mid re-targeting onto `domain/journeys` |
| `fix/journey-test-gates` | `2026-09-29-journey-gates-conda.md` | now 3 committed + validated (was 48 uncommitted files); adds `tests/__init__.py`; rewired docs to `python -m domain.journeys` |
| `sloughgpt-api-std` | staged 29-file set | owns `domain/testing/_internal/*`; do not touch |

## Ask

1. User + the three sessions re-confirm `tests/journeys/`, given it renames the
   card-078-approved runner.
2. Land `.wt-gates` (its commit go-ahead is still open) **before** the move.
3. Then: fresh worktree from `main` (⚠️ disk 96% / ~4.8 GB free) →
   `git mv domain/journeys tests/journeys` → 16-site rewrite → gates
   (`tests/test_ux_flows_library.py` 5, avion 283, ruff, mypy delta vs base)
   → commit, no push/PR unless asked.
