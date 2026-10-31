---
id: 20261001_conda_resolver_coordination
title: "Coordination — two sessions building the same conda-first interpreter resolver (card 5445575d vs fix/journey-test-gates)"
status: todo
date: 2026-10-01
tags: [coordination, conda, toolchain, shared-workspace, duplication]
---

# Coordination: conda resolver duplication

**Status: both sides are now committed, and their card reads `done` although
nothing has reached `main`. Evidence-based split below — needs the card
owner's sign-off. No further edits from me until then.**

Raised by the user 2026-10-01: *"recheck … some agents in another session…"* —
the recheck found a live duplicate of work I had just committed.

## Where both sides stand (updated 2026-10-01 ~07:00)

| | **Mine** — `fix/journey-test-gates`, 3 commits, clean + green |
|---|---|
| commits | `d4e999b78` style (8 files) · `9f0f1eac8` conda resolver + journey gates (47 files) · `0ffba2f14` resolver delegates to Python (1 file) |
| resolver | `scripts/python` (POSIX sh, 67 lines) |
| authority | delegates to `domain.shared.find_server_python` (adopted from this card's approach in `0ffba2f14`) |
| Makefile | `PY := $(CURDIR)/scripts/python` + `ENVBIN` |
| gates | AGENTS/QUICKSTART/DEVELOPER_GUIDE/STRUCTURE/USER_JOURNEYS truth-up; `conftest.py` insert(0); `tests/__init__.py`; `test_user_journeys.py` offline skip |

| | **Theirs** — card `5445575d`, branch `feat/conda-env`, **committed `031c38b01` (32 files, amended from `d9dfb7914`)** |
|---|---|
| resolver | `scripts/pyenv.sh` (bash, 50 lines) |
| authority | delegates to `find_server_python` — same single-authority idea |
| Makefile | `PYTHON ?= $(shell ./scripts/pyenv.sh)` + `PIP` + `CHARGE_BIN` |
| Python side | `find_conda_python()`, `PROJECT_CONDA_ENV`, `_candidate_conda_prefixes()` + `domain/shared/__init__.py` export |
| call sites | `cli.py`, `auto_trainer.py`, `mobile.py`, `run_training_tests.py`, `domain/shared/_internal/utils.py` |
| full card scope | 3 CI workflows · 6 Dockerfiles · `environment.yml` · 5 docs · **`tests/test_python_env_resolution.py`** |

**Both resolvers now delegate to Python**, so the design question is settled in
favour of theirs — my `0ffba2f14` removed the candidate scan I had duplicated.

## ⚠️ Card status: `done` but unmerged (checked 07:26)

```
card 5445575d  → column `done` (updated 05:58) · assignee `''` · notes `[]`

scripts/pyenv.sh                     main: absent
environment.yml                      main: absent
tests/test_python_env_resolution.py  main: absent
find_conda_python                    main: absent
031c38b01 ancestor of main?          NO — not merged (174 ahead / 208 behind)

main head: a1ebb7f2d — no conda commit anywhere in its last 12
```

The work exists **only** on `feat/conda-env`. Either "done" means *committed on
the branch, merge later*, or the card closed early. This note was first written
06:31 and revised 07:15 — **after** the 05:58 close, so it has not been read;
`notes: []` confirms no reply.

📌 **Their worktree directory is gone.** `/tmp/opencode/sloughgpt-conda-wt` no
longer exists (`git worktree list` reports it `prunable`), so nothing is lost —
the session was 0-dirty when it closed — but the work is reachable **only** via
the branch: `feat/conda-env` → `031c38b01`. Prune the stale registration with
`git worktree prune` when convenient; the branch itself must survive.

### Acceptance criteria

| Criterion | Status |
|---|---|
| *document toolchain in `docs/INDEX.md` / DEVELOPER_GUIDE* | ⚠️ **partial** — `docs/DEVELOPER_GUIDE.md` has 5 conda mentions ✅, but **`docs/INDEX.md` has zero** and was never touched by their commit |
| *shared copy reused: one env, no per-worktree venv/conda* | ✅ on their branch — `test_single_conda_env_is_declared()` asserts `environment.yml` carries `name: sloughgpt`, and `test_shared_indexes.py`'s failure text now names the conda env instead of `.venv` |
| *coordination named: this card* | ✅ — named throughout this note |

⚠️ **`test_single_conda_env_is_declared` hard-requires `environment.yml`**, and
that file exists only on their branch — it must land *with* the file, never
one commit before it, or CI goes red on a fresh checkout.

### Left unowned at merge time

1. the **9 genuine conflicts** below surface whenever `feat/conda-env` merges
2. the **5 bare-`python3` `test-doctor*` targets** ship as-is, inside the commit now marked `done`

## Verified merge state

```
git merge-tree fix/journey-test-gates 031c38b01  →  exit 1, 88 conflicts

  genuine overlap (both edited)           :  9   ← every file we share
  their branch's other 172 commits noise  : 79   ← stale base, not real
  total                                   : 88

  their branch: 174 ahead / 208 behind main
```

Their 07:12 amend (`d9dfb7914` → `031c38b01`) added `scripts/deploy/quickstart.sh`
plus two Docker tweaks — **none of the 9 overlap files**, so the audit below is
unaffected. It was an amend, not a rebase: the parent is still `96f7a71d9`.

**All 9 overlap files conflict:** `Makefile`, `run.sh`, `scripts/setup.sh`,
`.test-driver.py`, `AGENTS.md`, `QUICKSTART.md`, `docs/DEVELOPER_GUIDE.md`,
`docs/STRUCTURE.md`, `domain/shared/_internal/paths.py`.

⚠️ **Their conda commit is stacked on `feat/systemd-user-units`, not `main`**
(174 ahead / 208 behind). 79 of those conflicts are stale-base noise —
**rebase onto `main` first** or they will be resolving unrelated systemd work.

## Losslessness audit — theirs wins 8 of 9

Diffed my additions against theirs on every overlap file:

| File | Winner | Why |
|---|---|---|
| `domain/shared/_internal/paths.py` | theirs | `find_conda_python` / `PROJECT_CONDA_ENV` is the intended public API |
| `.test-driver.py` | theirs | `PYTHON = find_server_python(...)` yields a real interpreter path; mine is the *string* `"scripts/python"`, which breaks if the wrapper is ever removed |
| `AGENTS.md` | theirs | documents `environment.yml` + the recreate command + **both** entry points; mine names only my script |
| `run.sh` | theirs | **it does activate conda** (`conda info --base` → `conda activate sloughgpt` → `.venv` fallback) — my activation block is not lost |
| `docs/DEVELOPER_GUIDE.md`, `docs/STRUCTURE.md`, `QUICKSTART.md` | theirs | my "additions" were mostly table padding produced by the format commit, not content |
| `scripts/setup.sh` | theirs | has `--conda` / `--venv` / `--no-conda` / `--conda-env` + `USE_CONDA`; only real difference is default `true` (theirs, strict) vs `auto` (mine, graceful when conda is absent) |
| `Makefile` | **blend required** | see gap below |

## ⚠️ Gap found on their side — 5 targets bypass the resolver

```makefile
# their Makefile, lines 189 / 192 / 195 / 198 / 201
test-doctor:
	python3 scripts/test-doctor.py $(ARGS)      # ← bare python3
```

Their `pre_commit` and `pytest` targets **are** converted to `$(PYTHON)`; these
five `test-doctor*` targets were missed. Mine use `$(PY)`. **Fix whichever
Makefile version wins** — otherwise the migration leaves bare `python` in place.

## ⚠️ The hidden cost of dropping `scripts/python`

```
files in my branch referencing scripts/python : 37
  also touched by them (the 9 overlap)        :  9
  NOT touched by them                         : 28   ← would go stale or break
```

**Executed — break outright:**
- `package.json` — lint-staged runs `./scripts/python -m ruff …`
- 7 × `scripts/e2e/journey*.sh` — the journey gates
- `scripts/dev-stack.sh` (2 refs)

**Documentation only** (stale, not broken): `docs/USER_JOURNEYS.md`,
`scripts/run_ux_flows.py`, and ~18 docstrings across `packages/core-py/tests/*`
and the benchmark scripts.

So "delete my script" is a **27-file repoint** landing on files their commit
never touched — not a one-line change.

## Resolution options

| | Approach | Cost | Their test | Drift risk |
|---|---|---|---|---|
| **A** | Ship `pyenv.sh` only; repoint my 28 refs | 27-file follow-up | ✅ passes | none |
| **B** | **`pyenv.sh` = authority; `scripts/python` = 3-line `exec` shim** | ~5 lines | ✅ passes | **none** — a shim cannot drift |
| **C** | Ship mine; edit their test | edits *their* file | ❌ fails | none |

**B is recommended.** Their test (`tests/test_python_env_resolution.py`
lines 107-115) hard-asserts `scripts/pyenv.sh` exists, is executable, and
prints a valid interpreter — and nothing pins `scripts/python`. Option B keeps
their test green, keeps all 37 of my references live with zero repointing, and
ships exactly **one** resolution implementation, which was the actual card-078
concern (two filenames, but no duplicated logic to drift).

## What my branch uniquely carries (zero overlap with theirs)

```
conftest.py · tests/__init__.py · test_user_journeys.py (offline skip)
7 × scripts/e2e/journey*.sh · package.json (lint-staged) · dev-stack.sh
docs/USER_JOURNEYS.md · 14 docstrings · the 8-file format commit
```

That is the journey-test-gate half — the reason the branch exists — and it
touches nothing of theirs.

## Why this matters

The one-shared-copy rule exists because duplicates cost gigabytes and hours.
Two resolvers that disagree on the canonical interpreter is precisely the
failure mode the card's own acceptance criteria ("one env", "coordination
named") are trying to prevent.

## Ask

1. Card owner picks **B** (or counters with **A**) before either side touches
   the 9 overlap files again.
2. Card owner rebases `031c38b01` onto `main` — 79 of the 88 conflicts are
   stale-base noise from stacking on `feat/systemd-user-units`.
3. Fix the 5 bare-`python3` `test-doctor*` targets in whichever Makefile wins.
4. Take `environment.yml` and the Python-side API as theirs; I stand down from
   `domain/shared/_internal/paths.py`.
5. My journey-gate half merges independently — it overlaps nothing of theirs.
