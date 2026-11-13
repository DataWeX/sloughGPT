---
id: aecf2bb1
title: "scripts/python repo-wins: generalize beyond downcraft (installed twins: mogdb, planner)"
status: done
board_id: Main
date: 2026-10-15
tags: [launcher, packaging, one-shared-copy, scripts]
---

# Card aecf2bb1 — closed as won't-do (2026-10-15, session .wt-gpuops)

The card's escape hatch is its own scope line: *"first grep which
scripts/python consumers import mogdb/planner (if none, close as won't-do)."*
Consumers **do** exist, so that literal trigger is not what closed it — the
**measured absence of the defect the extension defends against** is. Every
claim below was measured in the conda env the launcher resolves to, not read
off the card.

## What the card assumed vs what is true now

The card was mined from three pip snapshots shadowing repo packages:
`downcraft==0.2.0`, `mogdb==0.2.0.post1.dev359+g3ad8339a4`, `planner==0.1.0`.
Measured today:

| package | installed in env? | shadowing defect? |
|---|---|---|
| `downcraft` | **yes, 0.2.0** | yes — **already fixed** (`40970bf67`, landed via `2d427428b`) |
| `mogdb` | **no** | none — nothing to shadow |
| `planner` | **no** | none — nothing to shadow |
| `app-planner`, `chargectl`, `bawl`, `infra_lib` | **no** | none |

`pip list` shows only `downcraft 0.2.0` and `sloughgpt 0.3.0` (an editable
install pointing at the repo root, i.e. the tree copy itself). So the two
snapshots this card names have been removed from the env since it was filed.

## The consumers are not broken

The card's premise is that scripts importing `mogdb`/`planner` need the
launcher to put the repo root on the path. They do not:

- `scripts/benchmark_mogdb_sync.py` line 25 does
  `sys.path.insert(0, "packages/mogdb/src")` **before** its imports at 27-28.
  Forced a module-level load via `runpy.run_path(..., run_name="not_main")`:
  imports cleanly, no `ModuleNotFoundError`.
- `scripts/benchmark_mogdb_sync.py --help` and `scripts/migrate_boards.py
  --help` both exit 0 through `./scripts/python`.

The scripts already self-serve the tree copy. Adding a launcher entry would
not change what they import.

## The future case is already ratcheted (verified, not assumed)

`tests/core-py/test_launcher_repo_copy.py` (on main, **4 passed**) makes the
launcher list a ratchet rather than an optimisation. Read its logic:

- `test_every_installed_repo_package_is_pinned_by_the_launcher` enumerates
  every `packages/` entry that holds an importable package — **both flat
  (`packages/<n>/<n>/__init__.py`) and src (`packages/<n>/src/<n>/__init__.py`)
  layouts** — and fails if `site_pkgs/<name>` exists but `packages/<name>` is
  absent from the launcher's list.
- `test_listed_paths_exist_in_the_tree` stops the list rotting.
- `test_repo_copy_actually_wins_at_runtime` runs
  `./scripts/python -c "import downcraft; print(downcraft.__file__)"` and
  fails if it resolved under `site-packages`.

So the moment `mogdb` or `planner` is pip-installed again, that test goes red
until the entry is added — exactly the condition this card would have
pre-empted, minus the dead entries.

## Why a blanket loop would be the wrong fix

The landed fix's comment (main's `scripts/python`) says the list is
*"deliberately narrow: only packages/ names that actually have an installed
copy belong in this list"* — adding the whole set drags pytest-only entries
into every invocation. Listing packages with no installed twin adds entries
that defend against nothing, and the ratchet above would already catch a real
regression. Not doing this is the conservative, correct outcome.

## Residual finding worth a separate ask (not this card, not acted on)

`app-planner` is **not installed** in the env and has no console script, so
the `app-planner sync` command that `AGENTS.md` documents as the canonical
board-sync step does not exist on `PATH`. It only runs from source via
`PYTHONPATH=packages/app-planner/src ./scripts/python -m app_planner sync`.
That is an ergonomic gap in the documented workflow, unrelated to shadowing,
and installing anything is a separate ask (shared-workspace policy).

## Evidence

- `pip list | grep -iE "mogdb|planner|downcraft"` → only `downcraft 0.2.0`,
  `sloughgpt 0.3.0`.
- `importlib.util.find_spec` probe: `mogdb`, `planner`, `app_planner`,
  `chargectl`, `bawl`, `infra_lib` → `NOT importable` (absence, not shadowing).
- `tests/core-py/test_launcher_repo_copy.py` → 4 passed on main.
- No code changed. Card closed; queue drops one unowned todo.
