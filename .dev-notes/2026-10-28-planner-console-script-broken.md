---
title: "Fix planner console script: entry point targets planner.cli but package is app_planner"
status: done
board: 830190a4-b675-4420-baeb-655123333df0
date: 2026-10-28
tags: [tooling, kanban, planner]


# Fix broken `planner` / `kanban` console scripts

## What

`packages/app-planner/pyproject.toml` declares:

```toml
[project.scripts]
planner = "planner.cli:main"
kanban = "planner.cli:main"
```

but the package directory is `app_planner`, so the installed entry points
die instantly:

```
$ planner verify …
ModuleNotFoundError: No module named 'planner.cli'
```

Discovered 2026-10-28 during the context-contract lane's board bookkeeping
(undocumented before then — every recent lane silently worked around it:
live-feed lane used `PYTHONPATH=packages/app-planner/src python -m
app_planner.cli …`, same workaround as this card).

## Fix shape

1. Point the scripts at the real module: `app_planner.cli:main` (check
   whether `packages/app-planner/src/planner` alias exists first — the
   Sept board cards mention `packages/planner` as the pre-rename home, so
   a compat alias may be intentional; if not, just repoint).
2. Reinstall editable (`pip install -e packages/app-planner`) in the
   `sloughgpt` env so `which planner` works — env mutation, do it when
   nothing else is mid-run (shared conda env).
3. Smoke: `planner verify 2026-10-27-context-contract-backend` → all
   consistent; `planner board stats`.

## Note

- `python -m app_planner.cli` keeps working regardless — nothing blocks
  on this card, it is UX/hygiene for every session.
- Related but separate: `migrate_planner.py` is a JSONL-store migration,
  NOT the entry point — don't confuse them.

## Evidence

Landed 2026-10-29 as **PR #236 → `18f848301`** (TDD: RED = 4 failures —
3 `planner.cli` console-script targets + `python -m planner`; GREEN after).

- `packages/app-planner/pyproject.toml` — `[project.scripts]` `planner` /
  `kanban` → `app_planner.cli:main` (pyproject wins over setup.py on modern
  setuptools, so these were the installed entry points).
- `packages/planner/setup.py` — `planner = app_planner.cli:main` (legacy
  dist keeps its command, now pointing at the consolidated CLI).
- `packages/planner/src/planner/__main__.py` — delegates to
  `app_planner.cli.main`.
- New repo-wide gate `packages/app-planner/tests/test_entry_points.py`:
  walks every console-script declaration (pyproject `[project.scripts]` +
  setup.py `entry_points` via AST), bootstraps import paths from the
  declaring dirs, resolves every target to a callable; `__main__` is loaded
  by path so a divergent editable install can't mask the repo source.
- Verification: app-planner suite 4 failed / 273 passed (the 4 are
  pre-existing at base — `test_gui` `move_card(validate=)` signature drift,
  follow-up candidate); legacy `packages/planner` tests 45 passed; ruff
  clean; CI 16/16 green.

Deliberately NOT done (shared conda env, other sessions mid-run): the
editable reinstall (`pip install -e packages/app-planner`) — `which
planner` on this machine still resolves the old editable until a quiet
window. Follow-up only: pyproject `name = "planner"` / v0.1.0 still
collides with legacy dist `planner` 0.2.0 and contradicts its own
setup.py (`app-planner`, 0.2.0).

Board observations while closing (pre-existing, not introduced here):
the board carries hundreds of duplicate card ids (this card had two
entries, both moved to done), and `verify_chain()` was already globally
broken at base — a full `compute_chains()` reseal on the hot shared board
was left alone.
