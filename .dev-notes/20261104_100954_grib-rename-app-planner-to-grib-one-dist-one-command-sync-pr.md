---
id: 20261104_100954_grib-rename-app-planner-to-grib-one-dist-one-command-sync-pr
status: done
board: 20261104_100954_grib-rename-app-planner-to-grib-one-dist-one-command-sync-pr
title: "grib rename — app_planner to grib, one dist one command, sync producer watchdog"
tags: ["kanban", "planner", "infra"]
sprint: ""
gh: null
---

# grib rename — app_planner to grib, one dist one command, sync producer watchdog

Sprint: -  Gh: -

User ruling: "no app_planner, just grib" — the kanban is the feature, grib is the name.

Landed on branch feat/grib-rename (2 commits):
- 0252b5c03 module app_planner -> grib, packages/app-planner -> packages/grib,
  dist name off the colliding "planner", single console script grib = grib.cli:main,
  APP_PLANNER_* -> GRIB_*. packages/grib/setup.py deleted (second manifest for one
  package); packages/planner becomes library-only (planner.engine stays, it is
  imported at 4 sites by grib gui/kanban). New gate: a command name may be declared
  by exactly one manifest.
  NOT renamed on purpose: the planner/1 JSONL schema tag (store.py writes it,
  kanban.py + helpers.ts find the board header with it — renaming strands every
  board.jsonl) and the /api/planner/* route tree (live API surface, its own contract
  change). Canonical ~/.config/{dev-notes,kanban} never carried the package name.
  scripts/kanban_sync.sh discovers the producer module dir instead of pinning the
  path (the timer runs from a third checkout; a pinned path would have frozen it).
- 3f11ebb76 sync watchdog: TimeoutStartSec=180 on the unit (user manager defaults it
  to infinity — verified live), lock holder publishes pid+start time, contention
  names the holder and past KANBAN_SYNC_LOCK_STALE (900s) exits 1 instead of the
  old silent exit 0; KANBAN_SYNC_LOCK added for testability;
  check_kanban_freshness.py reads origin/main via git show (was judging the checked-out
  tree, which reddened every branch against a healthy timer) and fails OPEN with WARN
  when main cannot be read.

Gates: 396 grib+planner, 15 board chain, 12 freshness, 5 new lock tests, ruff clean,
bash -n clean, pip wheel proves entry_points [console_scripts] grib = grib.cli:main.

Still owed: pip uninstall planner && pip install -e packages/grib (from the merged
checkout), and the LIVE ~/.config/systemd/user/kanban-sync.service needs the
TimeoutStartSec line + daemon-reload.
