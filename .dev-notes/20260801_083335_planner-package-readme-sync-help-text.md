---
id: 20260801_083335_planner-package-readme-sync-help-text
title: planner package README + sync help text
status: done
tags: planner,docs,sync
created: 2026-08-01T08:33:35.896447+00:00
---

planner package README + sync help text

Added packages/planner/README.md (downcraft-style): philosophy, install, quick start, command table, config resolution order, status<->column mapping, sync semantics incl. card moves, storage backends, GUI API, dev/test. Corrected kanban add example to --column. Updated planner.sync argparse description + module docstring to state the bidirectional behavior (create missing cards + move stale columns). 60 planner tests pass.