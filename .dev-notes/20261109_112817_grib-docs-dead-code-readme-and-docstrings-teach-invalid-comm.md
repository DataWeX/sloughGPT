---
id: 20261109_112817_grib-docs-dead-code-readme-and-docstrings-teach-invalid-comm
status: done
board: 20261109_112817_grib-docs-dead-code-readme-and-docstrings-teach-invalid-comm
title: "grib docs + dead code: README and docstrings teach invalid commands (root cause: kanban.py legacy parallel CLI)"
tags: ["grib", "docs", "cli"]
sprint: ""
gh: null
---

# grib docs + dead code: README and docstrings teach invalid commands (root cause: kanban.py legacy parallel CLI)

Survey 2026-11-09 (grib quality pass, chosen target F3+F4, F2 rides along). F3: README examples (lines 35-44: grib new/list/show/edit/search/add/move) and the module docstrings in core.py/kanban.py teach commands that DO NOT EXIST — verified: python -m grib.cli new x -> argparse "invalid choice: new (choose from note, notes, board, kanban, sync, verify, migrate, snapshot, export, gui)". Shipped syntax is grib note ... / grib board ... . Root cause F4: kanban.py:980-1369 holds a SECOND, parallel CLI (cli_main, ~390 lines, 0% tested, argparse prog="grib", reachable only via python -m grib.kanban) that the docs were written against; core.py:588 already delegates correctly to grib.cli — delete kanban cli_main the same way (keep the module-usage examples in docstrings, fix the CLI-usage ones). F2 rides along (same README): the install section (line 28, pip install -e packages/grib) must also install packages/planner — grib GUI/kanban import planner.engine (gui.py:362/370, kanban.py:345/726) while grib pyproject declares dependencies=[]; NEVER add planner as a pyproject dependency (planner on PyPI is a foreign package — name collision). Env state 2026-11-09: planner 0.2.0 (library-only, no console scripts) + grib 0.1.0 both installed --user; stale planner/kanban shims removed.
