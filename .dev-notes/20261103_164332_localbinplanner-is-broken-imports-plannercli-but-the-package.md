---
id: 20261103_164332_localbinplanner-is-broken-imports-plannercli-but-the-package
status: done
board: 20261103_164332_localbinplanner-is-broken-imports-plannercli-but-the-package
title: "~/.local/bin/planner is broken: imports planner.cli but the package is app_planner"
tags: ["planner", "kanban", "env", "install"]
sprint: ""
gh: null
---

# ~/.local/bin/planner is broken: imports planner.cli but the package is app_planner

Found 2026-11-03 while filing cards (SOP step 1). The installed console script dies immediately: /home/mana/.local/bin/planner line 3 does 'from planner.cli import main' and raises ModuleNotFoundError: No module named 'planner.cli'. packages/app-planner/pyproject.toml registers the entry points as planner = app_planner.cli:main and kanban = app_planner.cli:main, so the on-disk shim is stale relative to the app_planner rename; 'kanban' is not on PATH at all and app_planner is not importable from the sloughgpt conda env either. Workaround that works today: PYTHONPATH=<repo>/packages/app-planner/src ./scripts/python -m app_planner.cli <subcommand>. Fix is a reinstall of the editable package so the shim regenerates — that mutates ~/.local and needs user approval. AGENTS.md makes the planner the canonical path for notes/board, so a broken shim silently pushes sessions toward hand-editing board.jsonl, which is the chain-seal hazard AGENT_SYNC documents.
