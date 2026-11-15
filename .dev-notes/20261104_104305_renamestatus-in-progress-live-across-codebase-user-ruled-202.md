---
id: 20261104_104305_renamestatus-in-progress-live-across-codebase-user-ruled-202
status: done
board: 20261104_104305_renamestatus-in-progress-live-across-codebase-user-ruled-202
title: "rename(status): in_progress -> live across codebase (user-ruled 2026-11-04)"
tags: ["refactor", "status"]
sprint: ""
gh: null
---

# rename(status): in_progress -> live across codebase (user-ruled 2026-11-04)

User ruling: 'in_progress - thats .live, change it across codebase'. IN: 3 enums (CompletionStatus/DeploymentStatus/TaskStatus IN_PROGRESS->LIVE, value in_progress->live) + refs, routers/agents.py raw literal, 2 web status checks (OrchestrationPanel, agents-orchestration-card), i18n key consciousness_training.training_in_progress -> training_live (5 locales + usage), old packages/planner ColumnDef x2, read-compat legacy alias in TrainingStatusTracker.load_report (persisted reports on disk contain in_progress; _coerce leaves unmatched strings as raw str -> .value AttributeError risk). OUT: board column wip, migrate_boards legacy source map, app-planner retired-column tests+docstring, journal/history, _stream_train_progress substring false positives (repl.py + 39 test refs), user data files. TDD: legacy-alias test RED first + value-pinning tests flipped to live RED, then rename GREEN.
