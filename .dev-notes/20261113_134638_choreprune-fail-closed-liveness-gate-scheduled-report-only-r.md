---
id: 20261113_134638_choreprune-fail-closed-liveness-gate-scheduled-report-only-r
status: done
board: 20261113_134638_choreprune-fail-closed-liveness-gate-scheduled-report-only-r
title: "chore(prune): fail-closed liveness gate + scheduled report-only run of prune-worktrees.sh"
tags: ["worktree", "ops", "safety"]
sprint: ""
gh: null
---

# chore(prune): fail-closed liveness gate + scheduled report-only run of prune-worktrees.sh

cwd_owner must be tri-state so a failed /proc probe aborts instead of silently reporting live=0; then schedule a weekly report-only run mirroring the existing kanban-sync.timer. --apply stays an explicit human action.
