---
id: 20261113_095954_worktree-cleanup-preserve-then-remove-6-merged-stale-worktre
status: done
board: 20261113_095954_worktree-cleanup-preserve-then-remove-6-merged-stale-worktre
title: "worktree cleanup: preserve-then-remove 6 merged stale worktrees (release prep)"
tags: ["cleanup", "worktrees", "release"]
sprint: ""
gh: null
---

# worktree cleanup: preserve-then-remove 6 merged stale worktrees (release prep)

User ruling 2026-11-13: finish stale worktrees so a release can ship. APPROVED SCOPE (question tool): Tier 1 + Tier 2, preserve first. Parked UNMERGED lanes (23) stay per the 2026-11-05 park-don't-kill ruling.

TIER 1 (remove, dirt = proven formatting/harness noise): .wt-decodespace, .wt-chatgate, .wt-kernprobe — all branches merged to origin/main; diff -w shows formatting-only drift + .opencode/* harness churn; kernprobe's 3 probe scripts md5-identical to main.

TIER 2 (preserve real work, THEN remove): .wt-stage4 (untracked kernel_core/engine/ops + benchmark + test ~1017 lines NOT on main + benchmark warm-up fix not on main), wt-threadflake (race_enforce.py + stress_flake.py not on main), swift-harbor (ease-expo tailwind token edit), .batch-logs/ diag scripts in deodespace/chatgate. Preservation = salvage commits on new branches in main .git + copy of loose files to ~/Documents/wt-salvage-20261113/.

TOOLING: scripts/prune-worktrees.sh (3-gate safety model) committed on chore/worktree-prune-script — PR follow-up.

NEVER TOUCHED: primary checkout, kanban-sync-wt, wt-agent-phase2, .wt-quicktest (live pid), wt-prune (live).
