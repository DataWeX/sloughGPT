---
id: 20261104_062921_check-kanban-freshnesspy-reads-the-checked-out-tree-so-every
status: done
board: 20261104_062921_check-kanban-freshnesspy-reads-the-checked-out-tree-so-every
title: "check_kanban_freshness.py reads the checked-out tree, so every non-main checkout reports a false 'timer looks dead'"
tags: ["ci", "kanban", "monitoring", "false-positive"]
sprint: ""
gh: null
---

# check_kanban_freshness.py reads the checked-out tree, so every non-main checkout reports a false 'timer looks dead'

Found 2026-11-04 while chasing a stale export. The script stats .kanban/manifest.json in the CURRENT working tree, but the producer (scripts/kanban_sync.sh) exports inside the detached worktree ~/.cache/kanban-sync-wt and pushes to MAIN. Any session sitting on a lane branch therefore has an old snapshot and the alarm fires no matter how healthy the timer is. Observed directly: freshness printed FAIL age=14:21:12 > 2:00:00 while origin/main had 'kanban: sync 2026-11-04T06:02:28Z' and the timer pushed again at 07:02 ('pushed (2 paths)'). Fix: measure origin/main's manifest via 'git show origin/main:.kanban/manifest.json' (fall back to the working tree only when HEAD is main). Why it matters: an alarm that is red during normal work trains everyone to ignore it, and it nearly caused a manual re-export on top of a live single-writer. Related card: the sync unit's missing start timeout.
