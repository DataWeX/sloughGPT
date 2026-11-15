---
id: 20261115_081943_pr-253-resolve-stacked-branch-conflicts-port-memory-wiring-h
status: done
board: 20261115_081943_pr-253-resolve-stacked-branch-conflicts-port-memory-wiring-h
title: "PR #253: resolve stacked-branch conflicts — port memory-wiring + hub-size fixes onto feat/remove-next-entirely"
tags: ["ci", "inference", "area:inference"]
sprint: ""
gh: "253"
---

# PR #253: resolve stacked-branch conflicts — port memory-wiring + hub-size fixes onto feat/remove-next-entirely

PR #253 (fix/remove-next-ci-inherit -> feat/remove-next-entirely) is CONFLICTING after #254's 5-branch merge landed on the base (588fdfc7c). Conflicts: apps/api/server/routers/inference.py (4 hunks) + domain/inference/engine.py (1 hunk), 5 other files auto-merged. Resolve, run affected closure + lint, push, watch CI, merge green.
