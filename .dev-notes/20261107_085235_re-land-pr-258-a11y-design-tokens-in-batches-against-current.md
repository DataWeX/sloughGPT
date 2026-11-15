---
id: 20261107_085235_re-land-pr-258-a11y-design-tokens-in-batches-against-current
status: todo
board: 20261107_085235_re-land-pr-258-a11y-design-tokens-in-batches-against-current
title: "re-land PR 258 a11y + design tokens in batches against current main"
tags: ["frontend", "a11y", "pr-queue", "area:web"]
sprint: ""
gh: null
---

# re-land PR 258 a11y + design tokens in batches against current main

PR 258 (fix/audit-a11y-design-tokens, 28 commits, 341 files, author iamtowbee) is unrebaseable: 187/341 files conflict after Next removal + page pruning + route consolidation. Closed instead of merged. Re-land plan: group the changes by area (mobile components, web pages, workspace/training pages, tokens/baselines), cherry-pick each group onto a fresh branch from main, resolve the group's conflicts small, one PR per batch. First verify what already landed on main since Nov 1.
