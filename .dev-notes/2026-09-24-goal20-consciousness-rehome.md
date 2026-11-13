---
id: 20260924_034_goal20-consciousness-rehome
status: done
title: Goal 20 finish consciousness re-home
tags: roadmap,consciousness,cognition,refactor
---
# Goal 20 — finish consciousness re-home

Canonical code already at `domain/cognition/_internal/consciousness/` (commit 3aa5338b).
Remaining work is reference retarget + shim deletion.

## Plan
1. Branch `feat/goal20-consciousness-rehome`
2. Retarget all `domain.consciousness` → `domain.cognition` (or `_internal.consciousness` for private paths)
3. Update domain/core lazy map, packages/core-py/domains/consciousness re-export, context_core, scripts, tests
4. Delete `domain/consciousness/` (all shims) — goal says no deprecated shims
5. Fix ROADMAP wording if it still says `domain/cognitive` vs live `domain/cognition`
6. Public API branding stays `consciousness` (HTTP routes unchanged)
7. pytest consciousness + ruff; merge when green

## Verification (2026-09-24)
- Branch feat/goal20-consciousness-rehome
- domain/consciousness/ deleted (all shims)
- Retargets: domain/core lazy map, context_core, packages/core-py re-export, tests, demos, apps router tests
- 146 consciousness tests green; ruff clean; lazy facade imports domain.cognition
- ROADMAP goal 20 struck Done; path corrected cognitive → cognition
