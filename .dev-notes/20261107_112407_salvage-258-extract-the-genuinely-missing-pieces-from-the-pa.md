---
id: 20261107_112407_salvage-258-extract-the-genuinely-missing-pieces-from-the-pa
status: todo
board: 20261107_112407_salvage-258-extract-the-genuinely-missing-pieces-from-the-pa
title: "salvage #258: extract the genuinely-missing pieces from the parked audit-a11y mega-batch, then close it"
tags: ["ci", "pr-sweep", "salvage", "area:tooling"]
sprint: ""
gh: null
---

# salvage #258: extract the genuinely-missing pieces from the parked audit-a11y mega-batch, then close it

Goal: PR #258 (fix/audit-a11y-design-tokens) was PARKED by the sweep (card 20261106_074042, comment 6703623669) — 1045 behind, 39 conflicts, 100 files, 28 commits, zero CI since 2026-11-01. It cannot be merged wholesale. This card cherry-picks the genuinely-missing pieces onto fresh main (port-don't-rebuild), then closes #258 as superseded.

Unique-content audit (spot-checked against origin/main 2026-11-07):
- ALREADY LANDED elsewhere (do NOT re-extract): scatter gate (tests/contract/test_scatter_gate.py via 8e1a0c75c), self-gen flywheel (domain/training/_internal/selfgen.py), page/workspace prunes (1e40ff6ce class), docs(sync)/planner churn.
- HARMFUL as-is (never merge): .dev-notes/notes.journal.jsonl + .kanban/board.jsonl — canonical store moved to ~/.config/{dev-notes,kanban} on 2026-10-28; repo copies are sync-produced snapshots.
- SALVAGE CANDIDATES (verify each still missing, then port individually):
  1. .opencode/agents/gene.md (coordinator agent) — absent on main.
  2. .gitignore entries for .wt-* worktree dirs + junk files — absent on main (main checkout currently carries ~60 untracked .wt-* dirs).
  3. Mobile a11y labels + color-token audit (f246bd9d3) + type-scale/magazine token refactor.
  4. ModelProfile (12146fd15 — verify naming vs what landed).
  5. SLO_AUTH_REQUIRED + SLO_CORS_ORIGINS in _SKIP_ENV_KEYS (d4ae8e0d7/6b73545b8) — NOTE: a variant already landed in the fork-port (2026-11-02 foreman lane); check before porting.
  6. Off-loop perf: models urlopen + pool teardown join (1f0247cfa).
  7. .opencode/agents/systems-engineer.md + skills (verify against current skills set).
- Contains raw WIP dump (8ce171bff "WIP: batch commit of working tree changes") — never mergeable.

Method: one candidate per commit on a fresh branch off origin/main; scoped tests per candidate; close #258 with a pointer here once the last candidate lands (or is proven already-landed).

Not in scope: rebasing the branch itself (39 conflicts across canonical-data files), any .dev-notes/.kanban file from the PR.
