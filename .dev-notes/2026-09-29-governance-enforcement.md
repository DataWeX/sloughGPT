---
id: 20260929_081_governance_enforcement
status: done
title: "Card 081 — Enforce shared-packages governance: contract test, skill, permission rule, planner acceptance"
tags: [policy, governance, ci, agents, planner]
sprint: current
gh: null
---
# Card 081 — Governance enforcement (mechanical + agentic + in-app)

Extends card 080 (policy prose) with enforcement at three layers, plus
the same ritual inside the app's own planner workflow.

## Build list

1. **Contract test** `tests/test_shared_indexes.py` (precedent:
   `test_repo_root_package_json.py`):
   - every `apps/*` dir → entry in `apps/README.md`; `packages/*` →
     `packages/README.md`; core docs → `docs/INDEX.md`
   - duplicate `node_modules`/`.venv`/lockfiles outside shared roots
     (allowlist `apps/mobile`) = fail
   - rides the existing pytest CI job (no ci_cd.yml edit needed — that
     file is parallel-session WIP)
2. **`./verify.sh`** hook — same check in the local pre-push gate.
3. **Skill** `.opencode/skills/shared-workspace/SKILL.md` —
   description-triggered before install/scaffold/rebuild decisions.
4. **Permission rule** (opencode config): confirm on
   `npm install`/`pip install`/`rm -rf node_modules` with shared-copy
   pointer in the deny message.
5. **AGENTS.md SOP step 1** — cards must state: index updated /
   shared copy reused / coordinating card.
6. **App workflow** — planner card model (app planner + packages/
   planner) gains the same acceptance fields, so cooperation is
   visible in the product itself.

## Gates

pytest (new test green + full server suite), verify.sh, ruff, skill
lint (md), path-scoped commits on `feat/shared-governance`.

## Done — 2026-09-29

Commits (all reachable from `feat/shared-governance` =
`fix/training-recovery-state` tip 48681597b):

- `a396e01b2` — contract test (6 checks) + full 19-package index in
  `packages/README.md` + verify.sh gate (also fixed stale
  `domains/ui/api_server.py` path → `domains/api/__init__.py`, never
  existed in git) — verify.sh ran green end-to-end incl. web `npm run ci`
- `064228d38` — shared-workspace skill, opencode `permission.bash`
  ask-first on npm/pip installs + node_modules/.venv deletion + git
  clean, AGENTS.md SOP step 1 acceptance line
- `48681597b` — planner acceptance fields (types, helpers create/update,
  client payloads, CardEditor fieldset w/ strui Checkbox, CardItem
  indicators + 7 new tests) — full web suite 839 files / 8153 tests
  passed, tsc clean, eslint 0 errors, targeted 57/57 re-run on
  committed content

Restart note: permission rule + skill load at opencode startup — running
sessions keep old config until restart.

Coordination: parallel session's board cards 078–087 + their note
`2026-09-29-card081-kanban-content-hashes.md` reuse the 081 number
(hash-tree plan); this governance card keeps 081 (created first) —
hash-tree plan references it. Numbering resolved by reference, not
renumber.
