---
id: 20260929_083
board: 13aed9ad-9b15-45c8-be99-6b07c9b80e7c
title: "Card 083 — Doc-vs-code gap audit: docs vs APIs cross-referenced with tests"
status: done
date: 2026-09-29
tags: [docs, qa]
branch: docs/api-doc-gap-audit
worktree: .wt-audit
---

# Doc-vs-code gap audit: docs vs APIs cross-referenced with tests

## Background

Ask 2026-08-29: 'compare our old documentation, the updated one, with our current API implementations and codebase to see gaps we missed, and cross-reference it with our test results'. Produce gap report -> cards. Never done as a card.

## Source
Mined from stale opencode sessions (Aug 26-31 2026) at user's request: ~/.cache/slog-journeys/session-exports/.

## Landed (2026-09-29)

Branch `docs/api-doc-gap-audit` (worktree `.wt-audit`, uncommitted).

### Deliverable 1 — re-runnable audit: `scripts/audit_api_docs.py`

Enumerates live routes via probe app (`get_all_routers()` + `training.router` +
main.py's 4 routers → `probe.openapi()["paths"]`, fastapi >=0.139 lazy
`_IncludedRouter` makes `app.routes` under-report) and diffs `docs/routers.md`
in **(METHOD, path) pairs** — same unit as card 071's contract gate.
Flags: `--json`, `--max N`, `--doc PATH` (exit 0 always: report tool).

Baseline on main `9871d0c2f`:

| metric | value |
|---|---|
| code operations (method+path) | **674** |
| documented rows (routers.md) | 392 |
| undocumented (code, not docs) | 323 |
| stale (docs, not code) | 41 |
| method mismatches | 6 |
| router modules / with tests | 56 / **56** |

Method mismatches (doc says one method, code serves another):
`/companion/presets` (+POST missing), `/files` (doc claims POST, code GET-only),
`/files/{file_id}` (+DELETE), `/lora-eval/run` (GET→POST),
`/memory/archive` (POST→GET), `/security/keys` (+POST).

### Coordination with card 071 (owner: sibling session)

The doc-vs-code *fixing* half is **owned by card 071**
(`feat/routers-doc-truthup`, `tests/contract/test_routers_doc.py` gate:
stale==0, missing==0, floor >=600 — its pair-set comparison also catches the
6 method mismatches). I reverted my `docs/routers.md` edits
(`git checkout main -- docs/routers.md`) to avoid conflicting with their
+403/-52 truth-up. My independent audit reproduces their exact baseline
(674/392/323/41) → cross-validation of both toolchains. Their branch also
fixed their contract tests to probe enumeration (app.routes 608 vs 674).

### Deliverable 2 — router↔test cross-reference (unique to this card)

Content-index of all 56 router modules against test files → found
`model_stack` + `phoneme` with zero matching tests → **card 088, closed same
day** (now 56/56).

### Test-infra breakage found while verifying (fixed on this branch)

1. **7 files imported `tests.test_support`** (`test_errors/exception_handlers/
   feedback/files/images/inference_router`, `test_middleware`) — resolved to
   the stray user-site `~/.local/.../site-packages/tests` package instead →
   **collection aborted the ENTIRE server suite** (6 errors interrupt all
   1075 tests). Retargeted to bare `from test_support import …` like the
   other 40 files. Pre-existing on main (reproduced with changes stashed).
2. **Root `conftest.py` appended `packages/downcraft`** while site-packages
   has stale `downcraft 0.2.0` (flat layout, landed 2026-09-18) and the repo
   moved to `0.3.0` nested (`download/`) → `from downcraft.download import
   …` failed in tests → red `test_router_domain_facade` guard (hf_hub,
   models.py:845 lazy import). App is fine (`main.py:49` inserts at 0).
   Conftest now inserts at 0 too, mirroring the app bootstrap.

## Gates

- server suite (canonical `cd apps/api/server && pytest tests/`):
  **1206 passed / 0 failed** (before: 6 collection errors + 2 failures)
- `scripts/audit_api_docs.py`: 674/392/323/41/6, routers 56/56, ruff clean
- touched files ruff clean

## Not done / follow-ups

- Doc rows fixing: card 071 (in_progress, sibling).
- downcraft env drift reconciliation (env 0.2.0 vs repo 0.3.0, e.g. editable
  install) → card 089.
- `packages/core-py` suite still fails at `from tests.conftest import …` in
  non-gates worktrees (stray user-site `tests` shadow) — fixed by gates
  branch `tests/__init__.py`, awaiting merge.
