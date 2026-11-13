---
id: 20260929_089_avion_cdp_cli
status: doing
title: "Card 089 — avion: CDP console/network capture + `avion` CLI, skill-library invariants, vision capabilities() contract"
tags: [avion, journey, cdp, cli, skills, degradation]
sprint: current
gh: null
---

# Card 089 — avion CDP CLI + skills/degradation quality pass

## Scope (re-scoped from original plan after measuring clean main)

Original plan (session 2026-09-29) was written against stale branch
`feat/canonical-embedder` (542 dirty files, 202 behind main). Measuring
`main` (9e5666fde) revealed:

- **Commit ① already done on main**: `packages/avion/pyproject name = "avion"`,
  69 `from avion.*` refs, arken/voyager are back-compat shims (single
  `from arken` ref lives only in voyager shim — matches the design).
- **Fake `ChromeDevToolsBrowser` does not exist at committed main.**
  It exists only as staged adds in worktree `sloughgpt-api-std` (another
  session's domain-testing migration) and on stale `feat/canonical-embedder`.
  DO NOT race that migration — flagged to user instead.
- Journey area owned by in-flight sessions: `.wt-ux`
  (feat/journey-consolidation, domain/testing/ux_flows), `.wt-jrn`
  (feat/headless-journey-sweep), `.wt-gates` (fix/journey-test-gates),
  `sloughgpt-router-wiring` (feat/journey-library). Committed runner on
  main: `python -m domain.journeys` (flows/steps/runner, PlaywrightBackend).

## What THIS card builds (collision-free, all inside packages/avion)

1. **CDP console + network capture + wait_for_text** in
   `avion/backends/cdp.py` (Log.enable / Network.enable events).
2. **`avion` CLI** (console script + `python -m avion`): open, snapshot,
   click, fill, wait, shot, console, network, --repl — Python/agent parity
   with chrome-devtools MCP verbs. websockets → shared root .venv only.
3. **SkillLibrary invariants**: success-rate stats on replay, rank-by-success
   find(), trajectory-hash dedup, schema `version` + migrate-or-skip-loudly,
   atomic save. Test per invariant.
4. **Vision degradation contract**: `ImageAnalyzer.capabilities()` →
   {ocr, template, reason}; typed status instead of bare `[]` (opencv/
   tesseract absent in this env — degraded path exercised for real).
5. **Naming guard test** `test_naming.py`: import avion resolves, zero
   `arken` imports outside voyager shim, pyproject name == "avion".
6. packages/README.md + SOTA_COMPARISON.md truthup for avion row.

## Coordination

- domain/testing stub + TRAINING_PAGES + runner re-point: **owned by
  api-std migration session** — not touched here.
- domain/journeys / ux_flows runner work: **owned by .wt-ux / .wt-jrn** —
  not touched here. CLI is additive (avion package only).

## Gates

- packages/avion/tests (23 files) green in .wt-avion worktree.
- CLI smoke: real CDP session against Chromium from ms-playwright cache.
- naming guard test green; lint/typecheck/test per AGENTS SOP.

## Landed (branch `feat/avion-journeys`, worktree `.wt-avion`, 6 commits)

| # | sha | what |
|---|-----|------|
| 1 | `55cad0a76` | TOOL_CALL seam — `Agent(tools=, executor=)`, models.py enum + card |
| 2 | `7a88d4438` | CDP console/network capture + `avion` CLI (open/snapshot/click/fill/wait/shot/console/network/--repl) |
| 3 | `89afc48f6` | SkillLibrary invariants, naming guard, packages/README + SOTA truth-up |
| 4 | `4d1e374c6` | vision `capabilities()` — typed degradation contract |
| 5 | `396e8fe88` | OpenCode skills audit — repaired 3 undiscoverable skills + conformance guard |
| 6 | `35081c6d2` | `_free_port()` mypy cleanup (zero new mypy errors proven) |

Gates all met: **283 passed, 1 skipped** (avion) + **39 passed**
(`tests/test_opencode_skills.py`) + ruff clean + mypy error set identical
to base `9e5666fde` (86 = 86, 25 matching messages).

**CI baseline (proven, not assumed):** running CI's exact invocation
(`pytest tests/` with the `reusable-ci-core.yml` ignore list) on this
branch gives 3081 passed / 13 failed / 5 errors. Extracting the base
commit to a scratch dir and running the same 5 failing files there
reproduces **the identical 13 + 5 list** — they are pre-existing on
`9e5666fde`, in `test_system_router`, `test_feedback_domain`,
`test_inference_generate`, `test_rag`, `test_cli_chat`. No test imports
`avion`, and none of those files are in this branch's diff, so this
branch adds exactly one passing test and regresses nothing.

Scope note: card items 1-6 are all inside `packages/avion` as planned.
Added scope (item 5's sibling): an OpenCode skills audit found
`cleanup-agent` (no frontmatter), `component-state-completeness` and
`systematic-debugging` (unquoted `: ` in description) were silently
absent from `<available_skills>` — repaired, verified 13 -> 15 skills,
and guarded by `tests/test_opencode_skills.py` (in pytest `testpaths`,
so CI runs it).

## Remaining (decided 2026-09-30)

- **Merge: deferred to user coordination.** `main` is checked out by
  session tree `/home/mana/Documents/sloughgpt-api-std` (git marks it
  with `+`), so merging from here would require touching another
  session's tree. Decision: **leave the 6 commits local on
  `feat/avion-journeys` in `.wt-avion`; user coordinates the merge.**
  That tree is clean w.r.t. my files, so merging there will not conflict.
- **Primary-tree copies: intentionally left in place.** The 3 repaired
  skills + untracked `tests/test_opencode_skills.py` stay in the primary
  checkout (branch `feat/logs-unification`) so `cleanup-agent` and
  `component-state-completeness` remain discoverable right now.
  Content is byte-identical to what the branch carries; whoever merges
  into that tree should `git checkout --` the 3 files and remove the
  untracked test first, and the fix returns from the branch.
- Global-copy drift reported, `$HOME` untouched: 6 skills differ between
  `.opencode/skills/` and `~/.config/opencode/skills/`. Currently inert
  because `opencode.json` sets `skills.paths: [".opencode/skills"]`.
- `.agents/skills/SKILL.md` is stray for skill purposes (no `<name>/`,
  no frontmatter) but is intentional guidance referenced by
  `docs/STRUCTURE.md` — not deleted.
