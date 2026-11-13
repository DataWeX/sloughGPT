---
id: 20261004_avion_arken_import_fix
title: "avion: finish the arken→avion rename — imports only, + Avion aliases, + cli entry"
status: done
date: 2026-10-04
tags: [avion, rename, computer-use, python]
---

USER DIRECTIVE 2026-10-04: "fix these blockers" / "Patch avion here".

BLOCKER (measured): `import avion` on feat/create-router-projection dies with
`ModuleNotFoundError: No module named 'arken'` — avion/__init__.py:11 does
`from arken.ai.agent import ...`. 81 broken import statements across
packages/avion (78 `from arken.` + 5 `from arken import`), 32 src files +
14 test files. Pre-existing card 122109a9 diagnosed this; fix never landed
on this branch. packages/arken exists as a back-compat shim only
(src/arken/__init__.py re-exports `from avion import ...`) — it does NOT
supply arken.* submodules, so the broken imports can never resolve.

SCOPE (imports only — deliberately matching main):
- main's fix touched imports and NOTHING else: `git grep "from arken\." main`
  = ZERO, while arken_output / arken_session / ctx["arken"] /
  StructuredLogger("arken") are all still present on main as string
  literals. Those are an output path, a session-name default, a task-ctx
  key and a logger name — NOT imports. Renaming them here would diverge
  from main and manufacture merge conflicts. DO NOT TOUCH THEM.
- Fix: `from arken.` → `from avion.` (78) + `from arken import` →
  `from arken import`→`from avion import` (5).
- Add missing `Avion = Arken` / `AvionConfig = ArkenConfig` aliases in
  avion/__init__.py — main has them (back-compat), and packages/arken shim
  already imports `Avion`/`AvionConfig` from avion, so today the shim is
  ALSO broken.
- pyproject.toml `name = "arken"` → `"avion"` (main's value) + description.

ALSO (same session, env): playwright 1.62.0 was installed into
~/.local/lib/python3.12/site-packages (user site), invisible under the
mandatory PYTHONNOUSERSITE=1 (required: ~/.local/.../tests shadows the
repo tests/). Reinstalled playwright==1.62.0 + greenlet + pyee into the
shared conda env sloughgpt — the ONE toolchain (shared-workspace skill).
Verified: PYTHONNOUSERSITE=1 now imports playwright from the conda env.

GATES: PYTHONNOUSERSITE=1 import avion; py_compile over packages/avion;
packages/avion tests; ruff. Re-verify the live browser gates that already
passed this session: PlaywrightBackend E2E (click → DOM mutation) and the
CDP CLI (open/snapshot/click/wait/console against chrome-headless-shell 151
on :9222).

COORDINATION: parallel worktrees .wt-aviondrv/.wt-avion own their own
copies — do not edit them. This card touches ONLY packages/avion on
feat/create-router-projection. Existing dirty files in the tree
(domain/soul, domain/training, .kanban) belong to another session — do not
stage or revert them.

RESULT 2026-10-04 — committed b94798cbc (50 files: packages/avion +
packages/README.md only; the other session's files untouched).

Done as scoped: 81 `from arken` imports → `from avion`, Avion/AvionConfig
aliases, pyproject name + [project.scripts]. Plus three gaps found while
gating (all were pre-existing breakage, none from this rename):
- Ported backends/cdp.py, cli.py, __main__.py from main — this branch had
  the test for them but not the code, so the documented `python -m avion`
  did not exist here.
- session.py:84 passed a positional `%s` to StructuredLogger.info(msg,
  **fields) → TypeError on every session start (ef58a0c9c landed while the
  suite was uncollectable). Switched to the structured `backend=` field,
  which satisfies both the API and the no-f-string cleanup intent.
- ActionType.TOOL_CALL + Agent(tools=, executor=) dispatch: 81616a6f7
  copied tests/test_tools.py in without its implementation — 6 tests
  failed on the missing enum member. Ported the exact hunks from main;
  ai/models.py is now byte-identical to main.

GATES (all green): ruff clean; py_compile clean; avion suite 233 passed /
1 skipped (the skip is the missing-dep path for websockets, correct while
it's installed); tests/test_shared_indexes 6 passed; `import avion` +
`arken` shim OK under PYTHONNOUSERSITE=1; live CDP CLI
open/snapshot/click/wait/console against chrome-headless-shell 151 on
:9222; PlaywrightBackend E2E click → DOM mutation 'clicked'.

Follow-up candidates (not in this card): packages/README.md documents
`python -m avion` but the branch README table was stale (fixed the avion
cell); avion still has no live-UI journey surface — see journal.

