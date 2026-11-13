---
id: 20261001_avion_rename_tests
title: "avion test suite collectable: arken->avion rename completed (card 122109a9)"
status: done
sprint: 2026-10-01
tags: [avion, testing, rename, playwright, kanban]
gh: ""
created: 2026-10-01T08:30:00Z
---

# avion test suite collectable: arken->avion rename completed (card 122109a9)

## Summary

Closed board card `122109a9` ("avion test suite uncollectable: avion/__init__.py
imports arken (half-done rename)"). The suite went from **21 collection errors /
0 tests collected** to **236 collected, 229 passed, 1 skipped, 6 xfailed**,
running green twice. Branch `fix/avion-tests-collectable` in `.wt-shot`,
commits `41876e5ed` (fix) + `d562b38fd` (close row), pushed to
`feat/shared-governance` on both the local repo and origin.

## What was broken

- `packages/avion` is the renamed copy of `packages/arken`, but 138 refs still
  said `arken`: 32 src files, 17 test files, `pyproject.toml` (`name = "arken"`),
  identity strings (`arken_output/`, `arken_session`).
- Every test died at import with `ModuleNotFoundError: No module named 'arken'`
  (neither module is installed in the shared `.venv`; conftest puts
  `packages/avion/src` on `sys.path`, which only provides `avion`).
- First-ever real run then revealed 13 latent failures behind the collection
  wall.

## What changed (53 files, +256/-143)

1. **Rename completed** — word-boundary `arken -> avion` across
   `packages/avion/**/*.py` + `pyproject.toml` (classes keep the `Arken` name —
   matches what `feat/avion-journeys` does; module paths only).
   `SOTA_COMPARISON.md` keeps its historical "voyager -> arken" line untouched.
2. **`test_rename_guard.py` (new)** — package-local regression guard: no
   `from/import arken` inside `packages/avion`, `pyproject name == "avion"`.
   Written TDD-style: RED first, GREEN after rename. The guard itself is the
   only file allowed to contain the retired name.
3. **`session.py` logger fix** — `StructuredLogger.info()` is kwargs-only but
   got a printf positional; pre-formatted the message. Freed 3 journey tests.
4. **`PlaywrightBackend` browser resolution** — optional `executable_path`
   param + `_resolve_executable()`: explicit > playwright's own download >
   `$AVION_CHROMIUM` > flatpak export shim. Covers all default-backend callers
   (`session.py`, `agent.py`) in one place. House rule honored: never trigger
   `playwright install`. Live module skips when no chromium exists at all.
5. **`test_tools.py` TOOL_CALL x6** — marked `xfail(reason="TOOL_CALL seam
   lands with avion-journeys 55cad0a76")`: src support exists only on that
   branch (verified with `git log -S'TOOL_CALL' --all`). Non-strict, so the
   merge flips them to xpass without breaking anyone's gate.

## Verification

- Suite: `229 passed, 1 skipped, 6 xfailed` — identical on two runs (~18s,
  flatpak chromium launches included).
- `ruff check packages/avion` clean (1 isort nit auto-fixed).
- `mypy packages/avion/src`: 113 errors at tip baseline -> **112** after
  (baseline proven in a throwaway worktree at `b17dfec2d`; zero new errors —
  remaining 112 are the pre-existing `Optional` init pattern).
- `tests/test_shared_indexes.py` 6/6; no new app/package/doc (no index change
  needed); reuses the shared `.venv`, no installs.
- Close row appended to the tip-based board copy: `1 file changed,
  1 insertion(+)` — append-only, no row loss.

## Coordination

- `feat/avion-journeys` (other session, idle 16h+, pushed to origin): their new
  tests already use `avion` imports, their `pyproject name == "avion"`, and
  they add the repo-wide AST guard `test_naming.py` — my package-local guard is
  complementary and named differently (no filename clash). Same-change merges
  should auto-resolve; their `TOOL_CALL` seam turns my 6 xfails into xpasses.
- `packages/arken` 2-file skeleton + `voyager` shim + `scripts/run_journey_tests.py`
  deliberately untouched — the shim re-exports voyager -> arken -> avion by
  design (their branch's guard docstring confirms this architecture).
- The other session's note `2026-09-29-avion-cdp-cli.md` (in_progress) was not
  modified.
- Foreign staged `src/avion/ai/slo.py` in the main tree: unrelated change
  (`domains.` -> `domain.` import fix), different lines than mine — no
  collision.
