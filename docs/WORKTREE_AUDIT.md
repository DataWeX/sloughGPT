# Worktree Cleanup — 2026-11-13

## Result

| | Before | After |
|---|---|---|
| Worktrees | 115 | **33** (fluctuates — other sessions create them) |
| Disk used | 105G / 113G (**98%**, 2.9G free) | **84G / 113G (78%, 24G free)** |
| Reclaimed | — | **~21 GB** |

- **82 worktrees removed, 0 failures.**
- **1,019 local branches intact** — removing a worktree never deletes its branch.
- All spot-checked commits reachable (`fix/ci-baseline-red`, `feat/gate-product-surface`, `fix/vm-times-emit`).
- **No worktree with a running process was touched.** Verified via `/proc/*/cwd` scan immediately before removal.

## How the safe set was determined

Four gates, all must pass:

1. **Tier** — clean (A), infra-drift-only (B), or formatter-explained (C).
2. **`ruff format` equivalence** — for Tier C, each modified `.py` was extracted at its `HEAD` version, run through `ruff format --line-length 100`, and compared to the working copy. **4,520 of 4,572 files (98.86%) matched exactly.**
3. **No untracked source files** (`.py`/`.ts`/`.tsx`/`.js`/`.sh`/`.yml`/`.rs`/`.c`) — the formatter never creates files, so these are unexplained.
4. **No non-`.py` code diffs** (`.ts`/`.tsx`/`.js`/`.rs`/`.c`).

### Why `black` gave false positives

An earlier pass used `black` and reported 52 "genuine" files. That was wrong —
`lint-staged` runs **`ruff format`**, not black:

```json
"*.py": [ "scripts/python -m ruff check --fix", "scripts/python -m ruff format" ]
```

Re-tested with `ruff format`: 0 genuine in the sampled worktrees.

### The `.md` / `.ipynb` diffs are formatter output too

- `.md` — black formatting **Python code blocks inside markdown** (e.g. `PLAN.md`,
  `docs/REFACTOR_PROMPT.md`, `packages/bawl/README.md`). 850 modified `.md` files.
- `.ipynb` — black formatting notebook code cells; inserts a blank line after a
  function, which the whitespace-strip test misreads as "real content".
  `sloughgpt_colab.ipynb` alone accounts for 24 worktrees, with byte-identical
  diffs across most of them.

### Root cause

The formatter is **commit-gated, not working-tree-gated**. `lint-staged` formats
*staged files at commit time* only — it never rewrites a dirty working tree. A
worktree left uncommitted accumulates drift indefinitely, which is why 55 of them
appeared "dirty" in the audit. This is pending-*commit* state, not pending-*work*
state.

## Held back — 29 worktrees, 6.4 GB

Genuine uncommitted work. Each needs an individual decision.

| Reason | Worktrees |
|---|---|
| Genuine `.py` changes (52 files) | `.wt-avion-act`, `.wt-b2-heart`, `.wt-decode`, `.wt-heart`, `.wt-kernprobe`, `.wt-mod`, `.wt-pguard-heartbeat`, `.wt-pguardhb`, `.wt-seam`, `.wt-stage4`, `.wt-ux`, `sloughgpt-srv-tests` |
| Genuine `.ts`/`.tsx`/`.rs`/`.c` changes | `sloughGPT-frontend`, `.wt-config`, `.wt-gstage`, `.wt-strui`, `swift-harbor`, `swift-comet` |
| Untracked source files | `sloughGPT-separate` (61), `.wt-modular` (270), `.wt-stage4` (5), others |
| Live session — **do not touch** | `.wt-quicktest`, `kanban-sync-wt`, `wt-agent-phase2` |

The 52 genuine `.py` files cluster in `domain/infrastructure/` and `domain/testing/` —
mostly the pguard-heartbeat feature spread across four stale checkouts
(`.wt-heart`, `.wt-pguard-heartbeat`, `.wt-pguardhb`, `.wt-b2-heart`).

### Notable held items

- **`sloughGPT-separate` — HAS UNIQUE WORK. Do not delete.** (Superseded an
  earlier read of this as "entirely disposable" — that was wrong.)

  Structure: forked 2026-09-18 (`merge-base 69cb472d3`), now **2,906 commits
  behind main, 29 ahead**. Its own delta is 60 files / +1,766 −1,629.

  **Of its 61 untracked source files, 61 are superseded** — 50 are
  `packages/voyager/**`, 7 exist on main, 4 resolved by symbol analysis.
  So the *untracked* files are all disposable.

  **But its 29 commits are not.** Four files main lacks entirely:

  | File | Lines | What |
  |---|---|---|
  | `apps/cli/src/utils/prune.py` | 220 | stale-server GC: `_is_own_server`, `_proc_port`, `_health_ok`, `_protected_pids`, `_has_healthy_server_child` |
  | `apps/cli/tests/test_prune.py` | 231 | its tests |
  | `domain/shell/_internal/pretty.py` | 319 | `highlight()` syntax colouring, `format_traceback`, `format_error_brief`, `terminal_page_size` |
  | `packages/core-py/tests/test_shell_pretty.py` | — | its tests |

  Plus two REPL commands main has **no equivalent of**:
  - **`man`** — `ShellREPL._cmd_man`, registered in `permissions.py` and the
    command table (`repl.py:5127`)
  - **`traceback`** — `_last_traceback` + `format_traceback()`; main's
    `repl.py` has zero `traceback` references

  (`grep "pager"` does hit main — but in `console.py`/`interactive_prompt.py`,
  a different widget system, not sep's REPL pager.)

  It also touches 61 files. **Hunk-level pass done 2026-11-13** — the earlier
  "not determined" caveat is now answered:

  **48 of 61 files are fully superseded.** Measured three-way (merge-base →
  sep → main) on definitions: sep's own added defs are all present in main,
  and main's file is *larger* in every relocated comparison (e.g.
  `test_shell_repl_more.py` sep 30,023 L vs main 31,120 L; `core-py/conftest.py`
  sep 21 L vs main 132 L; `dev.py` sep 1,509 L vs main 1,730 L; `slo_cli.py`,
  `test_inference_router.py`, `test_souls_router.py`, `test_kb_router.py`,
  `knowledge/engine.py` — all `NONE — superseded`).

  **Relocations, not losses** — main moved `packages/core-py/tests/` →
  `tests/core-py/`, so 4 files are absent at their sep path but present:
  `test_shell_integration`, `test_shell_repl_more`, `test_shell_router`,
  `test_repl_execute`. Main also renamed `app-planner` → `planner`.

  **Genuinely unique — two clusters:**

  | Cluster | Files | Evidence |
  |---|---|---|
  | REPL `man` + `traceback` + pretty output | `repl.py`, `pretty.py`, `test_shell_pretty.py`, `test_repl_execute.py` | `repl.py` has 7 defs main lacks: `_cmd_man`, `_cmd_traceback`, `_read_more_key`, `_section`, `_status_section_{events,logs,metrics}` — all **0 occurrences anywhere in main's `domain/`**. `test_shell_pretty.py` truly absent. `test_repl_execute.py` carries 17 sep-only tests (`TestManCommand`, `TestTracebackCommand`, `TestDumpJson`, `TestCompleteArgsForMan`). |
  | Stale-server GC | `prune.py`, `test_prune.py` | Both truly absent from main at any path. |

  **Dead even in sep — discard:** `apps/api/server/tests/test_collections_router.py`
  was *added then removed inside sep's own 29 commits* (`808be1220`), is not at
  sep's HEAD, and **neither tree has a collections router**. Not unique work.

  **One genuinely ambiguous case — `packages/core-py/tests/test_kanban.py`
  (380 L, 47 tests).** Not a clean verdict either way:
  - Main renamed `app-planner` → `planner`, and sep's 7 `KanbanStore` methods
    (`block_card`, `is_blocked`, `_find_one`, `load`, `save`, `unblock_card`,
    `__init__`) are **all present in main's `packages/grib/src/grib/kanban.py`**
    — so the *implementation* was absorbed, not lost.
  - But main's `packages/planner/tests/` holds only `test_engine.py` — **no
    kanban tests for `planner`** — and main's kanban coverage sits on
    `packages/grib/tests/test_kanban.py`, a *different* implementation
    (grib 916 L vs planner 541 L, 27 shared methods).
  - None of sep's 47 test names appear anywhere on main.

  So: implementation absorbed, tests not. Whether grib's tests cover the same
  behaviours under different names was **not** established — that needs a
  human read, and it is the one item here I would not decide alone.
- **`sloughGPT-frontend`** — a coherent feature: `components/kanban/*` (6),
  `components/files/*` (5), `lib/file-format.ts` + test.
- **`swift-comet`** — a complete `consciousness/debug/` component set (8 files).
- **`.wt-stage4`** — `domain/slolib/_internal/kernel_{core,engine,ops}.py` +
  benchmark + test. Branch `feat/kernel-engine` is already MERGED, so this is
  post-merge follow-up.

### Untracked file classification (684 files)

| Class | Count | Examples |
|---|---|---|
| artifact | 506 | `.batch-logs/`, `.gate-logs/`, `.splitwork/`, `tmp_*.json`, `.testmondata-*`, `*.lock` |
| other | 45 | misc |
| source | 133 | ~95 in `sloughGPT-separate`, 18 in `sloughGPT-frontend`, ~20 individual |

## Live session hazard

During the audit, `opencode -s ses_f3cd87c68ffehUGxx3XgspxFQX` was running and
actively working in `.wt-quicktest` — the worktree changed state mid-audit
(`tmp/91-merge` → `tmp/55-merge`, 4,944 unmerged paths resolved to clean).
It is now on `tmp/254-merge`. **Excluded from removal.**

This is why `docs/WORKTREE_RECOVERY.md` forbids sweeping `~/.cache/wt-*` by name
pattern: the source of truth for liveness is `git worktree list` plus a
`/proc/*/cwd` scan, never a glob.

## Prevention — built

`scripts/prune-worktrees.sh` landed on main as `9eda5288f` (card
`20261113_092355_choreworktree-prune-script-remove-only-clean-worktrees-never`).

Default is a **dry run**; `--apply` performs removals. Three gates must all
pass, and none of them is a name pattern:

1. **git's own cleanliness gate** — removal runs *without* `--force`, so git
   itself refuses any worktree with modified or untracked files. Verified
   separately: `git worktree remove` on a dirty tree exits 128 and the file
   survives; the same call *with* `--force` destroys it, which is the one
   thing the script never does.
2. **liveness** — `/proc/*/cwd` scan; a process parked at a prompt is not
   idle, and mtime alone would miss it.
3. **recency** — anything touched within `--max-age` minutes (default 60) is
   skipped even when quiet.

Verified end to end: full dry run over all 34 worktrees (exit 0, correct
verdicts), `--apply` on a disposable target (branch and all 5,127 commits
survived), DIRTY refusal, and the `--list` path.

**It does not run automatically.** Nothing invokes it on a schedule yet — it
is a tool an operator runs, not a background reaper. Wiring it into a hook or
CI would be a separate decision, and a background reaper that can delete
worktrees is exactly the class of thing the 2026-10-21 incident warns
against.

Remaining disk consumers, for reference:

```
749M  ~/.cache/wt-agent-phase2   (live session)
707M  ~/.cache/huggingface
679M  ~/.cache/Cypress
656M  ~/.cache/ms-playwright
506M  .git
```
