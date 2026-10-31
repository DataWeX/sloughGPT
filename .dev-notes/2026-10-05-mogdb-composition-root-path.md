---
id: 20261005_mogdb_composition_root_path
title: "sloughgpt shell died with No module named 'mogdb' — declare packages/mogdb/src at every composition root"
status: done
date: 2026-10-05
tags: [mogdb, cli, infrastructure, path-bootstrap, root-cause]
---

## Symptom

```
$ ./sloughgpt shell
  Error: No module named 'mogdb'
```

## ROOT CAUSE — a missing composition-root declaration, not a bad import

`mogdb` is **source-only**: it is never pip-installed (`import mogdb` fails in
both the pinned `sloughgpt` conda env *and* base), and lives only at
`packages/mogdb/src`. So whoever composes the process must put it on
`sys.path`. Two roots did; two did not:

| composition root                              | declares `packages/mogdb/src` |
| --------------------------------------------- | ----------------------------- |
| `pytest.ini` `pythonpath`                     | yes (comment documents why)   |
| `Makefile:204` (app-planner tests)            | yes                           |
| `apps/api/server/main.py` path loop           | **NO**                        |
| `cli.py` + `apps/cli/src/cli.py` path loops   | **NO** ← the reported bug     |

Full chain:

```
./sloughgpt -> cli.py -> apps/cli/src/cli.py
  -> ShellREPL.__init__ -> ShellState()
  -> domain/shell/_internal/state.py:31  from mogdb import MogDB   # boom
```

The swallow point hid the traceback: `core/framework.py:681` `run_command`
catches `Exception` and prints only `Error: {e}`, then `sys.exit(1)`.

**Why the API server never visibly broke:** its `PYTHONPATH` (read from
`/proc/<pid>/environ`) is `downcraft:core-py:apps/api/server` — no mogdb
either. It survives only because `startup_history._import_mogdb()` mutates
`sys.path` as an incidental side effect when it happens to import early.
That is an ordering accident, not a declaration. Proven by removing it:

```
FAIL apps.api.server.training.job_store          -> No module named 'mogdb'
FAIL apps.api.server.routers.docstore            -> No module named 'mogdb'
FAIL domain.feedback._internal.database          -> No module named 'mogdb'
FAIL domain.infrastructure._internal.training_pipeline -> No module named 'mogdb'
```

41 `mogdb` import sites exist across `domain/ apps/ services/ packages/core-py`
(excluding tests), **7 of them module-level** — all assuming someone already
declared the path.

## FIX — 3 additive lines at the composition roots

- `cli.py` — add `_MOGDB_SRC` to the path loop
- `apps/cli/src/cli.py` — add `_MOGDB_DIR` to the path tuple
- `apps/api/server/main.py` — add `_MOGDB_ROOT` to the path tuple

Commit `e1e3ef65d` (branch `feat/soul-path-grammar`).

Deliberately **not** done:
- no `domain/**` leaf changes — pushing the insert into `state.py` would make a
  domain leaf own process-global path state and need repeating 41 times
- the two existing self-serve `_import_mogdb()` helpers (`startup_history.py`,
  `groups/db.py`) left intact — they are idempotent and give launch-independence
  when PYTHONPATH is arbitrary (pattern blessed by card `d1f544fb`)
- `pytest.ini` untouched — already correct, and cards `d2803a3c` / `56cf354c`
  hold A/B baselines against it

## Verification (clean env, no PYTHONPATH injected)

- `./sloughgpt shell 'uname'` -> `Linux 7.0.0-34-generic x86_64`, exit 0
- the 4 previously-failing server/domain modules now import
- `ruff check` + `ruff format` + `py_compile` clean
- `apps/cli/tests` -> 421 passed, 1 skipped
- `tests/test_shared_indexes.py` -> 6 passed
- `--help` / `version` / `db status` -> exit 0
- shadow risk: none — `packages/mogdb/src` exposes only `mogdb`

## Findings mined from the run (NOT fixed, separate cards)

1. **Pre-existing test failure, A/B-proven not caused by this fix.**
   `packages/core-py/tests/test_shell_integration.py::TestLocalCommands::test_help_shows_builtins`
   asserts `"health" in out`, but `health` is no longer a registered shell
   builtin (only `domain/shell/_internal/vm_devices.py:874` has an unrelated
   `"health"` key). Reverted all 3 files to HEAD and re-ran: **fails
   identically at HEAD**. Test/impl mismatch, likely from `a170a7694`'s
   `_cmd_help` rewrite.

2. **`.venv` is entirely gone**, but AGENTS.md and `pytest.ini` both still
   assume it exists (pytest.ini references `.pytest-libs` for playwright —
   that dir does still exist). Consequence: `./sloughgpt` runs conda **base
   Python 3.14.7**, not the pinned `sloughgpt` env **3.12.14** — visible in
   the help banner. Needs its own decision (recreate `.venv` vs. point the
   launcher at the conda env).

## Related cards

- `56cf354c` — recorded this root cause as measured fact
  (`+packages/mogdb/src -> 61 passed`); "mogdb is NOT installed in the pinned env"
- `d1f544fb` — blessed the self-serve `sys.path` insert pattern
- `d2803a3c`, `4a1e62fa` — test-config / packaging fallout from the same gap
