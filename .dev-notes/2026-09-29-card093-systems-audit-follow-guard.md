---
id: 20260929_093_systems_audit_follow_guard
title: "Card 093 — systems-pattern audit: fix `logs -f` hang + truth-up VM syscall tables"
status: doing
priority: high
tags: [shell, vm, docs, tests, systems]
sprint: current
gh: null
---

# Card 093 — systems-pattern audit fixes

## Source

User ask 2026-09-29: "check the systems pattern in our codebase" → audit of
kernel / init / devices / VM / TUI layering, then "continue building".

## Findings that survived re-verification against `main`

The first audit pass ran against the shared main tree, which was checked out
on a stale branch (`feat/canonical-embedder`, 202 behind `main`) — its doc-path
drift had already been fixed on `main` by card 076. Re-verified everything on
a fresh worktree from `main` (`9e5666fde`) before touching anything.

Real, still-open on `main`:

1. **`logs -f` blocks forever off-TTY.** `repl._cmd_logs("-f")` runs
   `while True: time.sleep(0.5)` that only exits on `KeyboardInterrupt`.
   `test_shell_repl_more.py::TestCmdLogsExplainV2::test_logs_follow` hangs the
   whole `test_shell_*` run at ~63% (reproduced twice; no `timeout` in
   `pytest.ini`). A second test for the same command was already
   `@pytest.mark.skip(reason="follow loop blocks on stdin")`.
2. **VM syscall tables are stale in two user/agent-facing places.** The real
   numbering is `X86SyscallHandler.SYS_*` in `vm.py` (1=EXIT, 2=READ,
   3=WRITE, … 10=GETPID, … 30=TRAIN_GET_RESULT) and the built-in programs in
   `vm_programs.py` already follow it. The systems SKILL.md table and
   `sloughgpt vm info` listed a table that matches nothing (2=SYS_PRINT,
   3=SYS_SCAN, 8–10=SYS_DEV_*).
3. **SKILL.md Key Files rows named classes that do not exist** in the files
   given (`ServiceDefinition`, `VFSAddon`, `MountTable`, `DeviceSystem` in
   `devices.py`, `Scheduler`/`TensorMemory`/`SyscallTable` in `kernel.py`,
   `SYSCALL_TABLE` as a symbol).

Known and deliberately NOT touched (owned elsewhere / already documented):
- `packages/core-py/tests/test_compressed_transfer_wire.py` — untracked, other
  session, breaks root collection unless `--ignore`d (card 075 note).
- `make test-py` → `cd packages/core-py && .venv/bin/python` but that venv does
  not exist; `make` is not installed in this environment. Workflow debt, not
  fixed here.
- Full-group hang at `test_pugqeep_process_mgmt` (fork futex) = card 075.
- Kernel `SyscallTable.dispatch()` does no permission check (VM path does) —
  RBAC asymmetry, needs its own decision.
- Two `DeviceBus` / two `Scheduler` classes (documented in the skill now, not
  renamed).

## Work

Branch `fix/shell-logs-follow-guard`, worktree `.wt-sysfix` (main tree was
mid-flight on another session's branch — did not edit there).

- `domain/shell/_internal/repl.py` — follow loop now gated on
  `getattr(self.io, "_is_tty", False)`; non-TTY (scripts, `capture_output`,
  `MemoryIO`, TuiIo) prints one snapshot and returns. Interactive TTY
  behaviour unchanged (Ctrl+C).
- `packages/core-py/tests/test_shell_repl_more.py` — added
  `test_logs_follow_non_tty_skips_loop` (regression guard for the hang).
- `.opencode/skills/systems-engineering/SKILL.md` — Key Files rows, full
  30-row INT 0x80 syscall table from source of truth, note on the duplicate
  `DeviceBus`/`Scheduler` names, kernel-vs-VM numbering callout, addon path typo.
- `apps/cli/src/commands/vm.py` — `vm info` syscall + RBAC tables truthed up.

## Verification (2026-09-30)

Baseline via a throwaway detached worktree pinned at `9e5666fde` (my branch's
exact base — main had moved to `5962f1742`, so comparing against `main` would
have been unfair). Worktree removed after the run. No `git stash` used: shared
stash ref raced me on 09-29 and cost two recoveries.

- **Hang gone.** Full `packages/core-py/tests/test_shell_*.py` run completes in
  ~11 min with the fix (baseline run: same ~11 min). Previously it stalled at
  ~63% on `test_logs_follow`.
- **No regressions, one fix.** Failure sets (stable node IDs, ` - msg` suffix
  stripped): with fix **132**, baseline **133**, and
  `comm` shows the *only* difference is
  `TestCmdLogsExplainV2::test_logs_follow`, which fails on baseline and passes
  with the fix. Fix set ⊂ baseline set ⇒ zero tests broken by my change.
  Remaining 132 failures are pre-existing (`repl_more` 83, `tui_live` 42,
  `commands` 4, +1 each `tui_repl`/`router`/`integration`); no non-`repl_more`
  test references `_cmd_logs`/`logs -f` at all.
- **Logs suites green:** `TestCmdLogsExplainV2` + `log_display` +
  `log_buffer` + `logger` = **208 passed, 0 failed** (exit 0), including the
  new `test_logs_follow_non_tty_skips_loop`.
- **Static:** `py_compile` OK; `ruff check` clean; `ruff format --check` →
  3 files already formatted.
- **No test coverage exists** for `apps/cli/src/commands/vm.py` (`vm info`
  tables) or for the SKILL.md tables — verified by grep, so no test to run.
  A short pytest final-count line never appears in captured logs (something
  calls `os._exit`-style teardown, present on baseline too); FAILED-line
  diffs were used instead.

## Follow-up (2026-09-30): Makefile `.venv` gap → **handed to the conda card**

Audit finding was real: `test-py` / `test-py-fast` / `test-py-slow` ran
`cd packages/core-py && .venv/bin/python`, and `packages/core-py/.venv` does
not exist — the targets fail instantly (reproduced: `No such file or
directory`). I drafted a `$(CURDIR)/.venv/bin/python` fix.

**Reverted before commit — user corrected the premise ("we use conda") and
the fix is already owned elsewhere.** Card
`20260929_journey_gates_conda_migration` (worktree `.wt-gates`, branch
`fix/journey-test-gates`, 47 files, uncommitted, awaiting commit go-ahead)
does it properly:

- new `scripts/python` resolver: `SLO_PYTHON` → conda env `sloughgpt`
  (python 3.12.14) → legacy `.venv` → `python3`; `--resolve` prints the
  path; always prepends repo root to `PYTHONPATH`.
- `Makefile`: `PY := $(CURDIR)/scripts/python`, `ENVBIN` for console
  scripts — `test-py`/`test-py-fast`/`test-py-slow` already rewired to
  `$(PY)`, plus `test-py-training`/`test-py-all` off bare `python3`.

My version (hardcoded `.venv`) would have conflicted with that and encoded
the pre-conda model. Coordination rule applied: no duplicate build, reverted
`Makefile` in `.wt-sysfix`, nothing left staged.

Verification still holds — the fix/132-vs-baseline/133 comparison ran both
sides on the same interpreter, and the two logs tests also pass under the
conda env itself (`sloughgpt`, Python 3.12.14, pytest 9.1.1):
`test_log_follow` (un-skipped) and `test_logs_follow_non_tty_skips_loop`.

Conda caveats found while verifying (both belong to the conda card, not
fixed here): the stray regular package `~/.local/lib/python3.12/site-packages/tests`
shadows the repo's namespace `tests/`, so a bare conda pytest run dies with
`ModuleNotFoundError: No module named 'tests.conftest'` — their pending
`tests/__init__.py` (untracked in `.wt-gates`) is the fix, and
`PYTHONNOUSERSITE=1` is the stopgap I used. The conda env also lacks
`pytest-timeout` (root `.venv` has it), so `--timeout=N` only works on the
legacy interpreter.

## Shared-workspace acceptance

- Indexes: no new app/package/doc added — `apps/README.md`,
  `packages/README.md`, `docs/INDEX.md` unchanged (edit-in-place only).
- Shared copies: reused root `.venv` (ruff/pytest/mypy), no installs, no new
  lockfiles or `node_modules`.
- Coordination: **hand-off to card 081** (kanban content hashes) — 2026-09-30
  the user asked for a read-out of the hash implementation; written into
  `2026-09-29-card081-kanban-content-hashes.md` as "Delivery verification",
  read-only, no code and no board edits. Findings: TS core dead, Python core
  deleted on every ref, `commits[]` never populated, `.planner/hashtrees/
  trees.jsonl` = sole surviving record of the 2026-09-01 card generation →
  adopt-never-recompute. Also handed off: Makefile `.venv` gap → conda card.
- Coordination: main tree untouched (it sat on `feat/canonical-embedder`);
  other sessions' untracked `.opencode/skills/...SKILL.md` + `vm.py` lint-sweep
  edits left alone — my SKILL.md edits are confined to the Key Files +
  syscall sections, so expect a text-level conflict at merge.
