---
id: 20261023_cleanup_rows_batch_closure
title: "cleanup-01..15: batch closure — pickaxe evidence + 2 real fixes found (RAG RNG, world_stream print)"
status: done
date: 2026-10-23
tags: [cleanup, kanban, code-quality, python]
---

SCOPE: the 15 contentless `cleanup-NN` [high] todo rows. All 15 titles are
work-records from `2026-09-03-code-cleanup-slnc-refactor.md` (that note has NO
frontmatter status, so sync parked every derived row in todo forever — the
root cause). Verification: pickaxe probes against today's HEAD, not the note's
word. Two rows were NOT stale — they led to real fixes this session.

## REAL FIXES (committed this session)

- **cleanup-03 (part)** — `domain/cognition/_internal/rag.py:297` still
  reseeded the process-global RNG per `_default_embedding` call: the note's
  `np.random.seed -> local RandomState` fix only ever landed on the deleted
  `packages/core-py/domains` tree (dual-tree step-4), never the surviving
  copy (born `5717e1f5f`). Fixed as the note itself specified —
  `np.random.RandomState(hash(text) % 2**32).randn(384)` (same MT19937
  stream → byte-identical vectors, state ownership changes only).
  RED-proven: `tests/test_rag.py::TestDefaultEmbeddingHygiene` fails
  pre-fix (global MT19937 advanced), determinism guard green both sides.
  GREEN: 247 passed (root test_rag + core-py test_rag/test_cognitive_rag/
  test_rag_grounding). Commit `76c815d06`.
- **hygiene-ratchet red found en route** — `tests/test_code_hygiene.py`
  failed 2/5 on the stack: (a) `atomic_write.py:7` bare-except = docstring
  false positive main already fixed AST-based (`b94452570`); (b)
  `world_stream.py:170` = REAL stack-side violation (card a3b39114 /
  `72076fbcf` added an argparse run-summary print) that would break main's
  hygiene gate at landing. Adopted origin/main's ratchet wholesale (file is
  now main's content + 1 allowlist line → `git diff origin/main` = 1 line,
  landing merge auto-resolves) + exempted world_stream in the reviewed
  `PRINT_ALLOWLIST` channel class (same rationale as realm_live.py — python
  -m entry, stdout IS its report). Commit `edf4b960b`. Hygiene 5/5.

## PER-ROW EVIDENCE (closed 2026-10-23)

- **01** Future annotations: `domain/` = 630/630 files carry the import;
  ratchet `test_domain_files_carry_future_annotations` green (hygiene 5/5).
  Original "355 in domains/" was the deleted tree; goal state enforced on
  HEAD (`4e7594a67` line of work).
- **02** Logging system fixes: `LogRecord` = `@dataclass(frozen=True)`
  (base.py:135) with no manual `__slots__`; `WebEventLogger` alive as
  backward-compat alias (`domain/logging/__init__.py:38`); `dashboard_filter`
  carries its logger instance.
- **03** RAG fixes: logging present (11 `logger.` calls), div-guards
  (`max(len,1)`, `if not chunks`), `TextChunk.metadata` plumbing, embedding
  providers injected via `embedding_fn` (strategy), + residual RandomState
  fix landed above. CLOSED WITH FIX.
- **04** print() → logger: `wgpu_be`/`vm_debugger` prints = 0; `vm.py`
  residuals are the guest `PRINT` opcode + VGA/debugger channels — the
  hygiene sweep (note `2026-10-04-cleanup-hygiene-sweep.md`, status done)
  AST-adjudicated all 92 remaining prints as stdout-channel-legit and the
  ratchet now pins the exemptions.
- **05** f-string lazy %s: ratchet `test_no_fstring_logger_calls` green
  (hygiene 5/5) + sweep found 0.
- **06** Unused imports: repo-wide `ruff check .` = 0 findings (F401
  included) this session.
- **07** Logging test failures: `test_logging.py` + `test_logging_config.py`
  = 110 passed / 0 failed this session.
- **08** vm.py BlockDevice: single `class BlockDevice(Device)` above use
  (line 975); `py_compile` clean.
- **09** Untracked/merge-bug tests: 0 conflict markers across
  `domain/ tests/ packages/` (proxy evidence — the 3 files are unnamed in
  the source note).
- **10** Shell cmd tests: 34 shell/related test files, 5533 `def test_`
  (claim was 185+).
- **11** SelfConsistency guard: `if not conclusions: return` present in all
  3 `reasoning/advanced.py` copies.
- **12** 441 new tests/21 files: 21/21 files present; 427 tests live today
  (14 later trimmed by refactors — count claim was as-of-2026-09-03).
- **13** CLI test fixes: `test_training_engine.py` absent (removed as
  claimed); StatusBlock uses real `StringIO`.
- **14** SLNC single-path: `domain/infrastructure/_internal/slnc/compiler.py`
  exists; `SLNCCompiler` referenced at 16 sites (both loaders auto-convert).
- **15** 793 passing: historical work-record (352 CLI + 441 core-py measured
  in-session then; suite is now ~45k).

## ROOT-CAUSE FOLLOW-UP (open, not done here)

The source note `2026-09-03-code-cleanup-slnc-refactor.md` has no
frontmatter (`id:`/`status:`) — any full-board `sync` may keep re-deriving
todo rows from it. Adding `status: done` frontmatter to that note would make
the closure self-consistent (user-approved edit to the journal — proposed,
not done).

Gates: hygiene 5/5; rag suites 247 passed; repo `ruff check .` = 0; board
moves verified single-row diffs.
