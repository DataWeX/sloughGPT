---
id: 20260928_075_corepy_full_group_hang
title: "Card 075 — full packages/core-py group hangs at test_pugqeep_process_mgmt (fork futex)"
status: done
date: 2026-09-28
tags: [tests, pugqeep, fork, deadlock, suite]
---

# Card 075 — core-py full-group hang at pugqeep spawn tests

## Symptom
`pytest packages/core-py/tests -q` (single process, whole group) freezes
inside `test_pugqeep_process_mgmt.py` — after 7 dots, at ~51% cumulative
(≈ byte 24247 of progress log), main thread `futex_do_wait`, zero
progress for 30+ min. Same hang at combined
`apps/api/server/tests + packages/core-py/tests + tests + apps/cli/tests`
invocation (stalls at same file).

## Evidence it is pre-existing (not card-067 branch)
- Reproduced at `main @ 7dd5cfe22` in clean worktree (preexist-gate):
  byte-identical freeze at [51%], futex, until timeout killed it.
- File alone in fresh process: completes (then-current baseline failures,
  since fixed by `114d76b19`).
- Pair runs (server+file, priority_queue+file, hf_hub+file): complete.
- Only full a→p context hangs → suspected fork-in-multithreaded-pytest
  (earlier files leak non-daemon threads; `SubprocessProcess` uses
  `start_method="fork"`, `monitor()` waits with no timeout).

## Notes
- Hang baseline predates `114d76b19` (pugqeep red→green fix, card 072);
  re-verify against current main before deep work.
- `test_compressed_transfer_wire.py` (untracked, other session) breaks
  root collection unless `--ignore`d — known, untouched.
- The hang left no pytest summary (killed/timeout), so failure-parity
  runs for the group must use per-group invocations.

## Resolution (2026-10-01)

The fork-in-multithreaded-pytest suspicion was correct; three stacked causes:

1. **Leaked daemon pretrain threads** in `domain/multimodal/_internal/manager.py`
   doing active BLAS compute → `os.fork()` in `SubprocessProcess.start()`
   deadlocks. Fixed (L1, branch `fix/pretrain-thread-leak` @ `ce6224d0d`):
   dedupe spawn + `MultimodalManager.stop_pretrain()` + cooperative `cancel`
   in `_pretrain_engine` + autouse conftest teardown.
2. **`TestCmdLogsExplainV2::test_logs_follow`** unbounded `while True`
   follow loop — fixed with SIGALRM interrupt (sibling-test pattern).
3. **`test_tracking` import-patch self-recursion** → RecursionError ×980 →
   pytest INTERNALERROR — fixed by capturing `real_import` first.

Evidence: full `packages/core-py/tests` **completes in 2h17m** (run5:
41,573 passed, no fork hang — previously froze forever at ~51%).
`--timeout=180` (CLI-only) now surfaces stragglers instead of freezing.

Follow-ups: L2 fork→forkserver card `df5affc7-0d8b-4483-a46a-baf2df495766`
(removes the fork hazard class); L3 thread-teardown card
`f005ddab-0f96-4b02-8e52-e2f808d97b5f` (44× timeout zone; branch
`feat/infra-thread-teardown` @ `aabe3dd09`).

## Closed (2026-10-02) — gate complete, merged, pushed

Resumable 7-chunk gate (`~/.cache/slog-journeys/gate_chunks/run_gate.sh`)
finished `ALL_CHUNKS_DONE`: **683F / 41629P / 0 pytest-errors** vs run7
baseline 594F / 41575P / 141E — playwright error class eliminated, per-file
parity 82 identical / 19 differing (3 converted error-class files + 16
improved). Full-suite hang: gone. One more real leak found by the gate and
fixed (`FireAndForgetPool.shutdown()` dropped `_STOP` on a full queue →
2 `faf-worker` daemons leaked per session). Everything landed on `main` in
one merge (`3cc62050a`, pushed; doc `c4a161a85`); L3 card `f005ddab` →
done; cross-session summary in `docs/AGENT_SYNC.md`. L2 (fork→forkserver)
remains open as `df5affc7`.
