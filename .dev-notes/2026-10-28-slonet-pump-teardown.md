---
id: 2026-10-28-slonet-pump-teardown
title: Fix idle CPU burn — stream pump teardown + honest metrics (SloNetServer)
status: done
merged: 1e5e76345 (PR #231, squash)
sync_entry: db8e703b1
branch: fix/slonet-pump-teardown (deleted, remote+local)
date: 2026-10-28
---

# SloNet pump teardown + metrics honesty

## Symptom
User: web app heavy, PC slow/hot even when idle. Measured: uvicorn (port 8000)
at 381% CPU with 4 native threads ~90% each while idle; load avg 25;
`/openapi.json` took 6.257s (event loop starved).

## Root cause (measured, not guessed)
- 3 chat streams hit `SloNet generation timed out after 120.0s` at 10:56.
- `generate_stream` timeout path (slonet_server.py) set `cancel_event` and
  raised TimeoutError but **never joined the pump thread** — only the
  `asyncio.CancelledError` path joined. Pumps stuck inside native
  `generate_numpy_stream` between token boundaries survived 15+ min at ~90%
  CPU each, holding `_model_lock`.
- Worse: the `finally` block recorded **every** outcome as success
  (`record_success` + `circuit_breaker.record_success()`), so timeouts and
  stream errors zeroed the breaker's failure signal — breaker could never open.
- Third leak: client disconnect (GeneratorExit) had NO teardown at all.
- Fourth hole (caught by the RED test): with no caller-supplied cancel_event
  there was NO cancellation channel to the pump at all.

## Fix (slice 1) — `domain/infrastructure/_internal/slonet_server.py`
- Consumer owns `pump_cancel` (merged with caller event via `_EitherCancel`).
- `outcome` flag: success | timeout | cancelled | incomplete.
- `finally` = the teardown invariant: non-success → set cancel, join pump with
  bounded `TEARDOWN_GRACE_S=5.0` (`_teardown_grace`), log loud warning if the
  pump survives (signal that stuck-inside-native-compute needs process-tier
  enforcement = slice 2).
- Metrics: success only on clean completion; stream error → record_failure;
  timeout keeps its raise-site records; cancelled = neither (client choice).

## RED-first tests — `packages/core-py/tests/test_slonet_server.py`
`TestGenerateStreamTeardown` (6 tests, all but the regression guard were RED
on main):
1. timeout joins pump before raising
2. timeout not recorded as success (metrics + breaker)
3. stuck pump: bounded + logged + no success
4. clean stream still records success (guard)
5. stream error = failure not success
6. abandoned stream (aclose) tears down pump

## Gates
- [x] RED run: 5 failed 1 passed
- [x] full test_slonet_server.py green (90P)
- [x] wider closure (provider suites) green: 200P + 1x; 1 failure pre-existing on main (stash-verified)
- [x] scripts/benchmark_latency.py: Δ−68.2% vs pre-fix baseline (improvement)
- [x] PR #231 merged 1e5e76345 + AGENT_SYNC entry db8e703b1

## Slice 2 (follow-up, NOT this branch)
- Process-tier cancel enforcement via existing ModelServer/InferenceClient
  (inference_protocol.py) — SIGKILL beats thread abandonment for stuck pumps.
- `generate()` non-stream timeout also abandons its executor thread
  (run_in_executor can't kill) — same class, bounded lifetime.
- OMP_NUM_THREADS / OPENBLAS_NUM_THREADS cap so one generation can never
  claim the whole box.
- Core stays sync; PGQ stays event-loop-free; no second framework.
