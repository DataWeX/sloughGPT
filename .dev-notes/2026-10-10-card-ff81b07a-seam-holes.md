---
id: ff81b07a
title: "Close 4 stalling seam holes (asyncio.to_thread) — models.py + training/router.py fenced to card 3f749b34"
status: review
board_id: Main
date: 2026-10-10
branch: fix/seam-holes-main
tags: [gene, backend, async, seam, event-loop, apps/api]
---

# Card ff81b07a — seam holes (batch of 2026-10-10, story: c550503c)

Batch note: `2026-10-10-gene-seam-event-loop-starvation.md`.

## Status

**Claimed** by Gene batch 2026-10-10. Worktree `.wt-seam`, branch
`fix/event-loop-seam`.

## Port-don't-rebuild (settled approach exists — do not reinvent)

Steps 1–3 already landed on `feat/contract-ts-projection`:

- `15cfcdac7` — fixes the two UNCONTESTED sites: `startup.py:1135`
  `proc.wait(10)` → `await asyncio.to_thread(proc.wait, 10)`;
  `profiles.py:96` nvidia-smi → `to_thread(subprocess.run, ...)`; plus
  `tests/api-server/test_async_seam.py` (thread-identity TDD) and
  `scripts/benchmark_event_loop_stall.py`.
- `ff85e6d02` — `routers/mobile.py` HuggingFace fetch off the loop.
- `74d0ecc67` — AST gate `tests/api-server/test_async_seam_scan.py`.
- `e7a42cb81` — gate calibrated 4/4; 4 more stalls closed
  (startup TrainingExecutor.shutdown wait=True, inference_pool
  executor.shutdown/thread.join, main idle-manager shutdown).
- `5a92109a8` — benchmark covers all 3 stall shapes.

## Skipped by design (fenced to card 3f749b34 — another session)

- `apps/api/server/routers/models.py` (site 1) — SKIP, fenced.
- `apps/api/server/training/router.py` (site 2) — SKIP, fenced.
- `tests/server/test_models_router.py`, `tests/server/test_audit_instrumentation.py` — fenced.

## Done when

Ported seams green in `.wt-seam`; seam-scan gate + async-seam tests pass;
ruff clean; full tests/api-server at baseline; no fenced file touched.

## Port result (2026-10-15) — complete, awaiting landing

The `fix/event-loop-seam` chain was stranded from `main` (none of its commits
were ancestors of `main`), so it was **ported**, not merged, onto a branch cut
from `main`: **`fix/seam-holes-main`** (base `47595e3b1`), 9 commits.

| source commit | ported as | note |
|---|---|---|
| `15cfcdac7` | `e9874e244` | the 2 uncontested sites; superseded by an equivalent, not re-picked |
| — | `686ac367e` | corrections on top: test moved to `tests/server/`, `CancelledError` reaping branch, warmed benchmark handler |
| `ff85e6d02` | `c3f01e077` | mobile HuggingFace fetch off the loop |
| `74d0ecc67` | `3ba94ed7c` | **AST gate** `tests/server/test_async_seam_scan.py` |
| `e7a42cb81` | `332a398e1` | gate calibrated 4/4 + 4 stalls closed |
| `015cf28b4` | `4edafd727` | CPU-bound detection + 2 more closed |
| `5a92109a8` | `ed649e5cc` | benchmark covers all 3 stall shapes |
| — | `738aa1157`, `1c450a6d8` | ruff-format after conflict resolution; keyword-args fix (below) |
| — | `041505e9e` | unrelated pre-existing failure fixed: shared-DB preset reset |
| — | `a84901261` | `main` merged in before landing (0 conflicts) |

`576f53baa` (SDK build fix) deliberately skipped — unrelated to this card.

**Port corrections found by measurement, not assumed:**

- Tests were relocated `tests/api-server/` → `tests/server/` — `pytest.ini`
  `testpaths` proves `apps/api/server/tests/` collects nothing, which is where
  git's rename hint tried to put them.
- The `KNOWN` ratchet's hardcoded `routers/models.py:1126` was stale on `main`
  (the fenced `urlopen` sits at 1101). Superseded anyway: `e7a42cb81` keys
  ratchets by `file:dotted-call` — line-independent — so the port takes that
  design rather than the local line-number patch.
- `IGNORED` held `startup.py:self._bg_pool.shutdown`, an attribute `main` does
  not have. Replaced with `self._bg_tree.shutdown`, read from source as
  non-blocking: `Tree.shutdown()` is `status = STOPPED` +
  `self._pool.shutdown(wait=False)` (`pugqeep/engine.py:404`).
- **The source commit had a latent broken test.** `e7a42cb81` passed
  `wait=True` positionally through `to_thread`, so
  `test_startup_routers.py:447` saw `shutdown(True)` and failed. Fixed by
  restoring `main`'s keyword style at both affected sites (this commit) —
  the port now changes only the seam, nothing else.

**Collateral fix (own commit `041505e9e`, not seam work).** The full-suite run
surfaced 2 errors unrelated to this card:
`test_companion_router.py::TestListPresets` errored **at setup** with
`KeyError: 'id'`. Root cause: the presets collection is shared across worktrees
and held a document carrying only MogDB metadata
(`{"_mogdb_meta": {...}, "_id": ...}`, no user fields), while `setup_method`
iterated `col.find()` and indexed `p["id"]` on each. Replaced the loop with
`delete_many({})` — which is exactly the intent the comment already states —
leaving the seeding path untouched. Control: planting 2 key-less docs still
yields 2 passed where it previously errored.

**Open item worth its own card.** That key-less document was *recreated within
96 s* of being read (`_id` and `_created` both changed between two reads), by a
process outside this session's commands, into this worktree's own data copy.
No upsert exists anywhere in the server, and MogDB's own `insert_one({})` tests
use `TemporaryDirectory()`, so the writer is unidentified. Someone is writing
empty documents into a shared collection.

**Gates (all on `fix/seam-holes-main`, after `a84901261` merged `main` in):**

- **Full `tests/server`: 1987 passed / 1 failed / 0 errors** (138.8 s). The one
  failure is the pre-existing `test_models_router.py::TestPrecision::
  test_cpu_path_uses_suggestion` (CUDA-vs-CPU). Proven not ours three ways: the
  file is **byte-identical to `main` and to the branch base** (0-line diff), the
  real change set contains **nothing precision-related**, and it still fails
  with `main`'s own newer quant fix merged in.
- Blast radius — 17 test files touching every changed module:
  **528 passed / 1 failed**, same single pre-existing failure.
- AST scan gate: 14 tests. **Bites** — direct `time.sleep` injected into an
  async handler turns it red. (Cannot resolve import aliases: `import time as
  t; t.sleep()` evades it — a lint-class guard, not a security control.)
- Behaviour seam tests: 5 passed (thread-identity + `CancelledError` reaping).
- `scripts/benchmark_event_loop_stall.py`: **exit 0**, re-run after both the
  keyword-args fix and the `main` merge, since both touched measured handlers:
  4 shipped handlers at **4.0–9.8 ms** against a 62 ms ceiling vs ~250 ms
  pre-fix references — **240–248 ms of stall removed per site**, all 3 shapes.
- `ruff check` + `ruff format --check`: clean on every touched file.

**Landing status:** `main` moved while this branch was in flight (it gained
5 commits — RAG/knowledge/quant ports), so it is no longer a fast-forward;
those were merged in as `a84901261` with **zero file overlap and 0 conflicts**.
`main` is checked out in the worktree `/home/mana/Documents/sloughgpt-api-std`,
so its ref is **not** mine to move — landing belongs to whoever owns that
worktree. The branch is green and ready. The two fenced files were never
touched (0-line diff each).
