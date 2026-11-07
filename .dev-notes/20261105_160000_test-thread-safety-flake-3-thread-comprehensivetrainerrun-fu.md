---
id: 20261105_160000_test-thread-safety-flake-3-thread-comprehensivetrainerrun-fu
status: wip
board: 20261105_160000_test-thread-safety-flake-3-thread-comprehensivetrainerrun-fu
title: "test_thread_safety flake — 3-thread ComprehensiveTrainer.run_full_cycle returns non-success (test-core-py CI flake)"
tags: [training, test-flake, ci-flake]
sprint: ""
gh: null
---

# test_thread_safety flake — 3-thread ComprehensiveTrainer.run_full_cycle returns non-success (test-core-py CI flake)

CI evidence (PR #267 rerun, run 42036453877 job 127309721013): tests/core-py/test_training_robustness.py::TestConcurrentSafety::test_thread_safety fails at assert all(r.success for r in results) — 3 threads each run ComprehensiveTrainer().run_full_cycle(data, FAST_CONFIG), no thread exceptions (len(errors)==0 and len(results)==3 pass), but one result.success is False. The captured stdout only shows normal phase logs (validating/configuring/preprocessing) — no error surfaced, so the diagnosis needs the result object's error field printed on failure. Different failure from run 1 of the same PR (test_learner_continual shutdown join) — two runs, two different single failures among ~42,790 tests = flaky concurrency suite under CI load, both in code byte-identical to origin/main (PR #267 touches only apps/web). Precedent: the same suite PASSED fully on PR #268's run (42032079341). Suggested fix direction: log/print r.error (or full result repr) in the assertion message first — diagnose which shared resource the concurrent trainers contend on (checkpoint/temp/state path), then fix the contention or the test's isolation.

## Work log

(claimed by foreman session 2026-11-07, dispatched to Gen)

## Work log (foreman session, 2026-11-07 — Gen hit free-usage limit, absorbed brief)

### Step 1: assertion now surfaces the error (done)
`assert all(r.success for r in results)` replaced with a message that prints
`r.error` + every phase's `(phase, success, error)` for failed results.

### Diagnosis: root cause PROVEN — unguarded `f.stat()` in `_enforce_disk_limit`
`domain/training/_internal/train_pipeline.py` line 2566:

```python
try:
    files = sorted(checkpoint_dir.glob("*.soul"), key=lambda p: p.stat().st_mtime)
except OSError:
    return
total_bytes = sum(f.stat().st_size for f in files)   # ← OUTSIDE the try
```

All 3 threads run `ComprehensiveTrainer.run_full_cycle` with the shared
hard-coded checkpoint dir `models/comprehensive`. Each thread's final
`save_checkpoint(is_final=True)` calls `_enforce_disk_limit` (glob+stat pass)
while another thread's `_prune_stale_checkpoints(keep_final=True)` unlinks
everything but the newest file. A file unlinked between glob and the sum →
`FileNotFoundError` escapes `save_checkpoint` → `train()` → caught by
`_run_phase` → `PhaseResult(success=False, error=...)` →
`run_full_cycle` returns `success=False` with **no thread exception** —
exactly the CI signature (run 42036453877: len(errors)==0, len(results)==3,
one success=False, failure after preprocessing, stdout shows only normal
phase logs).

`_prune_stale_checkpoints` correctly guards every stat/unlink with
try/except OSError; `_enforce_disk_limit` guards the glob but not the sum,
and not the while-loop `oldest.stat()` (both are the same hole).

### Evidence captured
- `race_enforce.py` (unit-level, real methods, 3 threads × 300 rounds,
  40 seeded files) → **FileNotFoundError escaped twice immediately**:
  `No such file or directory: 'models/comprehensive/seed_37.soul'`.
- `stress_flake.py` (6 concurrent full cycles × 8 rounds, 48 runs) →
  0 failures but showed the same contention class: 5× soft
  "Checkpoint not found" warnings from a concurrent prune.
- Port-don't-rebuild: `git log --all -S "_enforce_disk_limit"` → no fix on
  any branch.
- End-to-end repro of the test itself is probabilistic (window ≈ µs; 15/15
  green even with a 61-file seeded dir) — consistent with a rare CI flake.

### Plan (approved direction: fix contention, never weaken assertions)
1. Guard the sum + while-loop stat in `_enforce_disk_limit` (skip on
   OSError — disk-limit enforcement is best-effort; a vanished file means
   another saver is already pruning).
2. Domain touched → `scripts/domain_baseline_rehash.py` + rehash commit.
3. Gates: race_enforce.py → 0 escapes; target test ×20; full file green;
   ruff check + format.

## Benchmark (SOP step 11, 2026-11-07)
`scripts/benchmark_slonet_training.py --only gate` on the PR head (e0a83bd0a):
**GATE PASS** — 55.4K params, 37.4s, 3.21 steps/s, loss 2.9247 → 0.0007 (<10% goal),
PPL 1.0, 25.8 MB, rng-fingerprint 0032a7465041. vs quiet-pixel's recorded 3.46 s/s
= within shared-machine noise; the fix guards stat() calls in checkpoint pruning
and never enters the training loop. No regression.

## PR
- #297 fix/test-thread-safety-flake — rebased ×2 (baseline conflicts each time main
  rehashed; procedure: checkout --ours + hand-apply training hash + recorded_commit).
  First CI on 52b90d07f: test-core-py PASSED, sole reds = route-fallbacks floor
  (fixed on main by #298). Head now e0a83bd0a, CI running.
