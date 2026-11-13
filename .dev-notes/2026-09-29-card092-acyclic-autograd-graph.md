---
status: done
priority: high
tags: [training, performance, memory, autograd]
board: 44f5bd19-1519-44cd-b577-3cd33caa5f16
landed: 5962f1742 + dd61998cf + 88aa8f3da (main/api-std), b1b39a7e5 + 266ddedea + 61a734574 (feat/acyclic-autograd)
hash: 76689d0b3682994b43149625b45ffd613da5cb39db91c161af3f6deae1b9c39f
---

# Card 092 — Acyclic autograd graph (drop `_consumers` + fwd out self-captures)

## Problem

SloNet's Tensor graph is cyclic by construction, so with `_gc_parked` disabling the
cyclic GC (measured 2.9 → 31 steps/s), every training step's graph becomes
unreachable-but-unreclaimable garbage until an explicit `gc.collect()`:

1. **Bidirectional edges**: every op does `a._consumers.append(out)` (39 sites) while
   `out._children = (a, b)` — a doubly-linked Tensor list. `_consumers` has **zero
   readers** repo-wide (only init/append/clear). `backward()` clears it post-hoc
   (slonet.py:526) — i.e. the design already knows it's a leak.
2. **Self-capturing closures**: 5 `fwd` closures capture the `out` Tensor itself
   (`np.zeros_like(out.data)` / `out.shape`) — `_slice` :1118, `relu` :1303,
   `gelu` :1348, `silu` :1392, `flatten` :4041. `out._forward_fn → closure → out`
   never dies by refcount — and since a self-cyclic node holds `_children` to its
   inputs, it pins the entire ancestral graph. medium (gelu/silu) ≈ +0.77GB/step.

Pre-fix evidence: medium OOM-killed at 18.0G step ~15 (50-step collect window),
bounded only by the RSS-trigger collect (`9e5666fde`), medium peak 11888MB.

## Fix (branch `feat/acyclic-autograd` in `~/.cache/slog-hotloop`)

- Remove `_consumers`: `Tensor.__init__` attr, all 39 appends, `backward()` clear +
  comment. Forward-mode AD reads `_children`/`_forward_fn` only (test file comment
  confirms) — behavior unchanged.
- Rewrite the 5 `fwd` closures to capture `out_shape = out.shape` (+ `out_dtype` for
  `zeros_like` sites) instead of the Tensor.
- Result: pure out→parents DAG → refcount-freed the moment the loss tensor drops;
  GC parking stays as a pure speed knob, RSS trigger becomes belt-and-suspenders.

## Invariants (must hold)

- Gradients byte-identical: gate final loss 1.33e-05, convergence ≤ 0.01 in 120 steps.
- Forward-mode suite `test_slonet_bidirectional_dag.py` (dot-product identities) green.
- No parked-GC speed regression (gate benchmark ≈ 26.8s / profiled 11.77s).
- TDD: RED first — weakref tests must show nodes surviving without gc pre-fix.

## Replace

- `TestConsumersGraph` (4 tests asserting the removed write-only bookkeeping) →
  `TestAcyclicGraph` (weakref + attribute-absence invariants).

## Board

Card `44f5bd19-1519-44cd-b577-3cd33caa5f16` in `.kanban/board.jsonl`.

## Results (2026-09-29)

**Extra root cause found during RED phase**: the nested self-recursive traversals
(`backward()` `build`, `forward_grad()` `topo`, `jvp()` `find_leaves`) self-capture
their own cell, so `topo`/`order` (holding every graph tensor) survived *after every
`backward()`* until a cyclic-GC pass — the master per-step leak, bigger than the 5
fwd closures. Rewritten as order-preserving iterative post-order DFS (stack with
expanded flag, children pushed reversed → identical visit order → byte-identical
float accumulation). All other self-recursive AST hits verified benign
(module-level globals / decorators, no cells).

**Verification**:
- TDD RED→GREEN: `TestAcyclicGraph` (refcount-only freeing with `gc` disabled,
  after backward and after eval) fails pre-fix, passes post-fix; forward-mode suite green.
- 396 gate/training tests pass; slonet suites = 6 failures, all in the clean-main
  baseline (identical node-ids, byte-identical jvp failure message) — zero regressions.
- ruff check + format clean.

**Solo benchmarks** (batch runs are nondeterministic in BOTH codebases — tiny batch
wobbles 3.16–3.26 across runs — so configs must run solo for invariant comparison):

| config | peak before | peak after | elapsed | final loss |
|--------|------------|-----------|---------|-----------|
| gate   | 1808.8 MB  | 26.1 MB   | 41.0 s  | 1.33e-05 (byte-identical) |
| tiny   | 542.5 MB   | 18.9 MB   | 7.3 s   | 3.2590 (byte-identical) |
| small  | 4890.9 MB  | 112.6 MB  | 111.7 s | 2.9037 (byte-identical) |
| medium | 11888.2 MB | 796.6 MB  | 728.1 s | 2.6323 (byte-identical) — **15×** |

Profile driver stash A/B: 15.74 s original vs 15.50 s acyclic (same load) — no perf
regression; benchmark-script elapsed runs higher under acyclic only because
tracemalloc now traces every free (previously cycles deferred frees past the run).
Record: `data/benchmark_results/training/served_2026-09-29T121051821409Z.json`
(compare: no regression vs prior run).

**Stream identity proven at full precision** (`88aa8f3da`): `rng_fingerprint()`
hashes the entire global MT19937 state (624 words + position + gauss cache) to
12 hex chars; the benchmark prints it after every config. Evidence: solo ==
batch position (gate `0032a7465041`, tiny `b1ac06f6861d` in both single-config
and gate→tiny batch runs), and a cross-version A/B vs pre-acyclic `66d752587`
matched all 52 gate `loss_curve` entries at full float64 repr (final
`0.0003480619672761474` both sides) — byte-identity is exact, not 4-decimal
rounding. `TestRngFingerprint` locks stability/isolation/replay/compression-
neutrality; benchmark import guarded so old-slonet baselines still run.

**Batch nondeterminism root-caused + fixed** (`dd61998cf`): batch-position losses
wobble came from `_compress_in_background`'s daemon thread — `compress_checkpoint`
re-constructs SloTransformer via `randn()`, drawing the **global** `np.random`
stream while the next config trains (k-means was already per-tensor seeded).
Fixed with thread-local `isolated_rng()` in slonet (private RandomState under
the block; legacy global stream untouched outside it) wrapping the load in
`executor.compress_checkpoint`. TDD: `TestCompressCheckpointRngIsolation` RED→
GREEN; two identical `gate,tiny` batches now both reproduce tiny=3.2590 (=
solo) and gate=1.33e-05; pugqeep legacy global-seed tests unchanged; ruff clean.
Remaining known quirk: batch-mode peak_memory can include ~1MB of the previous
config's compression allocations — peaks are a solo-run metric.

**Hazards hit**: a `.wt-sysfix` session popped stashes worktree-wide ~12:30 and
clobbered uncommitted `slonet.py`; recovered from `/tmp/opencode/acyclic.diff`,
committed early (`b1b39a7e5`). Landing `5962f1742` via pathspec commit left the
other session's 29-file staged set byte-intact (verified vs snapshot). api-std
`stash@{0}` holds the identical recovered WIP — redundant with the landing, left
untouched for its declared owner.
