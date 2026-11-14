# Cards vq-huffman-04 + -05 — benchmark truth-up & suite gate

date: 2026-10-21
branch: feat/npu-caps (stack tail)
cards: vq-huffman-04 "Benchmark: Lloyd+Huffman vs Q4 block", vq-huffman-05 "Run full test suite"

## What -04 actually was

The benchmark already existed: `scripts/benchmark_block_quantization.py`
(exact title match: "Lloyd+Huffman (cluster) vs Block Q4 vs Block Q8"). Series
cards 01-03 (HuffmanTree, k-means++/adaptive Lloyd, Huffman wiring) had built
the codecs; the benchmark script predates the Huffman integration and was
never run-and-recorded. Port-don't-rebuild: fixed it, ran it, recorded it.

## Three defects fixed (commit on feat/npu-caps)

1. **Non-deterministic seeds** — `hash(name) % 1000` carries PYTHONHASHSEED,
   so runs differed per process (untrackable history). Now `zlib.crc32(name)`
   + `seed=` passed to `compress_cluster` (k-means++ sampling deterministic).
2. **Cluster serialization skipped** — the block-only guard ("cluster uses
   Huffman bitstream") was stale: cards 01-03 landed `ClusterCodec`
   serialize/generate; `Point.to_bytes()` round-trips cluster fine (probe:
   1126 B / nbytes 1163). Now ALL methods serialize-timed AND integrity-gated:
   roundtripped point must decode BIT-EXACT vs first-pass decompression,
   `SystemExit` on mismatch (downcraft SHA-abort house style).
3. **Single-sample timings** — block codecs are sub-ms (noise); now median of
   `--rounds` (default 3). Plus JSON history append to
   `scripts/benchmark_block_quantization.json` (downcraft precedent — no
   compression category in benchmark_results.py).

Ruff clean; run green: 15/15 rows integrity-gated, history appended
(2026-10-20T22:18:32Z, numpy 2.5.3).

## Verdict (avg across 5 realistic NN shapes, low-rank + 0.02-variance)

| codec | ratio | cosine | MSE | encode | decode |
|---|---|---|---|---|---|
| block_q4 | **5.33x** | 0.99695 | 2.45e-06 | **0.9 ms** | 0.2 ms |
| cluster (Lloyd+Huffman) | 3.43x | 0.99983 | 1.33e-07 | 248 ms | 137 ms |
| block_q8 | 3.20x | **0.99999** | **8.39e-09** | 0.9 ms | 0.1 ms |

- **block_q4 = default for weight storage**: best ratio (1.6x over cluster),
  ~275x faster encode, cosine 0.997.
- **block_q8 = fidelity pick**: 160x better MSE than q4, same speed, 3.2x ratio.
- **cluster is dominated on weight-like data**: worse ratio than q4, worse
  fidelity than q8, 275x slower encode. Its niche is data with real cluster
  structure (discrete/heavy-tailed), not gaussian weights — recorded so the
  next quantization default decision cites numbers.

## -05 full-suite evidence

Canonical roots per root `pytest.ini` (`testpaths`: core-py, wire-py, tests/,
apps/cli/tests). Ran with `--continue-on-collection-errors` (the 24 core-py
collection errors abort a plain run — see drift attribution below).

- **pugqeep/compressor gate: 2020 tests, 0 failures** (junit-verified).
- **wire-py: 83/83** (my surface, fresh run inside the canonical-remainder pass).
- **canonical remainder (wire-py + tests/ + apps/cli/tests): 3784 tests,
  8 failed + 2 errors, 14 skipped** — all 10 reds **stash-proven pre-existing**
  (reproduced identically with the benchmark commit stashed):
  `tests/server/test_models_router.py::TestPrecision::test_cpu_path_uses_suggestion`,
  `test_code_hygiene.py` (bare-except + print-allowlist scans),
  `test_docs_api_parity.py` (4), `test_domain_print_allowlist.py`,
  `tests/server/test_companion_router.py::TestListPresets` (2 errors).
  Belong to the drift-baseline lane (cards 56b49cf1 / 33773fce).
- **core-py 24 collection errors**: pre-date my fork — `git merge-base` proves
  `domain/shell/_internal/cmds/dashboard.py` etc. are absent at fork-point AND
  on my side (zero `--diff-filter=D` deletions in domain/shell); origin/main
  gained those modules after I forked. Known refactor-drift (33773fce).
- **core-py full run: 37,929 tests, 412 failures + 103 errors, 136 skipped
  (2922 s)** — the known drift baseline class (683-on-main card `56b49cf1`,
  lower here: different runner/flags). Attribution per overlapping file
  (7 of 108 failing files intersect my stack's 482-file diff — every one
  proven not-mine):
  - `test_shell_tui_live` (42, incl. `CmdModule` AttributeError) +
    `test_shell_integration` / `test_shell_repl_more` /
    `test_filesystem_addon_more` → **fork-point drift**: `CmdModule` +
    `cmds/dashboard.py` / `addons.*` are absent at merge-base AND on my side,
    gained by origin/main after my fork (my stack never touched `cmds/`);
  - `test_ml_types::test_cuda_unavailable_without_torch` → fails identically
    with **origin/main's** `ml_types.py` (real cupy + CUDA on this box);
  - `test_vm_benchmark::TestTensorControlFlow` (4, incl. 30 s timeouts) →
    fails identically with **origin/main's** `vm.py` (timeout at main's
    line 2629 — same run-loop);
  - `test_slolib_gpu` (12) → **identical failure sets** mine vs origin/main's
    `gpu.py` (CUDA-equipped box breaks the expects-CPU selection tests).
  Swap-method used throughout (`git checkout origin/main -- <file>` + single
  file runs + restore); working tree verified byte-identical to pre-swap.

Benchmark script has zero test surface (grep: nothing imports it) — suite
attribution is complete for the change: pugqeep gate + wire-py green, no
new reds anywhere.
