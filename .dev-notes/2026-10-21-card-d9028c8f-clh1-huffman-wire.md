---
title: Card d9028c8f — pugqeep storage truth-up: CLH1 Huffman wire + honest nbytes (option B)
created: 2026-10-21
updated: 2026-10-21
tags: pugqeep, compression, codec, wire-format, truth-up
status: done
---

card: d9028c8f-8c62-4e47-8df9-24e43b2507c4
branch: feat/npu-caps (stack tail)

Decision: user chose **option B** — persist the Huffman bitstream on the wire
with a format tag (old dense blobs keep parsing) + fix the nbytes residual
omission. Option A (dense-truth-only) rejected as leaving real compression
on the floor. Stack tail `feat/npu-caps`.

## What landed

- **`ClusterCodec` CLH1 wire format** (`domain/infrastructure/_internal/pugqeep/codecs.py`):
  `[CLH1][n_centroids:4][quantized:1][centroids(+scale+zp:8)]
  [tree_len:4][tree_json][total_bits:4][stream_len:4][bitstream]
  [n_assign:4][_pack_residual]`. Marker leads params, so `buf[:4]`
  dispatches legacy dense vs CLH1. Deserialize materializes dense
  assignments (idempotent re-serialization) and restores the huffman keys.
- **`point.py`**: outer type code `b"CLH1"` in `_TYPE_DECODE` → `"cluster"`;
  `to_bytes` picks CLH1 only when `params["huffman_data"]` is present, else
  legacy CLUS byte-identical. Old readers fail loudly on the new tag.
- **`nbytes` payload contract** (was the double-lie): huffman branch =
  `centroids + len(huffman_data) + tree_json_len + (8 if quantized)
  + residual`; residual counted in EVERY branch; never overstates the wire
  (`nbytes <= len(to_bytes())`); point-vs-residual-twin delta =
  exactly `residual.nbytes`. 5 legacy pins pass unchanged
  (test_pugqeep_point:111, test_point_compressor:162,:192,
  test_point_library:926, test_pugqeep_cache_tasks:134).

## TDD evidence

- RED first: `packages/core-py/tests/test_pugqeep_huffman_wire.py` —
  11 contract tests, 5 failed for right reasons pre-implementation.
- GREEN 11/11 post-implementation. Trap found: the rank-1 test helper
  originally built a 4096x4096 outer product (16.7M symbols) — the
  per-bit Python Huffman decode blew per-test timeouts; helper now makes
  ~n elements (sqrt side). Wire itself verified byte-exact by manual parse
  walk (tree_len/total_bits/stream_len/n_assign/residual all match).
- Full pugqeep suite: **2151 tests, 0 failures, 0 errors, 3 skipped**
  (junit-verified; counts recorded here — the /tmp capture was wiped by a
  server restart). ruff format+check clean.

## 3 old pins updated to the honest contract (suite re-greens)

- `test_pugqeep_compressor.py::TestPointNbytes::test_cluster_nbytes` —
  pinned the old huffman formula (sum-of-code-string-lengths tree, no
  residual); now pins the payload contract.
- `test_pugqeep_checkpoint.py::TestCompressCheckpoint::{creates_files,stats}`
  — pinned `compression_ratio > 1.0` / `>= 1.0`.

## DISCOVERY (honesty surfaced a real product wart — needs user decision)

`compress_checkpoint` (training/executor.py) uses `PointCompressor()`
(default residual_threshold=0.99). Random-init/trained weights k-means'd to
4-16 clusters almost never hit 0.99 base accuracy -> a FULL-SIZE float32
residual is stored per tensor -> honest payload > raw (measured ratio
0.856-0.91 on the test models). Capacity-vs-threshold effect: the B-era
knobs re-run shows k=64 clears 0.99 on all 6 benchmark layers (0 residuals
fired) — no residual, genuine compression. The old `nbytes` omission was
masking that **checkpoint "compression" inflates the library on disk** at
low cluster counts (the residual was always serialized; only the accounting
lied). Options:
(a) lossy checkpoint path — compress_checkpoint passes residual_threshold=0
    (VQ-only, genuinely compresses, reconstruction lossy);
(b) keep lossless-by-default and rename/reframe the stat (it's a storage
    format, not a ratio win).
Not taken in this card — behavior change, out of approved scope.

## Gates

- 2151 pugqeep/point/compression tests green (above) + `test_executor.py`
  44 green (the compress_checkpoint consumer of nbytes, outside the
  26-file suite).
- Benchmarks re-recorded post-B (third history record, B-era accounting):
  - **CLH1 lifts honest ratios across the frontier** (to_bytes ground
    truth): shipped default adaptive_k16_rt0.99 **3.84x -> 4.50x** (+17%);
    fixed_k16_rt0 **3.90x -> 7.49x** (+92% — dense assignments were the
    dominant cost and huffman halves them); fixed_k64_rt0 3.84x -> 4.30x.
  - **nbytes_gap = 306 B per 6-layer config — exactly wire headers**,
    under the 512-B tripwire now in benchmark_compression_knobs.py; the
    payload contract holds on real BERT-scale data.
  - Capacity nuance confirmed in the residual-fired column: k8/16/32 fire
    6/6 at rt0.99 (-> lossless class 0.83-0.92x, residual-dominated);
    k64/adaptive fire 0/6 (accuracy clears the threshold -> genuine
    lossy compression at 4.3-4.5x).
  - vq-approaches third record: shipped_vq **4.00x -> 5.75x**,
    shipped_vq_k16 (lossless) 0.80x -> 0.90x; PQ/RVQ/FSQ/legacy rows
    **byte-identical pre/post** — the wire change moves exactly and only
    huffman-carrying cluster points (regression-safety signal).
- Frontier table pre-B numbers superseded (marker appended to the
  6eaa413c note).
