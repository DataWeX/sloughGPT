# Card vq-huffman-06 — benchmark_vq_approaches truth-up (shipped-codec baseline)

date: 2026-10-21
branch: feat/npu-caps (stack tail)
card: 202365ec-24bc-4082-ac15-c09a800e7102 (created+claimed this session)

## What it was

Discovered while closing vq-huffman-04: `scripts/benchmark_vq_approaches.py`
framed its OWN local `vq_core` (quantile init, 3 Lloyd iters) as "current
baseline" and listed adaptive-k + k-means++ as PROPOSED approaches — both
shipped into `PointCompressor` by series card vq-huffman-02. Its rows
described a world that no longer existed, and it never exercised the real
codec stack (no Huffman bitstream, no Point, no residual path).

## Truth-up (one rewrite, port-don't-rebuild)

- **Rows anchored on the shipped codec**: `shipped_vq` (real PointCompressor
  defaults: k-means++ init, adaptive_k, 5 Lloyd iters, Huffman assignments,
  quantized centroids, residual path) + `shipped_vq_k16` (adaptive_k=False,
  n_clusters=16 — isolates adaptive-k on the REAL stack).
- **`legacy_quantile_k16`** kept as historical context (the pre-card-02
  baseline), clearly marked, not a candidate.
- **PQ/RVQ/FSQ exploration rows kept** — the genuinely unexplored set;
  documented as as-implemented (no entropy coding — their ratio would
  improve if shipped).
- Median-of-3 timing rounds, bit-exact decode integrity gate (SystemExit),
  JSON history (`scripts/benchmark_vq_approaches.json`, downcraft
  precedent), `--rounds`/`--no-history` CLI. Ruff clean, run green,
  history appended.

## Verdict (avg over 6 BERT-scale layers, make_weight low-rank suite)

| approach | ratio | cosine | MSE | encode |
|---|---|---|---|---|
| shipped_vq (adaptive) | 4.91x | 0.99776 | 1.80e-06 | 1282 ms |
| **shipped_vq_k16** | **8.22x** | **1.00000** | **3.11e-21** | 1052 ms |
| legacy_quantile_k16 | 3.90x | 0.96857 | 2.50e-05 | 348 ms |
| pq_8x16 | 3.47x | 0.96912 | 2.45e-05 | 379 ms |
| rvq_3stage | 1.30x | 0.99400 | 5.14e-06 | 1325 ms |
| fsq_8level | 3.99x | 0.94303 | 5.06e-05 | **22 ms** |

- **No exploration approach beats the shipped codec**: PQ 0.71x ratio + 13.6x
  worse MSE; RVQ 0.26x ratio; FSQ 0.81x ratio + 28x MSE — though FSQ encodes
  ~60x faster (22 ms), the only interesting exploration signal (candidate if
  encode speed ever dominates).
- **The headline finding: fixed k=16 + residual fallback DOMINATES adaptive-k
  lossy VQ on this data** — 8.22x ratio AND lossless (MSE 3.11e-21) vs
  adaptive's 4.91x lossy. Mechanism: k=16 accuracy 0.933 < residual_threshold
  0.99 → compressor.py:357 stores the residual → near-exact at better ratio
  than higher-k pure VQ. Real shipped behavior, not a benchmark artifact —
  cite when tuning n_clusters/residual_threshold for the model stack.
- legacy (old "baseline") beaten by shipped on ratio AND fidelity → card 02's
  improvements confirmed real.

## CORRECTION 2026-10-21 (same day, card 6eaa413c / vq-huffman-07)

The headline below ("fixed k=16 + residual fallback DOMINATES adaptive-k —
8.22x ratio AND lossless") is **RETRACTED as an accounting artifact.** This
script measured shipped rows via `point.nbytes()`, which (a) counts the
Huffman bitstream that `serialize_params` NEVER persists (disk = dense uint8
assignments; huffman_data dies with the process) and (b) omits the residual
that it DOES persist. Both lies run in the same direction for this claim.

Honest ratios (len(point.to_bytes()), re-run recorded in this script's
history): shipped_vq = **3.84x** (unchanged story), shipped_vq_k16 = **0.80x**
lossless — residual is the PRICE of losslessness (~4.8x worse than lossy),
not a dominator. Also flipped: FSQ = 1.04x shipped's honest ratio at 77x
faster encode (its "0.81x of shipped" line was inflated shipped accounting).
Held: PQ/RVQ/legacy dominated honestly; legacy tied on ratio (1.01x) with
13.8x worse MSE.

Full findings + corrected frontier table: card 6eaa413c note
`2026-10-21-card-6eaa413c-compression-knob-frontier.md` +
`benchmark_compression_knobs.json`. Storage-accounting fix filed as a
follow-up card (needs user approval: nbytes-truth-up vs persist-Huffman).
