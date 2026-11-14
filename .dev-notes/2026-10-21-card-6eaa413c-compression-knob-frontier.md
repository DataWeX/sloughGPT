# Card 6eaa413c — compression knob sweep + the storage-accounting double-lie (vq-huffman-07)

date: 2026-10-21
branch: feat/npu-caps (stack tail)
card: 6eaa413c-f66d-4e33-bfde-29c83d79d9ce (created+claimed+closed this session)

## What it was

Frontier follow-up to vq-huffman-06's headline ("fixed k=16 + residual
DOMINATES adaptive-k — 8.22x ratio AND lossless"). Grid: adaptive{on,off} x
n_clusters{8,16,32,64} x residual_threshold{0, 0.99, 1} = 15 configs x the
6-layer BERT suite, real PointCompressor, seed=0. New
`scripts/benchmark_compression_knobs.py` (suite imported from
benchmark_vq_approaches — port-don't-rebuild).

## THE FINDING (bigger than the sweep): storage numbers were double-wrong

First run (ratio = point.nbytes()) showed a red flag: ratio IDENTICAL across
rt0/rt0.99/rt1 at every k while decode flipped lossy->lossless. Investigation
(`codecs.py`):

1. **Huffman is never persisted.** `compressor.py:375` writes `huffman_data`
   into in-memory params; `ClusterCodec.serialize_params` (codecs.py:234-250)
   packs dense uint8 assignments instead; `deserialize_params` never rebuilds
   the tree. The 8.2x Huffman figure dies with the process. Disk = dense
   assignments (+ residual).
2. **`ClusterCodec.nbytes` (codecs.py:217-232) counts the Huffman branch
   (line 225) but omits the residual in BOTH non-plain branches**, while
   serialize packs it (line 249). Empirical proof: k16 rt0.99 stores a
   residual of exactly raw size (2,359,296 B) yet reports the byte-identical
   `nbytes=270,782` as rt0.

So nbytes() lies in BOTH directions: overstates lossy (claims Huffman never
written: adaptive 4.91x reported vs 3.84x honest) and understates/obscures
lossless (residual uncounted; reported ratio unchanged while true stored =
centroids+huffman-fantasy+residual-full-raw). Consumers receiving the lie:
`domain/training/_internal/executor.py:527`, `pugqeep/library.py:461`,
`pugqeep/dedup.py:71` (bytes_saved!), `pugqeep/generic.py:693,696`,
`pugqeep/point_interface.py:121`.

## Corrected frontier (ratio = len(point.to_bytes()) ground truth, 6-layer avg)

| config | ratio | cos | min-cos | resid fired | note |
|---|---|---|---|---|---|
| fixed_k8_rt0 | 3.92x | 0.88054 | 0.84688 | 0/6 | max ratio |
| fixed_k8_rt0.99 / rt1 | 0.80x | 1.00000 | 1.00000 | 6/6 | lossless class |
| fixed_k16_rt0 | 3.90x | 0.97233 | 0.96604 | 0/6 | |
| fixed_k16_rt0.99 / rt1 | 0.80x | 1.00000 | 1.00000 | 6/6 | |
| fixed_k32_rt0 | 3.88x | 0.99257 | 0.98981 | 0/6 | |
| fixed_k32_rt0.99 / rt1 | 0.79x | 1.00000 | 1.00000 | 6/6 | |
| fixed_k64_rt0 / rt0.99 | 3.84x | 0.99820 | 0.99742 | 0/6 | best lossy cos@k |
| fixed_k64_rt1 | 0.79x | 1.00000 | 1.00000 | 6/6 | |
| adaptive_k16_rt0 / rt0.99 (SHIPPED) | 3.84x | 0.99776 | 0.99697 | 0/6 | reference |
| adaptive_k16_rt1 | 0.79x | 1.00000 | 1.00000 | 6/6 | |

Structural truths the honest grid exposes:

- **Ratio is nearly k-independent in lossy mode (~3.8-3.9x)** — dense uint8
  assignments (= n bytes) dominate storage; k trades fidelity only. The whole
  -06 "ratio vs k" curve was an nbytes/Huffman artifact.
- **Lossless (residual) class = 0.79-0.80x raw** — better than raw, but ~4.8x
  WORSE ratio than lossy. Residual is the price of losslessness, not a free
  lunch.
- **fixed_k64 >= adaptive_k16 on this suite at equal bytes** (cos 0.99820 vs
  0.99776) — marginal, worth a note for the defaults question, NOT a change.

## RETRACTION of the -06 headline

"fixed k16 + residual dominates adaptive-k: 8.22x AND lossless" = accounting
artifact (Huffman counted-never-persisted + residual persisted-uncounted).
Honest: k16+residual = **0.80x lossless** vs adaptive **3.84x lossy**
(0.21x ratio of shipped). Corrected in: benchmark_vq_approaches.py docstring
(retraction paragraph) + its fresh history run + AGENT_SYNC + -06 note.

Re-run of benchmark_vq_approaches.py (to_bytes accounting) — conclusions:
- HOLDS: shipped beats PQ (0.90x, 13.6x MSE), RVQ (0.34x), legacy (tied
  ratio 1.01x but 13.8x worse MSE).
- FLIPS/strongens: **FSQ = 1.04x shipped's honest ratio, 77x faster encode**
  (10.9ms vs 835ms) at 28x MSE — the one exploration signal got stronger.

## Gates

ruff clean both scripts; both runs green (exit 0); history appended
(benchmark_compression_knobs.json + benchmark_vq_approaches.json, second
record now carries honest accounting).

## Follow-up filed (needs user approval — wire/defaults decision)

Card (todo): ClusterCodec.nbytes truth-up + the persist-Huffman question —
option A (nbytes reports dense truth, safe) vs option B (serialize the
Huffman bitstream, recovers real compression but changes the wire layout
against the deliberate "legacy layout" comment). Consumers listed above ride
whichever is chosen.

## SUPERSEDED BY OPTION B (2026-10-21, card d9028c8f)

The follow-up below was RESOLVED as option B: the Huffman bitstream now
persists on the wire under a self-identifying `CLH1` marker (legacy dense
blobs still parse), and `nbytes` was rewritten to the honest payload
contract (residual in every branch; huffman branch counts the persisted
tree JSON). See `2026-10-21-card-d9028c8f-clh1-huffman-wire.md`.
The ratios in this note's frontier table are pre-B accounting; the B-era
numbers live in the benchmark JSON histories (third record).
