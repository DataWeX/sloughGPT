---
id: 20261103_082215_perfquant-parallel-deflate-inflate-drop-the-redundant-copy-i
status: done
board: 20261103_082215_perfquant-parallel-deflate-inflate-drop-the-redundant-copy-i
title: "perf(quant): parallel DEFLATE inflate + drop the redundant copy in Quantine.load_weights"
tags: ["perf", "inference", "model-load"]
sprint: ""
gh: null
---

# perf(quant): parallel DEFLATE inflate + drop the redundant copy in Quantine.load_weights

Measured on the live restart 2026-11-03 (journal, `from_slnc timing`):
  parse=0.01s model=0.01s weights=2.92s convert=0.00s load_state=0.11s total=3.05s
  then 09:17:24 -> 09:17:32 = ~8s to load model.slnc.quant.npz
  process start 09:16:57 -> model_loaded 09:17:37 = ~40s total

The 2.4 GB fp32 .slnc loads in 2.92s (mmap + parallel). The 421 MB
model.slnc.quant.npz takes ~8s because `Quantine.save_weights` uses
`np.savez_compressed` (DEFLATE on all 338 entries) and int8 weights are
high entropy: 505 MB raw -> 421 MB stored = only 1.15x, i.e. 84 MB saved
for 6.2s of single-threaded zlib (measured 82 MB/s warm on lm_head.npy).

Two defects in `Quantine.load_weights`:
1. Serial inflate. zlib releases the GIL, so the 169 arrays can be
   decompressed concurrently; nothing does.
2. `arr = np.asarray(arr).copy(order='C')` — np.load already returns a
   C-contiguous array, so this is a pure 505 MB copy (extra time + peak).

Fix: parallel inflate over N NpzFile handles (one per worker, split the
169 names into groups so each worker parses the central directory once)
+ copy only when not already C-contiguous. Keep npz_stored/mmap as the
alternative (B) only if disk allows — it costs +84 MB and disk is at 98%.

Expected: ~8s -> ~1-2s; peak RSS down ~0.5 GB transient.
Benchmark: scripts/benchmark_model_load.py (new) with a gate on total
load_weights wall time, run against the real Qwen2.5-0.5B artifact.
