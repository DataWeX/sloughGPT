---
id: 20261101_110410_fixinference-prefill-attention-numpy-fallback-allocates-oseq
status: done
board: 20261101_110410_fixinference-prefill-attention-numpy-fallback-allocates-oseq
title: "fix(inference): prefill attention numpy fallback allocates O(seq^2) -- the real cause of 1 tok/s (kernel OOM)"
tags: ["inference", "performance", "memory"]
sprint: ""
gh: null
---

# fix(inference): prefill attention numpy fallback allocates O(seq^2) -- the real cause of 1 tok/s (kernel OOM)

SYMPTOM: server-side inference measured ~0.36 tok/s over HTTP while the in-process decode benchmark measured 17.0 tok/s (47x gap). Root cause is NOT the KV cache (kv_quant = 1.5% of decode) and NOT a leak (210 GETs across 17 endpoints + 9 POSTs left RSS flat at 1.98 GB, exactly 1 model load, 21 threads).

ROOT CAUSE: the numpy fallback in domain/training/_internal/slonet_kernels.py fused_attention_multi (lines 788-797; the numba call is line 786) materializes ~4 concurrent (H, seq, seq) float32 temporaries: scores, scores+causal, exp(...), attn/sum. With H=16 and max_seq_len=32768 (from Qwen max_position_embeddings):
  seq 4096 -> 4.0 GB, seq 8192 -> 16.0 GB, seq 16384 -> 64.0 GB
total_len = min(prompt_len + max_new, 32768) therefore imposes NO practical cap. Observed server anon-rss of 20.5-21.4 GB matches seq~8192 plus 2 GB of weights.

EVIDENCE the server took the numpy fallback rather than numba: thread dumps from the OOM-killed pid 1600 show slonet_kernels.py lines 792/794/795 inside fused_attention_multi, reached from slonet.py:7331 generate_numpy_stream. Line 786 (the numba call) was never the sampled line. numba works fine in a standalone shell (_nb_fused_attention_multi: True) but _ensure_fused() fails SILENTLY inside the server -- that gap is unexplained and is the highest-value fix: the numba path writes only out (seq,H,E_head) = 32 MB and never materializes (H,seq,seq) at all.

THE FIX ORDER (user-approved 2026-11-01):
1. Surface why _ensure_fused() fails inside the server so the numba path runs. This removes the quadratic allocation outright. Numba is present in the env (numba 0.67.0, llvmlite 0.49.0, numpy 2.5.3).
2. Blockwise-chunk the numpy fallback (defense in depth, not the primary lever). Prefill memory must be O(seq), not O(seq^2). Learn the blockwise technique from GGUF/llama.cpp but DO NOT add either as a dependency.
3. Clamp prefill total_len to a sane CPU limit. Scope correction: cap PREFILL only. Generation already streams fine and must NOT be affected.
4. Fix the misleading EMERGENCY message. apps/api/server/routers/inference.py:254 prints a freshly re-read psutil percent, so it reported 56% while the classification that blocked the request was >=95%. It must print the reading that TRIGGERED classification. Proven in journal: 10:49:10 CRI memory_pressure EMERGENCY 96.1% (19074 MB RSS), then 10:50:19 ERR chat "System memory at 56%", then 10:51:52 kernel OOM-killed pid 1600.

HARD CONSTRAINTS: no new training loops -- SloughGPTTrainer / SloChatTrainer adapters only. Work in worktree .wt-prefill on branch fix/prefill-attention-oom (created off main e59d03b19). The main working tree is occupied by another session (fix/audit-a11y-design-tokens, 2001 staged files) -- do NOT touch it. NOTE: slonet_kernels.py is currently STAGED-MODIFIED by that other session (a serial rewrite of both attention kernels adding a GQA reps parameter); coordinate before merging so their work is not clobbered.

VERIFY: (a) the canonical chat request "Count 1 to 10, one per line. Just numbers." returns 1..10 clean with zero leak markers (<|, endoftext, 三明治, <|im_start|>, User:, Assistant:) and does not overrun past 10; (b) POST /inference/generate "count 1 to 5" -> 1..5; (c) a long-prompt soak (>=8k tokens) with RSS watched via /proc must stay under ~4 GB; (d) scripts/benchmark_latency.py does not regress; (e) the chat and tools regression suites are green via scripts/run_affected.sh.
