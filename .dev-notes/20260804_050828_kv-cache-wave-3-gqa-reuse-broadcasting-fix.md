---
id: 20260804_050828_kv-cache-wave-3-gqa-reuse-broadcasting-fix
title: KV Cache Wave 3: GQA-reuse broadcasting fix
status: done
tags: slonet,kv-cache,inference
created: 2026-08-04T05:08:28.246369+00:00
---

KV Cache Wave 3: GQA-reuse broadcasting fix

Fixed GQA-reuse broadcasting bug in generate_numpy and generate_numpy_stream. Root cause: kernel GQA expand produces (1, H, new_len, E) shape, but the einsum fallback on reuse (step 0, _start_pos > 0) expected (1, new_len, H, E). Added _use_einsum check that transposes k/v back to einsum-compatible shape when the fused kernels won't be used. Applied to both generate_numpy and generate_numpy_stream. All 21 KV state tests pass, 196 slonet/generate/kernel/quant tests pass.