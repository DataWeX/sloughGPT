---
id: 20260805_082754_real-model-quantization-verification-pplint4-bugfixes
title: Real-model quantization verification + ppl/int4 bugfixes
status: done
tags: benchmark,quantization,bugfix
created: 2026-08-05T08:27:54.606829+00:00
---

Real-model quantization verification + ppl/int4 bugfixes

Verified the quantization benchmark on the real model (Qwen/Qwen2.5-0.5B-Instruct). Fixed the step-14 perplexity log-softmax math bug (stable m - logsumexp(m); old form gave ppl < 1.0 that passed the gate). Fixed int4 quantization to use per-channel scales (was per-tensor -> real-model logits broken, cosine -0.40; tiny int4 cos 0.8524 -> 0.9208). Added fixed _PPL_PASSAGE as the headline ppl ratio source (short 4-6 token prompts are sampling-noise dominated: int4 Q ppl 104-2072). Fixed _csv_output erasing legitimate 0.0 cells. Results: tiny int8 7/7 cos 0.9996 ppl 1.00; int4 7/7 cos 0.9208 ppl 1.00. Qwen int8 7/7 PASS (validate exit 0, cos 0.9868, ppl 1.02, 3.15x); int4 6/7 (quality FAIL cos 0.8366 < 0.85, ppl 2.58). All 109 unit + 74 e2e quantization tests pass. Report regenerated with real-model section.