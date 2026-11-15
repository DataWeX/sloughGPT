---
id: 20261030_071807_decode-perf-attribution-inference-benchmark-kind
status: done
board: 20261030_071807_decode-perf-attribution-inference-benchmark-kind
title: "Decode perf attribution + inference benchmark kind"
tags: ["perf", "bench", "inference"]
sprint: ""
gh: null
---

# Decode perf attribution + inference benchmark kind

Attributed why the engine "records 3 tok/s" and what merged main actually does (same harness, same minutes, Qwen2.5-0.5B int8, greedy, 64-tok): old stack (no swiglu, no decode lane, foreign checkout) 0.49-0.91 tok/s; old+our swiglu = ~3 tok/s (the recorded figure); MERGED main provider path quantize=True = 23.2-26.6 tok/s (mean 25.3); benchmark_decode_slope marginal slope = 9.8-16.1 tok/s (loadavg 4.3->8.3 swing); fp32 trap: provider WITHOUT quantize=True = 2.65-2.72 tok/s, profile shows lm_head_argmax 59% of time (261ms/token re-reading 544MB fp32 lm_head). New kind: benchmark_results.py inference (KINDS single declaration, decode_tok_s 30% rel gate, loadavg+provider recorded) — baseline record decode=9.8 provider=25.3 load=8.29. Gates: ruff check+format, tests/test_benchmark_results_effects 8P, record/history/compare CLI verified. Commits: a23ba5f51 (+ merges d6a680d82 swiglu, fd43c731a web-perf phase2 entry 806->480KB, 5ebd48925 /goal command).
