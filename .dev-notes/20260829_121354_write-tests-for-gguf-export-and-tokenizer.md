---
id: 20260829_121354_write-tests-for-gguf-export-and-tokenizer
title: Write tests for gguf_export and tokenizer
status: done
tags: testing,training
created: 2026-08-29T12:13:54.738517+00:00
---

Write tests for gguf_export and tokenizer

Wrote 158 tests for gguf_export.py and tokenizer.py. All passing. Covers: GGUFExportConfig, _as_float16, count_layers, detect_architecture, register_architecture, all 13 mapping classes, get_block_mapping, estimate_memory_requirements, list helpers, gpt2_pretokenize, default_pretokenize, SloBPE (train/encode/decode/serialize/decompose/analyze/special tokens), SloUnigram (train/encode/decode/serialize/viterbi/segmentations/scores/decompose/analyze).