---
id: 20260731_210922_infra-coverage-weight-dependent-suites-on-cached-qwen25
title: Infra coverage: weight-dependent suites on cached Qwen2.5
status: done
tags: infrastructure,tests,models
created: 2026-07-31T21:09:22.894521+00:00
---

Infra coverage: weight-dependent suites on cached Qwen2.5

Ran weight-dependent suites against locally-cached Qwen/Qwen2.5-0.5B-Instruct (models/hf-cache/hub flat layout, no downloads). Fixed 3 real bugs: (1) numpy_forward.py T() applied transpose_weights backwards (transposed (in,out) weights); broke non-square GQA k/v projections for Qwen-style checkpoints and would break GPT-2 — inverted to transpose (out,in) at all 3 sites (50/201/369). (2) morph_tokenizer.py:242 + safetensors_loader.py parents[3]->parents[4] (repo-root cache path). (3) list_cached_models() only scanned HF_HOME/hub, missed project-local flat cache — now scans both, dedupes by id. Tests repointed gpt2/Qwen2-0.5B fixtures to Qwen2.5 with skip-if-uncached gating (numpy_engine, morph_tokenizer, safetensors_loader, point_library); _is_cached uses loader primitives (_get_model_dir + _find_safetensors). Results: numpy_engine 27 passed/7 gpt2-skip (8 gen/stream failures fixed), morph_tokenizer 25+11, safetensors_loader+point_library 66 passed incl. 3 NumpyEngine+ModelTree integration tests. Regression (loggers/shell/model) 189 passed. Note: is_model_cached()/downcraft still require standard HF snapshot layout — flat cache only recognized via loader primitives.