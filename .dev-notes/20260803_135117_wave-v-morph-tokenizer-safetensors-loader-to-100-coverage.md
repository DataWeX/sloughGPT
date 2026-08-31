---
id: 20260803_135117_wave-v-morph-tokenizer-safetensors-loader-to-100-coverage
title: Wave V: morph_tokenizer + safetensors_loader to 100% coverage
status: done
tags: coverage,infrastructure,testing
created: 2026-08-03T13:51:17.542000+00:00
---

Wave V: morph_tokenizer + safetensors_loader to 100% coverage

morph_tokenizer.py 100% (401 stmts): added test_stem_ies_suffix (cries/spies hit the previously uncovered stem-form branch at line 636). safetensors_loader.py 100% (139 stmts): covered snapshot-dir branches (_find_safetensors, load_model_config), ValueError on missing safetensors, raw-parser path through load_model_weights, safetensors-package path via a fake safe_open module (BF16 fallback + plain slice), _try_convert_to_slnc success + silent-failure, list_cached_models dedup across hubs. Combined run: 100 passed, 11 skipped (test_morph_tokenizer, test_morph_tokenizer_wave_g, test_safetensors_loader).