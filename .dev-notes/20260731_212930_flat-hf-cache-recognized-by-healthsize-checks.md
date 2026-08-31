---
id: 20260731_212930_flat-hf-cache-recognized-by-healthsize-checks
title: Flat HF cache recognized by health/size checks
status: done
tags: infra,downcraft,cache
created: 2026-07-31T21:29:30.980240+00:00
---

Flat HF cache recognized by health/size checks

is_download_complete now detects HF local-dir/flat layout (top-level files + refs/ + stale .cache locks) across standard ~/.cache/huggingface/hub AND project-local models/hf-cache/hub (walk-up from CWD). Added find_cached_model_dir(); model_size.compute_model_size_gb resolves the existing dir. is_model_cached('Qwen/Qwen2.5-0.5B-Instruct') now True, size 0.92GB. downcraft 38 tests + model_size 26 + download_manager/morph_tokenizer/safetensors_loader/numpy_engine + test_models_router green. Site-packages downcraft synced.