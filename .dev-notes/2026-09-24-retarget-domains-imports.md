---
id: 20260924_057_retarget_domains_imports
status: done
title: Retarget all domains.* references to domain.* (dual-tree step 3)
tags: [architecture, migration, dual-tree]
---
# Dual-tree step 3 — clear plural importers

## Inventory (verified on main 6c5c0453b)
- 9 files: domain/{core,cognitive,cognition}/_internal/{processor,base,metacognition} — from domains import → from domain import
- cache/deployment _internal: BaseComponent/ComponentException + ICacheManager try → domain
- routers: benchmark.py ×3, inference.py ×1 late imports
- tests: 5 cognitive/domains_init/metacognition files; test_benchmark_router ×11 patch; test_inference_router ×2 patch; test_train_commands ×6 stale no-op patches (→ domain.inference._internal.slo_embedder)
- startup.py: prewarm list 8 entries + -m domains.infrastructure.inference_engine ×2 → singular (fixes shim-__main__ bug)
- hf_model_worker.py:11 + process_guard.py:850 cls_path strings → domain.infrastructure._internal.hf_model_worker.hf_model_loader
- benchmark_ml_types_comprehensive.py:220 sys.modules key → singular
- repl.py:330 comment; distill_gpt2.py:146 CLI hint (compiler has NO __main__ anywhere)
- docs/*.md import examples

## Intentionally left
- tests/test_benchmarking.py — imports nonexistent domains.ml_infrastructure inside skipped tests (dead; step-4 delete)
- tests/test_sloughgpt_colab_notebook.py — regression guard FOR the plural shim (step-4 updates notebook+test together)

## Plan
branch feat/retarget-domains-imports → baseline targeted → implement → gates (FAILED-set identical) → commit → merge → close.

## Result (done)
- 3 commits: 73e0d7c9b (code retarget, 32 files), b9a69c3f7 (ruff), aec563cdc (docs/skills/Engine export, 26 files) → merged main b00a6ee62
- Gates: ruff clean; suites 11 failed/634 passed = baseline(12) minus test_warms_model_load_graph (FIXED: singular sse_envelope now preloads; plural never existed); contract+targeted 220 passed post-merge; pugqeep 188 passed; B=6; identity+doc-import probes OK; plural reals = 0
- Residuals left intentionally: REPO_STRUCTURE_MIGRATION/PRODUCT_ENGINEERING (describe legacy), OPENCODE_SKILLS dead Component example, QUICKSTART optimized_trainer/torch_runtime (deleted modules), packages/core-py/{README,domains/inference/SPEC.md} (step-4 territory), tests/test_benchmarking.py (dead skipped), colab shim regression guard
- Step 4 (delete packages/core-py/domains/) now unblocked — needs explicit user approval
