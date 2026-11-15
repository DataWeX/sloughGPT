---
title: Wire remaining routers through feature engines (Build Order #4)
status: done
created: 2026-09-24
tags: [architecture, training, routers, engines]
sprint: current
gh: null
body: |
  Branch feat/wire-routers-engines (from fix/test-debt-044). No commit unless asked.
  Scope (expert-reviewed plan): only the 3 routers that bypassed engines + a contract guard.
  Out of scope: kb.py deep paths, settings.py, voice/knowledge (already green),
  model_stack cache_tags one-liner (deferred — user never answered).

  What changed:
  (1) NEW domain/training/tokenizer_engine.py — TokenizerEngine wraps TokenizerManager +
      TokenTreeManager (lazy properties). Tokenizer surface: is_trained, ensure_trained,
      stats, pretokenize, decompose, analyze, tokenize, detokenize, get_tokenizer, train.
      Tree surface: tree_stats/vocab/merges/similar/embedding/encode/path/decode/lineage/
      matrix/compare, list/save/load/delete/train_tree. Module-level re-exports
      get_tokenizer_manager() / get_token_tree_manager() so tests that patch
      routers.tokenizer.get_tokenizer_manager keep working (name stays bound on router).
  (2) domain/training/engine.py — TrainingEngine cloud section: _get_cloud_provider
      (function-local `from domain.training._internal.cloud import get_provider` so
      unittest.mock.patch on that attribute still intercepts), _job_to_dict,
      list_cloud_jobs, submit_cloud_job, cloud_job_status, cancel_cloud_job
      (all return TrainingResult; response envelopes unchanged).
  (3) domain/training/__init__.py — lazy exports: TokenizerEngine, get_tokenizer_engine,
      get_tokenizer_manager, get_token_tree_manager, TrainingEngine, get_training_engine.
  (4) apps/api/server/routers/tokenizer.py — import get_tokenizer_manager from
      domain.training.tokenizer_engine (was _internal.tokenizer_manager).
  (5) apps/api/server/routers/token_tree.py — import get_token_tree_manager from
      domain.training.tokenizer_engine (was _internal.token_tree_manager).
  (6) apps/api/server/routers/cloud_training.py — all 4 endpoints delegate to
      get_training_engine().*_cloud_* methods (was inline _internal.cloud imports).
  (7) tests/contract/test_router_boundary.py — test_no_new_router_internal_imports:
      shrink-only baseline of 56 `file|module` pairs for real
      `from|import domain.*._internal` statements in apps/api/server/routers/*.py.
      New pairs fail; baseline may only shrink. The 3 rewired routers contribute 0.

  Compatibility notes:
  - Router test patches (routers.tokenizer.get_tokenizer_manager, etc.) still bind
    because routers import those names at module level from the engine facade.
  - Cloud tests patch domain.training._internal.cloud.get_provider; engine re-imports
    get_provider inside _get_cloud_provider on every call → patch hits.
  - TokenizerEngine tree method names verified against TokenTreeManager public API
    (vocab_entries, top_merges, search_merges, list_saved, save, load, delete_saved,
    similar, embedding_info, matrix_summary, encode, path, decode, lineage, compare).
  - Cloud provider API verified: list_jobs(limit), submit_job(config, dataset_id,
    "train.py", {}), get_status(job_id), cancel_job(job_id); CloudTrainingConfig(provider=...).

  Gates (all green):
  - ruff on all touched files: clean
  - tests/contract/test_router_boundary.py: 4 passed
  - apps/api/server/tests/test_cloud_training_router.py: 4 passed
  - apps/api/server/tests/test_tokenizer_router.py + _v2 + test_token_tree_router.py: 42 passed
  - packages/core-py/tests/test_token{izer,izer_manager,izer_router,tree,tree_manager,tree_router}.py
    + test_cloud_training{,_router}.py: 278 passed
  - domain.training._internal imports in the 3 routers: 0 (regex-checked)
status_detail: done
---
