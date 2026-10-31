---
id: 20260924_058_step4_delete_plural_tree
status: done
title: Step 4 — delete plural tree + packaging cleanup
tags: [architecture, migration, dual-tree]
---
# Step 4 — delete packages/core-py/domains

## Prereqs (doc decision point 4)
- (2) reals migrated: plural reals = 0 ✓ (cards 047/048/051/052/054)
- (3) plural imports retargeted ✓ (card 057)
- green suite ✓

## Phase A retarget list
- examples/{quick_train,lora_train,inference}.py; verify_dpo.py ×3
- tests/test_sloughgpt_trainer_smoke.py docstring
- scripts/setup.sh: --cov=domains→domain; benchmark block (create_engine/benchmark_model = dead both namespaces, read context); quickstart.sh sanity import
- sloughgpt_colab.ipynb: 5 cells (slonet prose, train_pipeline, export, quant_core, quantization) + tests/test_sloughgpt_colab_notebook.py (sys.path root not core-py)
- docstring -m domains.* usage: inference_engine, auto_ingest, world_driver, realm_live ×6, phoneme_encoder_cli ×8
- loose domains. prose ~20 files (database.billing?, status, registry, tokenizer, lm_eval_char, train_pipeline, checkpoint_utils, export, slonet, structured_log, commands, vector_stores, apps routers/schemas/sse_fallback...)
- docs path refs packages/core-py/domains ×10 (skip plans/* historical)
- Makefile quant-core: → domain/infrastructure/_internal/quant_core/*.c
- packaging: setup.py include (domains*→domain*), root pyproject (package-data key, include, 2 per-file-ignores), core-py pyproject include
- colab notebook + tests

## Phase B
git rm -r packages/core-py/domains; REPO_STRUCTURE_MIGRATION.md updates.

## Gates
contract 9/9, broad suite FAILED-set vs baseline, ruff, probes: import domains → ModuleNotFoundError (expected), domain.* OK, packaging find sanity.

<!-- claimed by opencode 2026-09-25T07:55:00+01:00 — base merged with main@c8fb631dd -->

## Result (2026-09-25)

Merged to main as `68872d00c` (Phase A `a29f5d5b8` + Phase B `8e1a0c75c`, branch `feat/step4-delete-plural`).

- Phase B: deleted `packages/core-py/domains` (297 files, 5871 deletions); staged re-homed plural-only assets that were sitting untracked (`domain/inference/SPEC.md`, gpu `engine.c/h`, `dx12.c`, `metal.c`, `vulkan.c`, `shaders/`, pugqeep `API.md`/`README.md`, shell `bios.asm`) — they would have been lost from HEAD otherwise.
- Residual fixes beyond the plan: `tests/test_training_pipeline.py` loaded the pipeline from the plural path by file (10 failures → fixed, retargeted to `domain/infrastructure/_internal/training_pipeline.py`); `tests/contract/test_scatter_gate.py` `PROD_ROOTS` shrank to `domain` only; gpu `Makefile` smoke import → `domain.infrastructure._internal.gpu.gpu_engine` (smoke-verified); `docs/features/CONSCIOUSNESS_PLAN.md` path refs retargeted (consciousness → `domain/cognition/`, managers/workflow → `_internal` paths); contract comment/docstring updated (shims deleted); `conftest.py` newline.
- Gates: ruff clean; contract 9/9; tests/server 1895/1895; root broad suite FAILED-set identical to main baseline (after the 10 training_pipeline fixes; `test_cli_chat` 5 errors + inference_generate 6 + feedback 2 + rag 1 all reproduce verbatim on main); apps suite 30 failures = exactly the known pre-existing 7 files; `python -S` probe `find_spec('domains') → None`; `import domain` OK; packaging includes `domain*` only (no `domains*` anywhere in setup.py/pyproject).
- Caveat: the shared venv's editable finder maps `domains` to the main workspace checkout (absolute path), so a plain `import domains` still resolves until that checkout updates to main — probe verified with `-S` (site-packages excluded).
- Left in place per plan: `docs/plans/*` historical path refs; `.kanban/board.json` legacy historical strings; harmless `sys.path.insert(packages/core-py)` bootstraps (dir still exists for remaining core-py content).
