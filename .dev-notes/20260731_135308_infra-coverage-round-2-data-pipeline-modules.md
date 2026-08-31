---
id: 20260731_135308_infra-coverage-round-2-data-pipeline-modules
title: Infra coverage round 2: data-pipeline modules
status: done
tags: infrastructure,tests
created: 2026-07-31T13:53:08.203048+00:00
---

Infra coverage round 2: data-pipeline modules

Added 134 tests across 5 previously-untested infra modules: auto_ingest (27), download_manager (34), model_catalog (27), pt_loader (13), training_pipeline (33). All green. Torch-free .pt fixtures built via custom Pickler (persistent_id storage refs + _rebuild_tensor_v2 routing through find_class, protocol 2 to match torch legacy GLOBAL opcodes). ModelCatalog tested against real MogDB temp-dir instances. Full core-py suite: only 7 failures, all in concurrent untracked files (test_slnc_compiler.py x6, test_wandb_server.py x1); regression gate (shell+logging subset) green. Note: another agent closed round-2 note with different 6 modules; this note records the data-pipeline set.