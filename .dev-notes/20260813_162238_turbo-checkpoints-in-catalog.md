---
id: 20260813_162238_turbo-checkpoints-in-catalog
title: Turbo checkpoints in catalog
status: done
tags: training,turbo,backend
created: 2026-08-13T16:22:38.195831+00:00
---

Turbo checkpoints in catalog

Turbo-trained models (models/turbo-trained/) appear in GET /auto-train/checkpoints with source=turbo; loadable/deletable/downloadable/exportable. Added _find_checkpoint() resolver (CHECKPOINTS_DIR then TURBO_DIR). 4 catalog tests added earlier.

Load-for-chat button on TurboCard completion: basename from turboResult.model_path -> trainingJobsController.loadCheckpoint(name), success/failure toast, Loading state. 2 TurboCard tests.

Turbo badge in checkpoint catalog: Checkpoint.source field added to souls-controller.ts; ResultsStep renders a 'Turbo' Badge when source==='turbo'. 1 ResultsStep test.

BUG FIX (found via new real-training test): _run_turbo stored the raw TrainResult whose best_eval_loss=inf crashed /auto-train/turbo/status with json.dumps(allow_nan=False) ValueError (500). Added _finite_payload() recursive sanitizer (dataclass->dict, inf/nan->None), applied at result storage AND turbo_status read. Real training end-to-end test added: start-turbo with tiny real config (n_embed=32, 1 layer, ~600 char corpus) trains in ~0.6s -> status complete -> checkpoint in catalog with source=turbo -> load_checkpoint success. Fixed _reset_turbo fixture to patch TURBO_DIR/CHECKPOINTS_DIR/LORA_DIR alongside REPO_ROOT (previously stale import-time dirs).

SOUL METADATA FIXES (train_pipeline.py): (1) SloughGPTTrainer.save() now passes final_train_loss=self._last_train_loss to create_soul_profile (was default 0.0) so soul top-level final_train_loss is the real value; also mirrored in the metadata dict. (2) _build_training_state_metadata() gained include_optimizer_state=False param which drops the bulky per-param momentum buffers (optimizer['state']), keeping hyperparameters+t so resume rebuilds a fresh-momentum optimizer; SloughGPTTrainer.save() passes False. Verified via real training: final_train_loss now 4.20 (was 0.0), optimizer keys now ['hyperparameters','t'], soul 81KB for tiny model. Resume/restore path still works (load_state_dict try/except -> fresh optimizer).

Verification: turbo router 12/12, auto-train+training e2e 67/67, train_pipeline 133, training-related core-py 214 passed/18 skipped, slo_format 58, resume/restore 10/10, tsc 0 errors.