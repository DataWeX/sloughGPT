---
id: 20260805_112102_fine-tuned-models-end-to-end-surfacing
title: Fine-tuned models end-to-end surfacing
status: done
tags: finetuned,chat,training,backend,frontend
created: 2026-08-05T11:21:02.064604+00:00
---

Fine-tuned models end-to-end surfacing

Surfaced HF fine-tuned models across chat + training + models surfaces. Chat dropdown: dedicated Fine-tuned section (load via POST /training/finetuned-models/{name}/load, active-check on dir name, size MB). Training page: FineTunedModelsCard shows loss/epochs, active highlight via modelController.status(), onLoaded refreshes checkpoints. Backend: _write/_read_finetuned_metadata persists model/dataset/final_loss/epochs; list prefers metadata over dir-name parsing; quick-train output dir now model-prefixed; load_model_path gains identity=dir-name so health/model_type and process-guard worker name report the variant. Chat dropdown dedup: fetchInitialData filters fine-tuned dir names out of availableModels. Fixed stale 'Load model for chat' completion button: was POST /models/load with a dir path (treated as HF id), now routes through trainingJobsController.loadFineTuned(dirName) which handles identity + slnc compile; added test. Verified: 742 training+chat+hooks tests, 35 backend, tsc clean.