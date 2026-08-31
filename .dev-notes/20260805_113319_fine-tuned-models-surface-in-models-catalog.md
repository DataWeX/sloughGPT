---
id: 20260805_113319_fine-tuned-models-surface-in-models-catalog
title: Fine-tuned models surface in Models catalog
status: done
tags: frontend,models,finetuned
created: 2026-08-05T11:33:19.074694+00:00
---

Fine-tuned models surface in Models catalog

Models catalog page (/models) previously never showed local fine-tuned dirs — modelController.list() hits /models/hf (HF Hub + cache only). Added <FineTunedModelsCard> to the models page between ModelCatalogCard and ModelPlaygroundCard, wired activeModelId=activeRuntimeId (health.model_type, which reports the fine-tuned dir name when a variant is loaded) and onLoaded={refreshHealth + refetchModels}. Reuses the existing training card — no duplicated UI. Verified: tsc --noEmit clean, 19 models/training component files / 229 tests pass.