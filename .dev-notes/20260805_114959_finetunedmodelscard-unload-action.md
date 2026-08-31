---
id: 20260805_114959_finetunedmodelscard-unload-action
title: FineTunedModelsCard unload action
status: done
tags: frontend,finetuned,training
created: 2026-08-05T11:49:59.822421+00:00
---

FineTunedModelsCard unload action

FineTunedModelsCard previously hid the Load button when a model was active but offered no way to unload from the card — the user had to use the chat dropdown's Remove model. Added an Unload button (inline power-off SVG, matching the chat dropdown pattern) shown only for the active model, calling modelController.unloadModel(name) then onLoaded to refresh health/status. Tests: added modelController mock to FineTunedModelsCard.test.tsx + 2 tests (Unload shown for active, unload calls onLoaded); all 231 models/training tests pass, tsc clean.