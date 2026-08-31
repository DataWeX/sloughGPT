---
id: 20260805_094722_fine-tuned-model-identity-reporting
title: Fine-tuned model identity reporting
status: done
tags: backend,frontend,chat
created: 2026-08-05T09:47:22.807835+00:00
---

Fine-tuned model identity reporting

Loading a fine-tuned model now reports its dir name as the loaded identity instead of the base id, so health/models/UI distinguish a fine-tuned variant from the plain base model. load_model_path gains identity param (base id still used for tokenizer + process guard); finetuned load route passes name. Frontend: chat dropdown + training card match active state strictly by dir name (removed false-positive base-id fallback). Tests: 43 dropdown/card/hook tests, 20 backend tests, 728 chat/training/hooks, tsc clean.