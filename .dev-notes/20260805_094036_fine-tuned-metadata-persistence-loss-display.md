---
id: 20260805_094036_fine-tuned-metadata-persistence-loss-display
title: Fine-tuned metadata persistence + loss display
status: done
tags: training,backend,frontend
created: 2026-08-05T09:40:36.817672+00:00
---

Fine-tuned metadata persistence + loss display

Fine-tuned model list endpoint now reads authoritative model/dataset/loss/epochs from metadata.json written at job completion (both HF fine-tune and quick-train routes). Falls back to dir-name parse for legacy dirs. Quick-train dirs now include model prefix for consistency. Frontend FineTunedModelsCard shows loss/epochs when present. Tests: 11 router (2 new), 7 card (1 new); 20 related backend tests pass, tsc clean.