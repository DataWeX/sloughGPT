---
id: 20260818_053249_meta-weights-wire-into-generation-pipeline-full-field-suppor
title: Meta weights: wire into generation pipeline + full field support
status: done
tags: feedback,meta-weights,api,status:done
created: 2026-08-18T05:32:49.866674+00:00
---

Meta weights: wire into generation pipeline + full field support

Wired meta weights into the actual generation pipeline (previously computed but never applied). Extended from 2 fields (temperature, repetition_penalty) to all 6 fields (added top_p, top_k, style_bias, confidence_boost). Generation endpoints now auto-adjust params per-user based on feedback history and similar-message vector search. 24/24 tests pass.