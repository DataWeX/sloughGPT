---
id: 20260824_131912_training-schema-validation-shell-fixes
title: Training schema validation & shell fixes
status: done
tags: backend,shell
created: 2026-08-24T13:19:12.691031+00:00
---

Training schema validation & shell fixes

Added Field constraints to TrainingRequest (name/model required), DistillStartRequest (all fields bounded). Added try/except to train distill/hf epoch parsing.