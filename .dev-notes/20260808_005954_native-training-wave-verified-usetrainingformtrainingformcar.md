---
id: 20260808_005954_native-training-wave-verified-usetrainingformtrainingformcar
title: Native-training wave verified (useTrainingForm/TrainingFormCard)
status: done
tags: web,native-training,verification
created: 2026-08-08T00:59:54.285746+00:00
---

Native-training wave verified (useTrainingForm/TrainingFormCard)

Verified the editor's web wave: native SloNet training method added to training form (Method + 'native', NATIVE_PRESETS Tiny/Small/Medium/Large, nativeEmbed/Layers/Heads/BlockSize state, applyPreset, body sends n_embed/n_layer/n_head/block_size/checkpoint_dir=models/slonet-native to /auto-train/start) plus custom preset save/load/export/import (localStorage + downloadJson). Traced the full wire: frontend startTraining->startSSETraining->trainingJobsController.startAutoTrain(body)->POST /auto-train/start; backend AutoTrainStartRequest already accepts n_embed/n_layer/n_head/block_size/checkpoint_dir (auto_train.py:151-156) and the handler configures native training from them (auto_train.py:456-489). trainingJobsController.loadFineTuned exists (training-controller.ts:205). Also fixed a latent button bug: disabled={form.canStart} -> disabled={!form.canStart}. Verification: tsc exit 0; useTrainingForm.test.ts 16 + useTrainingForm.presets.test.ts 5 (editor's) + TrainingFormCard.test.tsx 30 = 51 passed; full components/training/ suite 148 passed.