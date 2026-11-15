---
id: 20261107_113837_native-training-jobs-mislabeled-as-qwen-final-save-soulsoul
status: done
board: 20261107_113837_native-training-jobs-mislabeled-as-qwen-final-save-soulsoul
title: "native training jobs mislabeled as Qwen + final save .soul.soul double extension"
tags: ["training", "ui"]
sprint: ""
gh: null
---

# native training jobs mislabeled as Qwen + final save .soul.soul double extension

Evidence job_cbc6162a: job record shows model=Qwen/Qwen2.5-0.5B-Instruct but TrainerConfig trained a native slonet transformer (n_embed=128 n_layer=4 vocab=5871) saved to models/slonet-native/*.soul. Root cause: useTrainingForm.ts startTraining always sends model: selectedModel regardless of method=native (apps/web/hooks/useTrainingForm.ts ~line 350 area, l.model=b). Also final save path was models/qt_trained.soul.soul (double .soul). Fix: omit/override model field for method=native (show slonet-native), and strip duplicate extension in save path.
