---
id: 20261115_081635_training-curriculum-staging-recipe-stage-gate-judge-dynamic
status: todo
board: 20261115_081635_training-curriculum-staging-recipe-stage-gate-judge-dynamic
title: "training: curriculum staging recipe + stage-gate judge (dynamic modular layers)"
tags: ["training", "vision", "weights"]
sprint: ""
gh: null
---

# training: curriculum staging recipe + stage-gate judge (dynamic modular layers)

User vision (confirmed 2026-11-14/15): layered training — layer1 our codebase, layer2 books/research PDFs, layer3 internet/synthetic (selfgen.py), layer4 comprehensive compiled, layers5-6 web (CommonCrawl/Wikipedia/audio, stubs). NOT hard-coded to 6: dynamic modular stages via curriculum.json (ordered stages, datasets, mixing ratio, replay % for anti-forgetting, ewc.py). ONE SloughGPTTrainer only (adapter seam), never N loops. Stage-gate judge = player/evaluator (quality_scorer + lm_eval_char) gating checkpoint advance between stages. Fragments exist: quality_scorer, training_advisor, outcome_tracker, lm_eval_char, rlhf, distillation, ewc, selfgen.
