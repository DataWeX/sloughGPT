---
id: 20261114_151505_consolidate-stray-training-loops-chat-trainer-consciousnesst
status: todo
board: 20261114_151505_consolidate-stray-training-loops-chat-trainer-consciousnesst
title: "Consolidate stray training loops: chat_trainer + consciousness/training.py onto SloughGPTTrainer"
tags: ["training", "debt"]
sprint: ""
gh: null
---

# Consolidate stray training loops: chat_trainer + consciousness/training.py onto SloughGPTTrainer

Known debt (AGENTS.md): three training-loop copies — train_pipeline.py (canonical SloughGPTTrainer), chat_trainer.py, consciousness/training.py. Consolidation rule: migrate WHEN NEXT TOUCHED, never a big-bang sweep (grandfather clause, same as routers). Each migration = dataset adapter feeding the existing loop (or a loss strategy if the loop lacked the mode), delete the copied forward/loss/backward/optimize loop, keep behavior byte-identical, prove with that suite's tests + benchmark_slonet_training.py. Depends on loss-modes + adapter-registry cards — those make each migration a small adapter instead of a loop re-fit. Order by next-touch: whichever of the two files a real feature hits first gets migrated first.
