---
id: 20261101_052145_sft-tool-use-training-dataset-adapter-into-sloughgpttrainer
status: wip
board: 20261101_052145_sft-tool-use-training-dataset-adapter-into-sloughgpttrainer
title: "SFT tool-use training: dataset adapter into SloughGPTTrainer (real lever for agent synthesis quality)"
tags: ["training", "sft", "tool-use", "agents", "model-quality", "area:training"]
sprint: "model-quality"
gh: null
---

# SFT tool-use training: dataset adapter into SloughGPTTrainer (real lever for agent synthesis quality)

User go 2026-11-01. Problem: 0.5B plans [] / refuses / loops; keyword rescue + degeneracy guards (6e0296c8d) are band-aids — synthesis quality ('No specific tool was observed...') needs actual training. Approach (AGENTS rule: NO new training loop): a dataset adapter that emits chat-format tool-use trajectories (plan -> tool calls -> observation -> synthesis) into the ONE SloughGPTTrainer (domain/training/_internal/train_pipeline.py). Data: (a) synthetic trajectories from live agent runs with the keyword-rescue path as teacher signal, (b) refusal-export filtered pairs (3205a5e55) with tool_calls preserved, (c) tiny hand-checked seed set for eval. Gates: scripts/benchmark_slonet_training.py + benchmark_stability.py + a held-out planning set (rescue-rate should DROP after training, i.e. model plans correctly unprompted). Do NOT duplicate chat_trainer.py — extend via adapter or trainer mode. Steps: 1) card-approved data format + adapter skeleton, 2) 50-100 pair seed, 3) short LoRA/few-step run on 0.5B, 4) eval vs pre-fix baseline, 5) iterate data. Coordination: check AGENT_SYNC — other lanes train too; GPU time to be claimed in the card before long runs.
