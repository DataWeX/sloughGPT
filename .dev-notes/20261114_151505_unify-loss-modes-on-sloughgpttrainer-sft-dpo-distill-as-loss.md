---
id: 20261114_151505_unify-loss-modes-on-sloughgpttrainer-sft-dpo-distill-as-loss
status: todo
board: 20261114_151505_unify-loss-modes-on-sloughgpttrainer-sft-dpo-distill-as-loss
title: "Unify loss modes on SloughGPTTrainer: SFT / DPO / distill as loss strategies (no new loop)"
tags: ["training", "architecture"]
sprint: ""
gh: null
---

# Unify loss modes on SloughGPTTrainer: SFT / DPO / distill as loss strategies (no new loop)

General-purpose trainer for our model (user ask, this session). Today the ONE loop (SloughGPTTrainer, train_pipeline.py) does next-token SFT only; distillation lives on a separate path (distill_gpt2.py), DPO doesn't exist, RLHF has its own update path in rlhf.py. Refactor the train step so the loss is a declared strategy: (a) SFT = current cross-entropy on shifted targets (behavior byte-identical); (b) DPO = preference-pair loss (chosen/rejected, ref-model logprobs, beta KL term) fed by preference pairs — the lighter alternative to full PPO; (c) distill = existing distill loss moved onto the loop. Loop does: batch -> forward -> strategy.loss(...) -> backward -> optimize; strategies are pluggable callables, NOT branches baked into the loop. Explicitly out of scope: multimodal/vision, new frameworks, retroactive loop merges (consolidation card handles chat_trainer/consciousness/training.py incrementally). Gate: existing training tests + benchmark_slonet_training.py must be unchanged for SFT mode; new DPO mode gated by its own tests. Depends: none; RM-trainer card's Bradley-Terry mode should land as a strategy HERE if this lands first (coordinate — do not build loss plumbing twice).
