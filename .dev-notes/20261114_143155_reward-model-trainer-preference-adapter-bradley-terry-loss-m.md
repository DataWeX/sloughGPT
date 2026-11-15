---
id: 20261114_143155_reward-model-trainer-preference-adapter-bradley-terry-loss-m
status: todo
board: 20261114_143155_reward-model-trainer-preference-adapter-bradley-terry-loss-m
title: "Reward-model trainer: preference adapter + Bradley-Terry loss mode (feeds PPO)"
tags: [training, rlhf, model]
sprint: ""
gh: null
---

# Reward-model trainer: preference adapter + Bradley-Terry loss mode (feeds PPO)

RLHF today has PPO + RewardModel class but NO trainer for the reward model —
an untrained scalar head is noise and PPO would faithfully optimize garbage
(rlhf.py review, this session). Build as adapter + loss mode on
SloughGPTTrainer — NEVER a second training loop (one-trainer rule).

1. **preference adapter**: load (prompt, chosen, rejected) triplets from
   existing owned data — `datasets/feedback/corrections.jsonl`
   (corrected_response=chosen, assistant_response=rejected) + training feed
   ratings (>=4 chosen, <=2 rejected); same sources experience_adapter
   already touches.
2. **loss mode**: Bradley-Terry `L = -log sigmoid(r_chosen - r_rejected)` as
   a loss strategy on SloughGPTTrainer; base model frozen, only scalar head
   (+optionally last layers) in optimizer.
3. **sanity gate**: held-out ranking accuracy — must beat ~70% on unseen
   pairs before PPO may consume it (~50% = noise, block).

Placement: `domain/training/_internal/` alongside `rlhf.py` (RM trainer feeds
PPO as stage 1: SFT -> RM -> PPO; ref model = frozen SFT). Depends on
unified loss-modes card if it lands first; coordinate, do not duplicate the
loss-mode plumbing. Gate: `benchmark_rlhf.py` + affected tests + ruff.

Board card `20261114_143155` (created 2026-11-14T14:31:55Z, priority high).
