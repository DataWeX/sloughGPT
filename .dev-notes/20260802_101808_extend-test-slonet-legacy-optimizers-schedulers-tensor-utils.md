---
id: 20260802_101808_extend-test-slonet-legacy-optimizers-schedulers-tensor-utils
title: extend test_slonet_legacy: optimizers, schedulers, tensor utils (128 tests)
status: done
tags: core,tests,slonet
created: 2026-08-02T10:18:08.853821+00:00
---

extend test_slonet_legacy: optimizers, schedulers, tensor utils (128 tests)

Extended test_slonet_legacy 88->128 then 128->169 (combined with parallel editor additions). Wave A: SloFeedForward SwiGLU + checkpoint npz. Wave B: SloSGD/SloAdam/clip_grad_norm_/LR schedulers/tensor utils (verified vs source: Tensor coerces float32). Wave C (this): kl_div_loss reductions+backward, SloDataset abstract, SloLRScheduler base edges (param_groups-only, empty-groups default 0.0, abstract get_lr, get_last_lr fallback, explicit-epoch step), create_scheduler all branches + unknown ValueError; editor added Tensor.type/JVP, accel fallbacks, embedding 3D, LSTM skip_embed, state_dict, cross-attn JVP, layernorm generation. Fixed latent source bug: Tensor.backward() crashed when a child is an ndarray (kl_div_loss target) - now filters Tensor children, matching forward_grad. Full slonet sweep 538 passed. slonet.py coverage 85%->90%. Commits: d56e791, a6d3dd0, 6e6f7c6 (backward fix + wave C).