---
id: 20260813_211336_sloadamw-decoupled-weight-decay-optimizer-trainer-wiring
title: SloAdamW: decoupled weight decay optimizer + trainer wiring
status: done
tags: training,slonet,optimizer
created: 2026-08-13T21:13:36.126842+00:00
---

SloAdamW: decoupled weight decay optimizer + trainer wiring

SloAdamW: decoupled weight decay optimizer (torch.optim.AdamW equivalent), wired into SloughGPTTrainer._create_optimizer.

- slonet.py: SloAdam refactored - extracted _reduce_to_param_shape (sums ALL leading broadcast axes, replacing the old single upd.sum(axis=0) hack) and _update_step (bias-corrected m_hat/(sqrt(v_hat)+eps) in param shape). Legacy coupled-L2 semantics preserved exactly (590 tests green).
- SloAdamW(SloAdam): step() applies the Adam update then p -= lr*wd*p (decoupled, LR-annealed), default weight_decay=0.01. State serialized by name, interchangeable with SloAdam checkpoints.
- train_pipeline.py: _create_optimizer() now returns SloAdamW (verified live lr=0.001 wd=0.01).
- Tests: test_slonet_adamw.py 14 tests - decay semantics, LR-annealed decay, decoupling-vs-L2, clipping, bias correction, state round-trip + SloAdam interchange, autograd linear-regression convergence (MSE<1e-2), reference AdamW equivalence, broadcast-grad reduction before moments (multi-axis).
- Verified: py_compile clean, 590/590 slonet+trainer+scheduler+broadcast tests pass.