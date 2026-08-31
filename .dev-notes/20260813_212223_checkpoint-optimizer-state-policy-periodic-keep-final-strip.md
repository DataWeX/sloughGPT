---
id: 20260813_212223_checkpoint-optimizer-state-policy-periodic-keep-final-strip
title: Checkpoint optimizer-state policy (periodic keep / final strip)
status: done
tags: training,checkpoints
created: 2026-08-13T21:22:23.995212+00:00
---

Checkpoint optimizer-state policy (periodic keep / final strip)

Checkpoint metadata policy — made uniform and honest across ALL .soul writers.

OPTIMIZER STATE (design confirmed: periodic keeps / final strips)
- SloughGPTTrainer.save(path, ..., include_optimizer_state=True): kwarg renamed to match internal helper. True embeds step+hyperparams+momentum; False embeds step+hyperparams only (fresh-momentum resume).
- save_checkpoint(metrics, is_final) passes include_optimizer_state=not is_final; train() final save is is_final=True. Periodic interval/eval saves keep momentum for exact crash-resume.
- CheckpointManager.save(): was ALWAYS embedding full momentum regardless of its is_final param — now passes include_optimizer_state=not is_final (aligned with trainer policy), and embeds final_train_loss=metrics.get('loss') with final_val_loss. Docstring updated.
- train_sloughgpt.py already consistent: periodic _save_checkpoint embeds state, final save_model is weights-only.

HONEST METADATA (trainer save())
- final_train_loss: None when never trained (was 0.0); final_val_loss: None when no eval ran (was inf). No fabricated numbers.
- epochs_trained: actual completed epochs (current_epoch+1, 0 when untrained) — was config.epochs. Fixes previously-failing test_epochs.py (expected 2, got 10).

TESTS (all green: 244 passed)
- +test_save_keeps_optimizer_state_by_default, +test_save_checkpoint_periodic/final, +test_save_sou_untrained_losses_are_null
- +test_checkpoint_periodic_keeps_optimizer_state, +test_checkpoint_final_strips_optimizer_state
- test_epochs.py now 2/2 (was 0/2).