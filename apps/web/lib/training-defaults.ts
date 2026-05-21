/**
 * Default training hyperparameters for the web UI.
 * Keep in sync with ``apps/api/server/training/schemas.py``
 * (`TrainRequest` / `TrainingRequest` optional field defaults).
 *
 * On the training host, periodic `step_*.pt` under `checkpoint_dir` embeds char vocab
 * (`stoi` / `itos` / `chars`) for `cli.py eval` — see repo `docs/policies/CONTRIBUTING.md`
 * (*Checkpoint vocabulary*).
 */
export type TrainingMixedPrecisionDtype = 'bf16' | 'fp16'

export const TRAINING_API_DEFAULTS = {
  epochs: 3,
  batch_size: 32,
  learning_rate: 1e-3,
  n_embed: 128,
  n_layer: 4,
  n_head: 4,
  block_size: 128,
  /** Steps between progress logs / UI ``on_progress`` train_loss updates. */
  log_interval: 10,
  /** Steps between eval passes (eval loss on job + progress callback). */
  eval_interval: 100,
  dropout: 0.1,
  weight_decay: 0.01,
  gradient_accumulation_steps: 1,
  max_grad_norm: 1.0,
  use_mixed_precision: true,
  mixed_precision_dtype: 'bf16' as TrainingMixedPrecisionDtype,
  warmup_steps: 100,
  min_lr: 1e-5,
  scheduler: 'cosine',
  use_lora: false,
  lora_rank: 8,
  lora_alpha: 16,
  checkpoint_dir: 'checkpoints',
  checkpoint_interval: 500,
  save_best_only: false,
  max_checkpoints: 5,
}
