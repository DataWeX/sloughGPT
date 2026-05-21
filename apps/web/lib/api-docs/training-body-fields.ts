import type { ApiDocBodyField } from './types'

/** Corpus selectors for POST /train and POST /training/start (TrainDataSourceBody). */
export const TRAIN_CORPUS_BODY_FIELDS: ApiDocBodyField[] = [
  { field: 'dataset', type: 'string', required: false },
  { field: 'manifest_uri', type: 'string', required: false },
  { field: 'dataset_ref.dataset_id', type: 'string', required: false },
  { field: 'dataset_ref.version', type: 'string', required: false },
  { field: 'dataset_ref.manifest_uri', type: 'string', required: false },
]

/** Shared TrainRequest / TrainingRequest hyperparameters (_TrainHyperparameters). */
export const TRAIN_HYPERPARAM_BODY_FIELDS: ApiDocBodyField[] = [
  { field: 'epochs', type: 'number', required: false },
  { field: 'batch_size', type: 'number', required: false },
  { field: 'learning_rate', type: 'number', required: false },
  { field: 'n_embed', type: 'number', required: false },
  { field: 'n_layer', type: 'number', required: false },
  { field: 'n_head', type: 'number', required: false },
  { field: 'block_size', type: 'number', required: false },
  { field: 'max_steps', type: 'number', required: false },
  { field: 'log_interval', type: 'number (default 10)', required: false },
  { field: 'eval_interval', type: 'number (default 100)', required: false },
  { field: 'dropout', type: 'number (default 0.1)', required: false },
  { field: 'weight_decay', type: 'number', required: false },
  { field: 'gradient_accumulation_steps', type: 'number', required: false },
  { field: 'max_grad_norm', type: 'number', required: false },
  { field: 'use_mixed_precision', type: 'boolean', required: false },
  { field: 'mixed_precision_dtype', type: 'string (bf16 | fp16)', required: false },
  { field: 'warmup_steps', type: 'number', required: false },
  { field: 'min_lr', type: 'number', required: false },
  { field: 'scheduler', type: 'string', required: false },
  { field: 'use_lora', type: 'boolean', required: false },
  { field: 'lora_rank', type: 'number', required: false },
  { field: 'lora_alpha', type: 'number', required: false },
  { field: 'checkpoint_dir', type: 'string', required: false },
  { field: 'checkpoint_interval', type: 'number', required: false },
  { field: 'save_best_only', type: 'boolean', required: false },
  { field: 'max_checkpoints', type: 'number', required: false },
  { field: 'device', type: 'string (optional, training host)', required: false },
]
