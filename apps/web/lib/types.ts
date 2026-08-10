/**
 * Canonical data types for the sloughGPT API.
 *
 * These are the single source of truth for all frontend types.
 * Controllers import from here instead of defining their own copies.
 *
 * Backend source: packages/core-py/domains/feedback/workflow.py:get_status()
 * Backend source: apps/api/server/routers/*.py response models
 */

// ── Workflow ──

export interface WorkflowConfig {
  aggregate_interval_minutes: number
  prune_interval_minutes: number
  export_interval_hours: number
  auto_dpo_interval_minutes: number
  health_check_interval_seconds: number
  background_training_interval_seconds: number
  background_training_enabled: boolean
}

export interface WorkflowStats {
  workflow_runs: number
  aggregations_performed: number
  prunes_performed: number
  exports_performed: number
  feedback_recorded: number
  auto_train_steps: number
  dpo_train_steps: number
  dpo_train_rejected: number
  user_adapter_trained: number
  user_adapter_rejected: number
  start_time: number | null
}

export interface WorkflowLastRuns {
  aggregate: number
  prune: number
  export: number
  dpo: number
  health_check: number
  last_rollback: number
  background_training: number
}

export interface WorkflowStatus {
  running: boolean
  stats: WorkflowStats
  pending_thumbs_up: number
  auto_train_threshold: number
  config: WorkflowConfig
  last_runs: WorkflowLastRuns
  systems: Record<string, unknown>
}

// ── Model ──

export interface ModelInfo {
  id: string
  name: string
  type?: string
  source?: string
  description?: string
  tags?: string[]
  size_mb?: number
  size_gb?: number
  params?: string
  cached?: boolean
  loaded?: boolean
  thumbnail?: string
}

// ── Dataset ──

export interface DatasetInfo {
  id: string
  name: string
  source?: string
  type?: string
  size?: number
  samples?: number
  created_at?: string
  tags?: string[]
  vlm_metadata?: {
    image_count: number
    has_video: boolean
  }
}

// ── Training ──

export type TrainingJobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface TrainingJob {
  id: string
  name?: string
  status: TrainingJobStatus
  progress: number
  model?: string
  dataset?: string
  method?: string
  epochs?: number
  current_epoch?: number
  global_step?: number
  loss?: number
  train_loss?: number
  eval_loss?: number
  checkpoint?: string
  data_source?: string
  created_at?: string
  started_at?: string
  completed_at?: string
  error?: string
  explanation?: string
  loss_history?: number[]
  reward_history?: number[]
  result?: Record<string, unknown>
  metrics?: Record<string, unknown>
  epochs_completed?: number
  status_message?: string
}

// ── Checkpoint ──

export interface CheckpointInfo {
  name: string
  soul?: string
  loss?: number
  steps?: number
  epochs?: number
  size_mb?: number
  tagline?: string
  description?: string
  created_at?: string
  born_at?: string
  epochs_trained?: number
  final_train_loss?: number
  final_val_loss?: number
  system_prompt?: string
  tags?: string[]
  personality?: Record<string, number>
  traits?: Record<string, number>
  is_loaded?: boolean
  verdict?: string
  perplexity_delta?: number
  bleu_delta?: number
  tokenizer_type?: string
  vocab_size?: number
  training_dataset?: string
  training_duration_s?: number
  lineage?: string
  model_type?: string
}

// ── Soul ──

export interface SoulInfo {
  name: string
  description: string
  traits: string[]
  personality?: Record<string, number>
}

// ── Adapter ──

export interface AdapterInfo {
  id: string
  name: string
  user_id?: string
  created_at?: string
  updated_at?: string
  size_mb?: number
  rank?: number
  quality_score?: number
  verdict?: string
  perplexity_delta?: number
  bleu_delta?: number
  is_active?: boolean
}

// ── Feedback ──

export interface FeedbackStats {
  db_stats: {
    conversations: number
    messages: number
    feedback_total: number
    thumbs_up: number
    thumbs_down: number
    ratio: number
  }
  current_weights: {
    temperature: number
    repetition_penalty: number
  }
  history_length: number
}

export interface TrainingStats {
  feedback_pairs: number
  last_training: string | null
  quality_score: number | null
}
