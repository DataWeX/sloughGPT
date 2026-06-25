import { describe, expect, it } from 'vitest'
import { TRAINING_API_DEFAULTS } from './training-defaults'

describe('TRAINING_API_DEFAULTS', () => {
  it('epochs is a number', () => {
    expect(typeof TRAINING_API_DEFAULTS.epochs).toBe('number')
    expect(TRAINING_API_DEFAULTS.epochs).toBe(3)
  })

  it('batch_size is 32', () => {
    expect(TRAINING_API_DEFAULTS.batch_size).toBe(32)
  })

  it('learning_rate is 1e-3', () => {
    expect(TRAINING_API_DEFAULTS.learning_rate).toBe(1e-3)
  })

  it('n_embed is 128', () => {
    expect(TRAINING_API_DEFAULTS.n_embed).toBe(128)
  })

  it('n_layer is 4', () => {
    expect(TRAINING_API_DEFAULTS.n_layer).toBe(4)
  })

  it('n_head is 4', () => {
    expect(TRAINING_API_DEFAULTS.n_head).toBe(4)
  })

  it('block_size is 128', () => {
    expect(TRAINING_API_DEFAULTS.block_size).toBe(128)
  })

  it('log_interval is 10', () => {
    expect(TRAINING_API_DEFAULTS.log_interval).toBe(10)
  })

  it('eval_interval is 100', () => {
    expect(TRAINING_API_DEFAULTS.eval_interval).toBe(100)
  })

  it('dropout is 0.1', () => {
    expect(TRAINING_API_DEFAULTS.dropout).toBe(0.1)
  })

  it('weight_decay is 0.01', () => {
    expect(TRAINING_API_DEFAULTS.weight_decay).toBe(0.01)
  })

  it('gradient_accumulation_steps is 1', () => {
    expect(TRAINING_API_DEFAULTS.gradient_accumulation_steps).toBe(1)
  })

  it('max_grad_norm is 1.0', () => {
    expect(TRAINING_API_DEFAULTS.max_grad_norm).toBe(1.0)
  })

  it('use_mixed_precision is true', () => {
    expect(TRAINING_API_DEFAULTS.use_mixed_precision).toBe(true)
  })

  it('mixed_precision_dtype is bf16', () => {
    expect(TRAINING_API_DEFAULTS.mixed_precision_dtype).toBe('bf16')
  })

  it('warmup_steps is 100', () => {
    expect(TRAINING_API_DEFAULTS.warmup_steps).toBe(100)
  })

  it('min_lr is 1e-5', () => {
    expect(TRAINING_API_DEFAULTS.min_lr).toBe(1e-5)
  })

  it('scheduler is cosine', () => {
    expect(TRAINING_API_DEFAULTS.scheduler).toBe('cosine')
  })

  it('use_lora is false', () => {
    expect(TRAINING_API_DEFAULTS.use_lora).toBe(false)
  })

  it('lora_rank is 8', () => {
    expect(TRAINING_API_DEFAULTS.lora_rank).toBe(8)
  })

  it('lora_alpha is 16', () => {
    expect(TRAINING_API_DEFAULTS.lora_alpha).toBe(16)
  })

  it('checkpoint_dir is checkpoints', () => {
    expect(TRAINING_API_DEFAULTS.checkpoint_dir).toBe('checkpoints')
  })

  it('checkpoint_interval is 500', () => {
    expect(TRAINING_API_DEFAULTS.checkpoint_interval).toBe(500)
  })

  it('save_best_only is false', () => {
    expect(TRAINING_API_DEFAULTS.save_best_only).toBe(false)
  })

  it('max_checkpoints is 5', () => {
    expect(TRAINING_API_DEFAULTS.max_checkpoints).toBe(5)
  })
})
