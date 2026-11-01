/**
 * Training-launch configuration for the VM console.
 *
 * Pure data and pure functions — no React, no IO — so the assembly tab can
 * validate and render hints while `useVmWorkbench` owns persistence.
 */
export const DEFAULT_MAX_STEPS = 5000
export const MAX_STEPS_LIMIT = 1_000_000

export const DEFAULT_TRAIN_CONFIG = {
  dataset: 'shakespeare',
  epochs: 1,
  lr: 0.001,
  batch_size: 32,
  n_layer: 4,
  n_head: 4,
  embed_dim: 128,
}

export type TrainConfig = typeof DEFAULT_TRAIN_CONFIG

export function buildTrainSource(config: TrainConfig): string {
  const json = JSON.stringify(config)
  return `[BITS 32]

; Launch a training job through the x86 training bridge.
; SYS_TRAIN_START: EAX=28, EBX=config JSON addr.
; Returns EAX = job_id (>=1), or -2 when permission denied.
; Requires the ADMIN role (role selector above).
; The Training card on the right polls the job to completion, shows the
; final result, and offers a Stop button for running jobs.

MOV EBX, config
MOV EAX, 28
INT 0x80
HLT

config: db '${json}', 0`
}

export function clampSteps(n: number): number {
  return Number.isFinite(n)
    ? Math.min(MAX_STEPS_LIMIT, Math.max(1, Math.floor(n)))
    : DEFAULT_MAX_STEPS
}

export function clampTrainConfig(config: TrainConfig): TrainConfig {
  return {
    ...config,
    dataset: config.dataset.trim() || DEFAULT_TRAIN_CONFIG.dataset,
    epochs: Number.isFinite(config.epochs)
      ? Math.max(1, Math.floor(config.epochs))
      : DEFAULT_TRAIN_CONFIG.epochs,
    lr: Number.isFinite(config.lr) && config.lr > 0 ? config.lr : DEFAULT_TRAIN_CONFIG.lr,
    batch_size: Number.isFinite(config.batch_size)
      ? Math.max(1, Math.floor(config.batch_size))
      : DEFAULT_TRAIN_CONFIG.batch_size,
    n_layer: Number.isFinite(config.n_layer)
      ? Math.max(1, Math.floor(config.n_layer))
      : DEFAULT_TRAIN_CONFIG.n_layer,
    n_head: Number.isFinite(config.n_head)
      ? Math.max(1, Math.floor(config.n_head))
      : DEFAULT_TRAIN_CONFIG.n_head,
    embed_dim: Number.isFinite(config.embed_dim)
      ? Math.max(1, Math.floor(config.embed_dim))
      : DEFAULT_TRAIN_CONFIG.embed_dim,
  }
}

export const num = (v: string) => (v.trim() === '' ? NaN : Number(v))

export function trainConfigHints(config: TrainConfig): { label: string; message: string }[] {
  const hints: { label: string; message: string }[] = []
  if (!config.dataset.trim()) {
    hints.push({ label: 'Dataset', message: `using default "${DEFAULT_TRAIN_CONFIG.dataset}"` })
  }
  if (!Number.isFinite(config.epochs)) {
    hints.push({ label: 'Epochs', message: `using default ${DEFAULT_TRAIN_CONFIG.epochs}` })
  }
  if (!Number.isFinite(config.lr) || config.lr <= 0) {
    hints.push({ label: 'Learning rate', message: `using default ${DEFAULT_TRAIN_CONFIG.lr}` })
  }
  if (!Number.isFinite(config.batch_size)) {
    hints.push({ label: 'Batch size', message: `using default ${DEFAULT_TRAIN_CONFIG.batch_size}` })
  }
  if (!Number.isFinite(config.n_layer)) {
    hints.push({ label: 'Layers', message: `using default ${DEFAULT_TRAIN_CONFIG.n_layer}` })
  }
  if (!Number.isFinite(config.n_head)) {
    hints.push({ label: 'Heads', message: `using default ${DEFAULT_TRAIN_CONFIG.n_head}` })
  }
  if (!Number.isFinite(config.embed_dim)) {
    hints.push({ label: 'Embed size', message: `using default ${DEFAULT_TRAIN_CONFIG.embed_dim}` })
  }
  return hints
}
