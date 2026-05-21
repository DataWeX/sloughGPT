/**
 * Souls Controller — axios-based API for personality management.
 */

import { apiGet, apiPost, apiClient } from './http-client'

export interface Soul {
  name: string
  description: string
  traits: string[]
  personality: Record<string, number>
}

export interface Checkpoint {
  name: string
  download_url?: string | null
  soul: string
  loss?: number
  steps?: number
  epochs?: number
  size_mb?: number
  tagline?: string
  description?: string
  born_at?: string
  epochs_trained?: number
  final_train_loss?: number
  final_val_loss?: number
  system_prompt?: string
  tags?: string[]
  personality?: Record<string, number>
  lineage?: string
  model_type?: string
  traits?: Record<string, number>
  verdict?: string
  perplexity_delta?: number
  bleu_delta?: number
  tokenizer_type?: string
  vocab_size?: number
}

export interface SoulsResponse {
  souls: Soul[]
  current_soul?: string
}

export interface CheckpointsResponse {
  checkpoints: Checkpoint[]
  active_checkpoint?: string
}

export const soulsController = {
  async list(): Promise<SoulsResponse> {
    return apiGet<SoulsResponse>('/souls')
  },

  async getCurrent(): Promise<Soul | null> {
    try {
      return await apiGet<Soul>('/souls/current')
    } catch {
      return null
    }
  },

  async switch(name: string, checkpointName?: string): Promise<void> {
    const params: Record<string, string> = { name }
    if (checkpointName) params.checkpoint_name = checkpointName
    await apiClient.post('/souls/switch', null, { params })
  },

  async listCheckpoints(): Promise<CheckpointsResponse> {
    return apiGet<CheckpointsResponse>('/auto-train/checkpoints')
  },

  async loadCheckpoint(name: string): Promise<{ status: string; name: string; soul?: string; loss?: number; steps?: number; traits?: Record<string, number>; path?: string }> {
    return apiClient.post(`/auto-train/checkpoints/${encodeURIComponent(name)}/load`).then(r => r.data)
  },
}
