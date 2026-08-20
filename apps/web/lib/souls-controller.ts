/**
 * Souls Controller — axios-based API for personality management.
 */

import { apiGet, apiPost, apiDelete, authFetch } from './http-client'

export interface Soul {
  name: string
  path?: string
  description: string
  traits: string[]
  personality: Record<string, number>
  born_at?: string
  training_dataset?: string
  epochs_trained?: number
  final_train_loss?: number | null
  final_val_loss?: number | null
  lineage?: string
  base_model?: string
  version?: string
  size_mb?: number
  behavior?: Record<string, unknown>
  cognition?: Record<string, number>
  emotion?: Record<string, number>
  generation_params?: Record<string, unknown>
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
  is_loaded?: boolean
  verdict?: string
  perplexity_delta?: number
  bleu_delta?: number
  tokenizer_type?: string
  vocab_size?: number
  training_dataset?: string
  training_duration_s?: number
  source?: string
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
    const souls = await apiGet<Soul[]>('/souls')
    // _meta is attached as non-enumerable by http-client StandardResponse unwrapper
    const meta = (souls as unknown as { _meta?: { current_soul?: string } })?._meta
    return {souls: souls || [], current_soul: meta?.current_soul}
  },

  async getCurrent(): Promise<Soul | null> {
    try {
      return await apiGet<Soul>('/souls/current')
    } catch {
      return null
    }
  },

  async switch(name: string, checkpointName?: string): Promise<void> {
    const body: Record<string, string> = { name }
    if (checkpointName) body.checkpoint_name = checkpointName
    await apiPost('/souls/switch', body)
  },

  async listCheckpoints(): Promise<CheckpointsResponse> {
    const data = await apiGet<Checkpoint[] | { checkpoints: Checkpoint[] }>('/auto-train/checkpoints')
    const checkpoints = Array.isArray(data) ? data : (data?.checkpoints ?? [])
    return {checkpoints}
  },

  async loadCheckpoint(name: string): Promise<{ status: string; name: string; soul?: string; loss?: number; steps?: number; traits?: Record<string, number>; path?: string }> {
    return apiPost<{ status: string; name: string; soul?: string; loss?: number; steps?: number; traits?: Record<string, number>; path?: string }>(`/auto-train/checkpoints/${encodeURIComponent(name)}/load`)
  },

  // ── Trait Weights ──

  async getTraitWeights(): Promise<{
    personality: Record<string, number>
    cognition: Record<string, number>
    emotion: Record<string, number>
  }> {
    return apiGet('/souls/weights')
  },

  // ── Weight Snapshots ──

  async getModes(): Promise<{
    personality: { label: string; confidence: number; scores?: Record<string, number> }
    memory: { label: string; confidence: number; capacity?: number; scores?: Record<string, number> }
    style: { label: string; confidence: number; scores?: Record<string, number> }
    task: { label: string; confidence: number; scores?: Record<string, number> }
  }> {
    return apiGet('/souls/weights/modes')
  },

  async listWeightSnapshots(): Promise<{ name: string; saved_at?: string; label?: string }[]> {
    const data = await apiGet<{ name: string; saved_at?: string; label?: string }[] | { snapshots: { name: string; saved_at?: string; label?: string }[] }>('/souls/weights/snapshots')
    // Handle both StandardResponse (unwrapped array) and legacy {snapshots: [...]}
    return Array.isArray(data) ? data : (data?.snapshots ?? [])
  },

  async saveWeightSnapshot(name: string): Promise<string> {
    const res = await apiPost<{ path: string } | { status: string; path: string }>(
      `/souls/weights/snapshot/${encodeURIComponent(name)}`
    )
    return res.path
  },

  async loadWeightSnapshot(name: string): Promise<number> {
    const res = await apiPost<{ traits_loaded: number } | { status: string; traits_loaded: number }>(
      `/souls/weights/snapshot/${encodeURIComponent(name)}/load`
    )
    return res.traits_loaded
  },

  async deleteWeightSnapshot(name: string): Promise<boolean> {
    const res = await apiDelete<{ deleted: boolean } | { status: string; data: { deleted: boolean } }>(
      `/souls/weights/snapshot/${encodeURIComponent(name)}`
    )
    return 'deleted' in res ? (res as { deleted: boolean }).deleted : false
  },

  async saveTraitWeights(weights: Record<string, Record<string, number>>): Promise<{ status: string }> {
    return apiPost<{ status: string }>('/souls/weights', weights)
  },

  async deleteCheckpoint(name: string): Promise<{ status: string }> {
    return apiDelete<{ status: string }>(`/auto-train/checkpoints/${encodeURIComponent(name)}`)
  },

  async getSoul(name: string): Promise<Soul | null> {
    try {
      return await apiGet<Soul>(`/souls/${encodeURIComponent(name)}`)
    } catch {
      return null
    }
  },

  async getStats(): Promise<{ total_souls: number; current_soul: string | null; available_souls: string[] }> {
    return apiGet('/souls/stats')
  },

  async checkpointInfo(name: string): Promise<Checkpoint | null> {
    try {
      return await apiGet<Checkpoint>(`/auto-train/checkpoints/${encodeURIComponent(name)}/info`)
    } catch {
      return null
    }
  },

  async downloadCheckpoint(name: string): Promise<Blob> {
    const response = await authFetch(`/auto-train/checkpoints/${encodeURIComponent(name)}/download`)
    if (!response.ok) throw new Error('Download failed')
    return response.blob()
  },
}
