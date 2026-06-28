/**
 * Model Controller — axios-based API for model management.
 *
 * Usage:
 *   import { modelController } from '@/lib/model-controller'
 *   const models = await modelController.list()
 *   await modelController.load('gpt2')
 */

import { apiGet, apiPost } from './http-client'

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

export interface ModelLoadResponse {
  status: string
  model_id?: string
  device?: string
  error?: string
}

export interface ModelStatus {
  loaded: boolean
  model_type: string | null
  device: string | null
}

export interface HealthStatus {
  status: string
  model_loaded: boolean
  model_type: string
  summary: string
  is_inferencing?: boolean
  inference_count?: number
  soul_engine_active?: boolean
  soul_name?: string | null
  vocab_size?: number
  block_size?: number
  num_parameters?: number
  device?: string
}

export const modelController = {
  async list(): Promise<ModelInfo[]> {
    try {
      const data = await apiGet<{ models: (string | ModelInfo)[] }>('/models/hf')
      const modelList = data.models ?? []
      return modelList.map((m: string | ModelInfo) =>
        typeof m === 'string' ? { id: m, name: m, type: 'huggingface' } : m,
      )
    } catch (e) {
      console.error('Failed to list models:', e)
      return []
    }
  },

  async getHealth(): Promise<HealthStatus | null> {
    try {
      return await apiGet<HealthStatus>('/health', undefined, { silent: true })
    } catch { return null }
  },

  async load(modelId: string, device = 'auto'): Promise<ModelLoadResponse> {
    const result = await apiPost<ModelLoadResponse>('/models/load', { model_id: modelId, device })
    if (result.status === 'error') throw new Error(result.error || 'Model load failed')
    return result
  },

  async status(): Promise<ModelStatus> {
    try {
      const data = await apiGet<{ model_loaded?: boolean; model_type?: string; device?: string }>('/health', undefined, { silent: true })
      return {
        loaded: data.model_loaded ?? false,
        model_type: data.model_type ?? null,
        device: data.device ?? null,
      }
    } catch {
      return { loaded: false, model_type: null, device: null }
    }
  },

  async info(modelId: string): Promise<ModelInfo | null> {
    try {
      const models = await this.list()
      return models.find((m) => m.id === modelId) || null
    } catch {
      return null
    }
  },

  async loadModelPath(modelPath: string): Promise<ModelLoadResponse> {
    return this.load(modelPath, 'cpu')
  },

  async unloadModel(modelId: string): Promise<Record<string, unknown>> {
    return await apiPost('/models/unload')
  },

  async isLoaded(modelId?: string): Promise<boolean> {
    const status = await this.status()
    if (!status.loaded) return false
    if (modelId && status.model_type !== modelId) return false
    return true
  },

  async loadVisualModel(modelDir: string, modelId = 'visual'): Promise<{ status: string; model_id: string; type: string; vision_encoder?: string; llm?: string }> {
    return apiPost(`/models/visual-load?model_dir=${encodeURIComponent(modelDir)}&model_id=${encodeURIComponent(modelId)}`)
  },

  async getCacheUsage(): Promise<{ total_bytes: number; total_gb: number; model_count: number; cache_dir: string }> {
    return apiGet('/models/cache-usage')
  },
}

export async function* streamModelEvents(
  _modelId: string,
): AsyncGenerator<{ phase: string; progress: number }> {
  yield { phase: 'downloading', progress: 0 }
  yield { phase: 'loading', progress: 50 }
  yield { phase: 'ready', progress: 100 }
}
