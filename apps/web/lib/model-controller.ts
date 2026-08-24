/**
 * Model Controller — axios-based API for model management.
 *
 * Usage:
 *   import { modelController } from '@/lib/model-controller'
 *   const models = await modelController.list()
 *   await modelController.load('gpt2')
 */

import { apiGet, apiPost } from './http-client'
import { logger } from './dev-log'
import type { ModelInfo } from './types'

const _log = logger.child('model-controller')

export type { ModelInfo } from './types'

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
  model_loading?: boolean
  model_type: string
  summary: string
  is_inferencing?: boolean
  inference_count?: number
  request_count?: number
  soul_engine_active?: boolean
  soul_name?: string | null
  vocab_size?: number
  block_size?: number
  num_parameters?: number
  device?: string
  quantization?: {
    quantized: boolean
    bits?: number
    mode?: string
    summary?: {
      bits: number
      tensors: number
      avg_cosine_sim: number
      min_cosine_sim: number
    }
  }
}

export interface QuantizationResult {
  quantized: boolean
  bits: number
  mode: string
  model_type: string
  layers_quantized: number
  total_layers: number
  summary: {
    tensors: number
    bits: number
    avg_cosine_sim: number
    min_cosine_sim: number
  }
  per_tensor: Record<string, { scale: number; zero_point: number; cosine_sim: number }>
  avx2_enabled: boolean
}

export const modelController = {
  _listInFlight: null as Promise<ModelInfo[]> | null,

  async list(): Promise<ModelInfo[]> {
    if (this._listInFlight) return this._listInFlight
    this._listInFlight = this._listImpl().finally(() => { this._listInFlight = null })
    return this._listInFlight
  },

  async _listImpl(): Promise<ModelInfo[]> {
    try {
      const data = await apiGet<ModelInfo[] | { models: (string | ModelInfo)[] }>('/models/hf')
      const modelList = Array.isArray(data) ? data : (data.models ?? [])
      return modelList.map((m: string | ModelInfo) =>
        typeof m === 'string' ? { id: m, name: m, type: 'huggingface' } : m,
      )
    } catch (e) {
      _log.error('Failed to list models', { exception: String(e) })
      return []
    }
  },

  async getHealth(): Promise<HealthStatus | null> {
    try {
      return await apiGet<HealthStatus>('/health', undefined, { silent: true })
    } catch (e) {
      _log.warning('Could not health check', { exception: String(e) })
      return null
    }
  },

  async load(modelId: string, device = 'auto'): Promise<ModelLoadResponse> {
    const result = await apiPost<ModelLoadResponse>('/models/load', { model_id: modelId, device })
    if (result.status === 'error') throw new Error(result.error || 'Could not model load')
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
    } catch (e) {
      _log.warning('Could not status check', { exception: String(e) })
      return { loaded: false, model_type: null, device: null }
    }
  },

  async info(modelId: string): Promise<ModelInfo | null> {
    try {
      const models = await this.list()
      return models.find((m) => m.id === modelId) || null
    } catch (e) {
      _log.warning('Could not model info fetch', { exception: String(e) })
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

  async quantize(bits: number = 8, mode: string = 'symmetric'): Promise<QuantizationResult> {
    return apiPost<QuantizationResult>('/models/quantize', { bits, mode })
  },

  async dequantize(): Promise<{ dequantized: boolean; model_type: string; layers_reset: number }> {
    return apiPost('/models/dequantize')
  },

  async getExportFormats(): Promise<Record<string, string>> {
    return apiGet<Record<string, string>>('/models/export/formats')
  },
}

export async function* streamModelEvents(
  _modelId: string,
): AsyncGenerator<{ phase: string; progress: number }> {
  yield { phase: 'downloading', progress: 0 }
  yield { phase: 'loading', progress: 50 }
  yield { phase: 'ready', progress: 100 }
}
