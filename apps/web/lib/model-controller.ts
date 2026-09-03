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

export interface ProviderDiagnostics {
  providers: Record<string, {
    type: string
    module: string
    text_provider?: string
    processors?: string[]
    model_id?: string
    server?: { type: string; has_circuit_breaker: boolean }
  }>
  default_provider: string | null
  model_state: {
    model: string | null
    model_type: string | null
    tokenizer: string | null
    provider: string | null
  }
  startup_phase: string
}

export interface StartupProgress {
  phase: string
  step: number
  total: number
  message: string
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

  async waitForReady(timeoutMs = 90_000, pollMs = 2_000): Promise<ModelStatus> {
    const start = Date.now()
    while (Date.now() - start < timeoutMs) {
      const s = await this.status()
      if (s.loaded) return s
      await new Promise(r => setTimeout(r, pollMs))
    }
    return this.status()
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

  async unloadModel(): Promise<Record<string, unknown>> {
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

  async getCurrentModel(): Promise<{ model_id: string; model_type: string; device: string; loaded_at: number; quantization?: Record<string, unknown> } | null> {
    try {
      return await apiGet('/models/current')
    } catch (e) {
      _log.warning('Could not get current model', { exception: String(e) })
      return null
    }
  },

  async getCatalog(): Promise<{ models: Array<{ id: string; name: string; type: string; size_gb: number; cached: boolean; source: string }>; count: number }> {
    try {
      return await apiGet('/models/catalog')
    } catch (e) {
      _log.warning('Could not get catalog', { exception: String(e) })
      return { models: [], count: 0 }
    }
  },

  async getCatalogStats(): Promise<{ total_models: number; total_size_gb: number; cached_count: number; sources: Record<string, number> }> {
    try {
      return await apiGet('/models/catalog/stats')
    } catch (e) {
      _log.warning('Could not get catalog stats', { exception: String(e) })
      return { total_models: 0, total_size_gb: 0, cached_count: 0, sources: {} }
    }
  },

  async startDownload(modelId: string, totalBytesHint = 0): Promise<{ status: string; model_id: string }> {
    return apiPost('/models/download', { model_id: modelId, total_bytes_hint: totalBytesHint })
  },

  async getDownloadStatus(modelId: string): Promise<{ model_id: string; status: string; progress: number; bytes_downloaded: number; total_bytes: number; speed_bps: number; error?: string }> {
    return apiGet(`/models/download/${encodeURIComponent(modelId)}`)
  },

  async listDownloads(): Promise<{ downloads: Array<{ model_id: string; status: string; progress: number; bytes_downloaded: number; total_bytes: number; speed_bps: number }>; count: number }> {
    try {
      return await apiGet('/models/downloads')
    } catch (e) {
      _log.warning('Could not list downloads', { exception: String(e) })
      return { downloads: [], count: 0 }
    }
  },

  async cancelDownload(modelId: string): Promise<{ status: string }> {
    return apiPost(`/models/download/${encodeURIComponent(modelId)}/cancel`)
  },

  async retryDownload(modelId: string): Promise<{ status: string }> {
    return apiPost(`/models/download/${encodeURIComponent(modelId)}/retry`)
  },

  async verifyDownload(modelId: string): Promise<{ verified: boolean; model_id: string; error?: string }> {
    return apiPost(`/models/download/${encodeURIComponent(modelId)}/verify`)
  },

  async getEngineStatus(): Promise<{ engine: string; version: string; models_loaded: number; uptime_s: number; memory_usage_mb: number }> {
    try {
      return await apiGet('/models/engine/status')
    } catch (e) {
      _log.warning('Could not get engine status', { exception: String(e) })
      return { engine: 'unknown', version: '0.0.0', models_loaded: 0, uptime_s: 0, memory_usage_mb: 0 }
    }
  },

  async reloadEngine(): Promise<{ status: string }> {
    return apiPost('/models/engine/reload')
  },

  async debugProviders(): Promise<ProviderDiagnostics | null> {
    try {
      return await apiGet<ProviderDiagnostics>('/models/debug/providers')
    } catch (e) {
      _log.warning('Could not get provider diagnostics', { exception: String(e) })
      return null
    }
  },

  async getStartupProgress(): Promise<StartupProgress | null> {
    try {
      return await apiGet<StartupProgress>('/health/startup-progress')
    } catch {
      return null
    }
  },

  async setPrecision(mode: 'auto' | 'fp32' | 'fp16'): Promise<{ mode: string; applied: boolean }> {
    return apiPost('/models/precision', { mode })
  },

  async exportModel(outputPath: string, format: 'sou' | 'safetensors' | 'onnx' | 'gguf' = 'sou', includeTokenizer = true): Promise<{ status: string; output_path: string }> {
    return apiPost('/models/export', { output_path: outputPath, format, include_tokenizer: includeTokenizer })
  },
}

export async function* streamModelEvents(
  modelId: string,
): AsyncGenerator<{ phase: string; progress: number }> {
  const { apiGet } = await import('./http-client')
  while (true) {
    try {
      const data = await apiGet<{ model_id: string; cached?: boolean; status?: string; progress?: number }>(`/models/download/${encodeURIComponent(modelId)}`)
      if (data.cached) {
        yield { phase: 'ready', progress: 100 }
        return
      }
      const status = data.status ?? 'downloading'
      const progress = (data.progress ?? 0) * 100
      yield { phase: status, progress }
      if (status === 'complete' || status === 'error') return
    } catch {
      yield { phase: 'downloading', progress: 0 }
      return
    }
    await new Promise(r => setTimeout(r, 1000))
  }
}
