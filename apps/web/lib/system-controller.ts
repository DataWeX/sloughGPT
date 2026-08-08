/**
 * System Controller — system metrics, info, disk, detailed health, and output stream.
 *
 * Usage:
 *   import { systemController } from '@/lib/system-controller'
 *   const metrics = await systemController.getMetrics()
 *   for await (const line of systemController.streamOutput()) { ... }
 */

import { apiGet } from './http-client'
import { PUBLIC_API_URL } from './config'

export interface SystemMetrics {
  cpu_percent: number
  memory_percent: number
  memory_used_gb: number
  memory_total_gb: number
}

export interface SystemInfo {
  platform: string
  platform_release: string
  platform_version: string
  architecture: string
  processor: string
  cpu_count: number
}

export interface DiskUsage {
  total_gb: number
  used_gb: number
  free_gb: number
  percent: number
}

export interface GPUInfo {
  backend: string
  device_type: string
  vram_gb: number
  tier: string
  memory_hint: string
}

export interface KvSessionsInfo {
  enabled?: boolean
  active_sessions?: number
  max_sessions?: number
  cached_tokens?: number
  ttl_seconds?: number
  oldest_session_age?: number
}

export interface DetailedHealth {
  status: string
  uptime_seconds: number
  timestamp: string
  request_count: number
  error_count: number
  avg_latency_ms: number
  requests_per_minute: number
  path_latencies: Array<{ path: string; avg_ms: number; count: number; p95_ms: number }>
  recent_errors: Array<{ path: string; method: string; status: number; message: string; error_type: string; ts: number }>
  inference_count: number
  total_tokens: number
  tokens_per_sec: number
  avg_tokens_per_request: number
  health_score: { score: number; status: string }
  status_message: string
  model_metrics: Array<{ model: string; count: number; total_tokens: number; tokens_per_sec: number; avg_tokens: number }>
  model_events: Array<{ type: string; model: string; detail: string; ts: number }>
  health_history: Array<{ score: number; status: string; ts: number }>
  memory_history: Array<{ rss_mb: number; virtual_mb: number; system_percent: number; ts: number }>
  rate_violations: Array<{ path: string; count: number; limit: number; ts: number }>
  system: {
    cpu_percent: number
    memory_percent: number
    memory_available_mb: number
    open_files?: number
    threads?: number
    gc_gen0?: number
    gc_gen1?: number
    gc_gen2?: number
    process_cpu_percent?: number
    process_memory_percent?: number
    rss_mb?: number
  }
  gpu?: GPUInfo
  model_loaded: boolean
  model_loading?: boolean
  model_type: string | null
  device?: string | null
  num_parameters?: number | null
  soul: string | null
  inference: {
    is_inferencing?: boolean
    inference_count?: number
    total_generated?: number
  }
  kv_sessions?: KvSessionsInfo
  quantization?: unknown
  training_pool?: { active_jobs: number; max_workers: number; total_tracked: number } | null
}

export interface OutputLine {
  text: string
  level: string
  source: string
  ts: number
  tag?: string
  context?: Record<string, unknown>
}

export interface OutputResponse {
  lines: OutputLine[]
  size: number
  seq: number
}

export interface ExecutorJob {
  job_id: string
  tree_id: string | null
  status: string
  submitted_at: number
  started_at: number | null
  completed_at: number | null
  elapsed_s: number
  error: string | null
  cancel_requested: boolean
  result_keys?: string[]
  result_size_bytes?: number
}

export interface ExecutorStatus {
  initialized: boolean
  active_jobs: number
  max_workers: number
  total_tracked: number
  jobs: ExecutorJob[]
}

export interface InferencePoolStatus {
  initialized: boolean
  max_workers?: number
  queue_timeout?: number
  error?: string
}

export interface ProcessGuardStatus {
  enabled: boolean
  active: boolean
  model_id: string | null
  health: { alive: boolean; memory_mb?: number; restarts?: number } | null
}

export const systemController = {
  async getMetrics(): Promise<SystemMetrics> {
    return apiGet<SystemMetrics>('/system/metrics', undefined, { silent: true })
  },

  async getInfo(): Promise<SystemInfo> {
    return apiGet<SystemInfo>('/system/info', undefined, { silent: true })
  },

  async getDisk(): Promise<DiskUsage> {
    return apiGet<DiskUsage>('/system/disk', undefined, { silent: true })
  },

  async getDetailedHealth(): Promise<DetailedHealth> {
    return apiGet<DetailedHealth>('/health/detailed', undefined, { silent: true })
  },

  async getOutput(n: number = 100): Promise<OutputResponse> {
    return apiGet<OutputResponse>(`/system/output?n=${n}`, undefined, { silent: true })
  },

  async *streamOutput(tail: number = 50): AsyncGenerator<OutputLine> {
    const res = await fetch(`${PUBLIC_API_URL}/system/stream?tail=${tail}`)
    if (!res.ok) throw new Error(`Stream failed: ${res.status}`)
    const reader = res.body?.getReader()
    if (!reader) throw new Error('Stream not available')
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const events = buffer.split('\n\n')
      buffer = events.pop()!
      for (const evt of events) {
        const line = evt.replace(/^data: /, '').trim()
        if (!line) continue
        try {
          yield JSON.parse(line)
        } catch { /* malformed SSE event — skip */ }
      }
    }
  },

  async getExecutorStatus(): Promise<ExecutorStatus> {
    return apiGet<ExecutorStatus>('/system/executor', undefined, { silent: true })
  },

  async cancelExecutorJob(jobId: string): Promise<{ cancelled: boolean }> {
    const { apiPost } = await import('./http-client')
    return apiPost<{ cancelled: boolean }>(`/system/executor/${jobId}/cancel`)
  },

  async purgeExecutorJobs(maxAgeS: number = 3600): Promise<{ purged: number }> {
    const { apiPost } = await import('./http-client')
    return apiPost<{ purged: number }>(`/system/executor/purge?max_age_s=${maxAgeS}`)
  },

  async getInferencePoolStatus(): Promise<InferencePoolStatus> {
    return apiGet<InferencePoolStatus>('/system/inference-pool', undefined, { silent: true })
  },

  async getProcessGuardStatus(): Promise<ProcessGuardStatus> {
    return apiGet<ProcessGuardStatus>('/models/process-guard', undefined, { silent: true })
  },

  async setProcessGuardEnabled(enabled: boolean): Promise<ProcessGuardStatus> {
    const { apiPost } = await import('./http-client')
    return apiPost<ProcessGuardStatus>('/models/process-guard', { enabled })
  },
}
