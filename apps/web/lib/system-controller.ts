/**
 * System Controller — system metrics, info, disk, detailed health, and output stream.
 *
 * Usage:
 *   import { systemController } from '@/lib/system-controller'
 *   const metrics = await systemController.getMetrics()
 *   for await (const line of systemController.streamOutput()) { ... }
 */

import { apiGet, apiPost, apiPut, streamSSE } from './http-client'

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

export interface BatteryStatus {
  level: number
  is_charging: boolean
  is_plugged: boolean
  health: string
  capacity: number
  voltage_mv: number
  current_ma: number
  time_to_full_min: number | null
  time_to_empty_min: number | null
  source: 'sysfs' | 'simulated'
  name: string
  level_band: 'low' | 'ok' | 'high' | 'full'
  updated_at: number
  cycle_count: number
  energy_full: number
  energy_full_design: number
  health_percent: number
}

export interface BatteryControl {
  supported: boolean
  writable: boolean
  path: string | null
  current_limit: number | null
  reason: string
  start_supported: boolean
  start_path: string | null
  current_floor: number | null
  incumbent: string | null
}

export interface BatteryAdvice {
  limit: number
  action: 'unplug' | 'cap_at_80' | 'plug_in' | 'maintain'
  reason: string
}

export interface BatteryPolicyValues {
  enabled: boolean
  floor: number
  ceiling: number
  mode: 'band' | 'ceiling'
  interval_seconds: number
  band: string
}

export interface BatteryPolicy extends BatteryPolicyValues {
  file: string
  error: string | null
  explain: string
}

export interface BatteryDaemonState {
  present: boolean
  active: boolean
  pid?: number | null
  age_seconds?: number | null
  owned?: boolean
  dry_run?: boolean
  last_action?: string | null
  last_value?: number | null
  last_reason?: string | null
  last_outcome?: string | null
  explain?: string | null
}

export interface BatteryInfo {
  status: BatteryStatus
  control: BatteryControl
  advice: BatteryAdvice
  policy: BatteryPolicy
  daemon: BatteryDaemonState
}

export interface BatteryLimitResult {
  applied: boolean
  supported: boolean
  limit: number | null
  reason: string
  path: string | null
  floor_limit: number | null
}

export interface BatteryPolicyInput {
  enabled?: boolean
  floor?: number
  ceiling?: number
  mode?: 'band' | 'ceiling'
}

export interface BatteryPolicyResult {
  ok: boolean
  error: string | null
  load_error?: string | null
  policy: BatteryPolicyValues
  file?: string
  explain?: string
}

export interface KvSessionsInfo {
  enabled?: boolean
  active_sessions?: number
  max_sessions?: number
  cached_tokens?: number
  ttl_seconds?: number
  oldest_session_age?: number
}

/** Cumulative edge byte counters, mirrored from the Rust gateway. */
export interface BandwidthStats {
  /** Uncompressed payload bytes the edge served (identity size). */
  identity_bytes: number
  /** Bytes actually emitted on the wire (after compression, or unchanged). */
  wire_bytes: number
  /** identity_bytes − wire_bytes; negative when compression expanded. */
  saved_bytes: number
  /** Share of identity bytes not sent (can be slightly negative). */
  saved_pct: number
  compressed_responses: number
  identity_responses: number
  zstd_responses: number
  gzip_responses: number
}

export interface DetailedHealth {
  status: string
  uptime_seconds: number
  timestamp: string
  request_count: number
  error_count: number
  avg_latency_ms: number
  p95_latency_ms: number
  requests_per_minute: number
  path_latencies: Array<{ path: string; avg_ms: number; count: number; p95_ms: number }>
  recent_errors: Array<{
    path: string
    method: string
    status: number
    message: string
    error_type: string
    ts: number
  }>
  inference_count: number
  total_tokens: number
  tokens_per_sec: number
  avg_tokens_per_request: number
  /**
   * Composite health score. `summary`/`diagnoses` come from the health flow and
   * are what the DiagnosticsCard renders; older payloads omit them, hence `?`.
   */
  health_score: {
    score: number
    status: string
    summary?: string
    diagnoses?: Array<{ check: string; severity: string; score: number; message: string }>
  }
  status_message: string
  model_metrics: Array<{
    model: string
    count: number
    total_tokens: number
    tokens_per_sec: number
    avg_tokens: number
  }>
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
  /** Edge bandwidth counters — null/absent without a reachable gateway. */
  bandwidth?: BandwidthStats | null
  lifecycle?: {
    phase: string
    profile?: string
    is_running: boolean
    is_draining?: boolean
    uptime?: number
    in_flight?: number
    error?: string
  }
  resource_allocation?: {
    mode?: string
    compute_threads?: number
    io_threads?: number
    omp_num_threads?: number
    mkl_num_threads?: number
    openblas_num_threads?: number
    numexpr_num_threads?: number
    inference_pool_size?: number
    train_pool_size?: number
    task_queue_workers?: number
    dataloader_workers?: number
    concurrent_reads?: number
    concurrent_writes?: number
    process_guard_concurrent?: number
  }
  process_guard?: {
    active?: boolean
    enabled?: boolean
    health?: { alive: boolean; memory_mb?: number; restarts?: number }
  } | null
  memory_pressure?: {
    current_mb?: number
    peak_mb?: number
    pressure_level?: string
    tracked_count?: number
  } | null
  registry?: { healthy: boolean; default_model?: string; models?: Array<Record<string, unknown>> }
  versions?: {
    app?: string
    api?: string
    package?: string
    torch?: string
    pydantic?: string
    features?: Record<string, { backend: string; api: string }>
  }
  mps_monitor?: { usage: number; locked_to_cpu: boolean } | null
  idle?: { enabled: boolean; idle_seconds?: number }
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

export interface ServicesHealth {
  status: 'healthy' | 'degraded'
  services: Record<string, { status: string; error?: string; [key: string]: unknown }>
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

  async getBattery(): Promise<BatteryInfo> {
    return apiGet<BatteryInfo>('/system/battery', undefined, { silent: true })
  },

  async setBatteryLimit(percent: number): Promise<BatteryLimitResult> {
    return apiPost<BatteryLimitResult>(`/system/battery/limit?percent=${percent}`, undefined, {
      silent: true,
    })
  },

  async setBatteryPolicy(input: BatteryPolicyInput = {}): Promise<BatteryPolicyResult> {
    const params = new URLSearchParams()
    if (input.enabled !== undefined) params.set('enabled', String(input.enabled))
    if (input.floor !== undefined) params.set('floor', String(input.floor))
    if (input.ceiling !== undefined) params.set('ceiling', String(input.ceiling))
    if (input.mode !== undefined) params.set('mode', input.mode)
    const qs = params.toString()
    return apiPut<BatteryPolicyResult>(`/system/battery/policy${qs ? `?${qs}` : ''}`, undefined, {
      silent: true,
    })
  },

  async getDetailedHealth(): Promise<DetailedHealth> {
    return apiGet<DetailedHealth>('/health/detailed', undefined, { silent: true })
  },

  async getOutput(n: number = 100): Promise<OutputResponse> {
    return apiGet<OutputResponse>(`/system/output?n=${n}`, undefined, { silent: true })
  },

  async *streamOutput(tail: number = 50, signal?: AbortSignal): AsyncGenerator<OutputLine> {
    try {
      for await (const event of streamSSE(`/system/stream?tail=${tail}`, {
        method: 'GET',
        signal,
      })) {
        const d = event.data
        if (
          d &&
          typeof d.text === 'string' &&
          typeof d.level === 'string' &&
          typeof d.source === 'string' &&
          typeof d.ts === 'number'
        ) {
          const line: OutputLine = { text: d.text, level: d.level, source: d.source, ts: d.ts }
          yield line
        }
      }
    } catch (err) {
      throw new Error(`Stream failed: ${err instanceof Error ? err.message : 'unknown'}`)
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

  async getServicesHealth(): Promise<ServicesHealth> {
    return apiGet<ServicesHealth>('/health/services', undefined, { silent: true })
  },
}
