/**
 * Live Status — single source of truth for server health.
 *
 * Replaces useApiHealth (28s poll) and useBackendWatcher (8s poll) with
 * a single SSE-backed store that pushes health every 3s. Falls back to
 * HTTP polling if the SSE stream fails.
 *
 * Usage:
 *   import { useLiveStatus } from '@/hooks/useLiveStatus'
 *   const { health, status, summary, connected } = useLiveStatus()
 *
 *   // Or access the store directly:
 *   import { liveStatusStore } from '@/hooks/useLiveStatus'
 *   const snap = liveStatusStore.getState()
 */

'use client'

import { createStore } from 'zustand/vanilla'
import { createSSEStream, type SSEEnvelope } from '@/lib/sse-client'
import type { HealthStatus } from '@/lib/model-controller'
import { systemController, type DetailedHealth } from '@/lib/system-controller'
import { PUBLIC_API_URL } from '@/lib/config'

export type ConnectionStatus = 'connected' | 'connecting' | 'offline' | 'reloading'

export interface LiveHealthSnapshot {
  model_loaded: boolean
  model_loading: boolean
  model_type: string | null
  device: string | null
  soul: string | null
  is_inferencing: boolean
  inference_count: number
  uptime_seconds: number
  request_count: number
  error_count: number
  tokens_per_sec: number
  avg_latency_ms: number
  p95_latency_ms: number
  requests_per_minute: number
  total_tokens: number
  avg_tokens_per_request: number
  cpu_percent: number | null
  memory_percent: number | null
  health_score: number
  health_status: string
  health_summary: string
  diagnoses: Array<{ check: string; severity: string; score: number; message: string }>
  num_parameters: number | null
  quantization: unknown | null
  training_pool: { active_jobs: number; max_workers: number; total_tracked: number } | null
  model_metrics: Array<{ model: string; count: number; total_tokens: number; tokens_per_sec: number; avg_tokens: number }>
  model_events: Array<{ type: string; model: string; detail: string; ts: number }>
  rate_violations: Array<{ path: string; count: number; limit: number; ts: number }>
  health_history: Array<{ score: number; status: string; ts: number }>
  memory_history: Array<{ rss_mb: number; virtual_mb: number; system_percent: number; ts: number }>
  path_latencies: Array<{ path: string; avg_ms: number; count: number; p95_ms: number }>
  recent_errors: Array<{ path: string; method: string; status: number; message: string; error_type: string; ts: number }>
}

export interface LiveStatusState {
  /** Connection status: connected (SSE active), connecting (trying), offline (failed), reloading (about to reload) */
  connectionStatus: ConnectionStatus
  /** Latest health snapshot from SSE or fallback poll */
  health: LiveHealthSnapshot | null
  /** Legacy HealthStatus shape for backward compat with useApiHealth consumers */
  healthLegacy: HealthStatus | 'offline' | null
  /** Timestamp of last successful health update */
  lastUpdate: number | null
  /** How many consecutive SSE/poll failures */
  failureCount: number
  /** Last connection error message */
  lastError: string | null
  /** True once health endpoint first responds — gates feature polling hooks */
  ready: boolean

  // Actions
  setConnectionStatus: (s: ConnectionStatus) => void
  setHealth: (h: LiveHealthSnapshot) => void
  setHealthLegacy: (h: HealthStatus | 'offline' | null) => void
  setFailureCount: (n: number) => void
  incrementFailures: () => void
  setReady: (r: boolean) => void
  reset: () => void
}

const FALLBACK_POLL_MS = 8000
const MAX_FAILURES_BEFORE_RELOAD = 6
const RELOAD_DELAY_MS = 2000

/**
 * Map the full /health/detailed response onto the live snapshot shape.
 * Used by the HTTP fallback poll so every field survives an SSE outage.
 */
export function mapDetailedToSnapshot(d: DetailedHealth): LiveHealthSnapshot {
  const healthScore = d.health_score ?? { score: 0, status: 'unknown' }
  return {
    model_loaded: Boolean(d.model_loaded),
    model_loading: Boolean(d.model_loading),
    model_type: d.model_type ?? null,
    device: d.device ?? null,
    soul: d.soul ?? null,
    is_inferencing: Boolean(d.inference?.is_inferencing),
    inference_count: d.inference_count ?? 0,
    uptime_seconds: Number(d.uptime_seconds) || 0,
    request_count: Number(d.request_count) || 0,
    error_count: Number(d.error_count) || 0,
    tokens_per_sec: Number(d.tokens_per_sec) || 0,
    avg_latency_ms: Number(d.avg_latency_ms) || 0,
    p95_latency_ms: Number(d.p95_latency_ms) || 0,
    requests_per_minute: Number(d.requests_per_minute) || 0,
    total_tokens: Number(d.total_tokens) || 0,
    avg_tokens_per_request: Number(d.avg_tokens_per_request) || 0,
    cpu_percent: d.system?.cpu_percent != null ? Number(d.system.cpu_percent) : null,
    memory_percent: d.system?.memory_percent != null ? Number(d.system.memory_percent) : null,
    health_score: Number(healthScore.score) || 0,
    health_status: String(healthScore.status || d.status || 'unknown'),
    health_summary: String(d.status_message || ''),
    diagnoses: [],
    num_parameters: d.num_parameters != null ? Number(d.num_parameters) : null,
    quantization: d.quantization ?? null,
    training_pool: d.training_pool ?? null,
    model_metrics: Array.isArray(d.model_metrics) ? d.model_metrics : [],
    model_events: Array.isArray(d.model_events) ? d.model_events : [],
    rate_violations: Array.isArray(d.rate_violations) ? d.rate_violations : [],
    health_history: Array.isArray(d.health_history) ? d.health_history : [],
    memory_history: Array.isArray(d.memory_history) ? d.memory_history : [],
    path_latencies: Array.isArray(d.path_latencies) ? d.path_latencies : [],
    recent_errors: Array.isArray(d.recent_errors) ? d.recent_errors : [],
  }
}

export const liveStatusStore = createStore<LiveStatusState>((set) => ({
  connectionStatus: 'connecting',
  health: null,
  healthLegacy: null,
  lastUpdate: null,
  failureCount: 0,
  lastError: null,
  ready: false,

  setConnectionStatus: (connectionStatus) => set({ connectionStatus }),
  setHealth: (health) => set((s) => ({ health, lastUpdate: Date.now(), failureCount: 0, lastError: null, ready: s.ready || true })),
  setHealthLegacy: (healthLegacy) => set({ healthLegacy }),
  setFailureCount: (failureCount) => set({ failureCount }),
  incrementFailures: () => set((s) => ({ failureCount: s.failureCount + 1 })),
  setReady: (ready) => set({ ready }),
  reset: () => set({ connectionStatus: 'connecting', health: null, healthLegacy: null, lastUpdate: null, failureCount: 0, lastError: null, ready: false }),
}))

/**
 * Subscribe to the live health SSE stream.
 * Call once at app root. The store updates automatically.
 */
export function initLiveStatus(): () => void {
  const store = liveStatusStore.getState()
  let fallbackTimer: ReturnType<typeof setInterval> | null = null
  let fallbackDelayTimer: ReturnType<typeof setTimeout> | null = null
  let _receivedHealthEvent = false
  let _stopped = false

  function startFallbackPoll() {
    if (fallbackTimer || _stopped) return
    const poll = async () => {
      if (_stopped) return
      try {
        const h = await systemController.getDetailedHealth()
        if (h && h !== null) {
          liveStatusStore.getState().setHealthLegacy({ status: 'healthy', model_loaded: h.model_loaded, model_type: h.model_type || '', summary: '', inference_count: h.inference_count, is_inferencing: h.inference?.is_inferencing })
          liveStatusStore.getState().setConnectionStatus('connected')
          // Convert full detailed health to the live snapshot shape
          const snap = mapDetailedToSnapshot(h)
          liveStatusStore.getState().setHealth(snap)
        } else {
          liveStatusStore.getState().setHealthLegacy('offline')
          liveStatusStore.getState().incrementFailures()
          checkReload()
        }
      } catch {
        liveStatusStore.getState().setHealthLegacy('offline')
        liveStatusStore.getState().incrementFailures()
        checkReload()
      }
    }
    poll()
    fallbackTimer = setInterval(poll, FALLBACK_POLL_MS)
  }

  function stopFallbackPoll() {
    if (fallbackTimer) {
      clearInterval(fallbackTimer)
      fallbackTimer = null
    }
    if (fallbackDelayTimer) {
      clearTimeout(fallbackDelayTimer)
      fallbackDelayTimer = null
    }
  }

  function checkReload() {
    const { failureCount } = liveStatusStore.getState()
    if (failureCount >= MAX_FAILURES_BEFORE_RELOAD) {
      liveStatusStore.getState().setConnectionStatus('reloading')
      setTimeout(() => {
        if (!_stopped) window.location.reload()
      }, RELOAD_DELAY_MS)
    }
  }

  // Convert SSE envelope to our store shape
  function onHealthEvent(envelope: SSEEnvelope) {
    if (envelope.stream !== 'health') return
    _receivedHealthEvent = true
    stopFallbackPoll()
    const d = envelope.data as Partial<LiveHealthSnapshot>
    const snap: LiveHealthSnapshot = {
      model_loaded: Boolean(d.model_loaded),
      model_loading: Boolean(d.model_loading),
      model_type: d.model_type ?? null,
      device: d.device ?? null,
      soul: d.soul ?? null,
      is_inferencing: Boolean(d.is_inferencing),
      inference_count: Number(d.inference_count) || 0,
      uptime_seconds: Number(d.uptime_seconds) || 0,
      request_count: Number(d.request_count) || 0,
      error_count: Number(d.error_count) || 0,
      tokens_per_sec: Number(d.tokens_per_sec) || 0,
      avg_latency_ms: Number(d.avg_latency_ms) || 0,
      p95_latency_ms: Number(d.p95_latency_ms) || 0,
      requests_per_minute: Number(d.requests_per_minute) || 0,
      total_tokens: Number(d.total_tokens) || 0,
      avg_tokens_per_request: Number(d.avg_tokens_per_request) || 0,
      cpu_percent: d.cpu_percent != null ? Number(d.cpu_percent) : null,
      memory_percent: d.memory_percent != null ? Number(d.memory_percent) : null,
      health_score: Number(d.health_score) || 0,
      health_status: String(d.health_status || 'unknown'),
      health_summary: String(d.health_summary || ''),
      diagnoses: Array.isArray(d.diagnoses) ? d.diagnoses as Array<{ check: string; severity: string; score: number; message: string }> : [],
      num_parameters: d.num_parameters != null ? Number(d.num_parameters) : null,
      quantization: d.quantization ?? null,
      training_pool: d.training_pool ?? null,
      model_metrics: Array.isArray(d.model_metrics)
        ? d.model_metrics as Array<{ model: string; count: number; total_tokens: number; tokens_per_sec: number; avg_tokens: number }>
        : [],
      model_events: Array.isArray(d.model_events)
        ? d.model_events as Array<{ type: string; model: string; detail: string; ts: number }>
        : [],
      rate_violations: Array.isArray(d.rate_violations)
        ? d.rate_violations as Array<{ path: string; count: number; limit: number; ts: number }>
        : [],
      health_history: Array.isArray(d.health_history)
        ? d.health_history as Array<{ score: number; status: string; ts: number }>
        : [],
      memory_history: Array.isArray(d.memory_history)
        ? d.memory_history as Array<{ rss_mb: number; virtual_mb: number; system_percent: number; ts: number }>
        : [],
      path_latencies: Array.isArray(d.path_latencies)
        ? d.path_latencies as Array<{ path: string; avg_ms: number; count: number; p95_ms: number }>
        : [],
      recent_errors: Array.isArray(d.recent_errors)
        ? d.recent_errors as Array<{ path: string; method: string; status: number; message: string; error_type: string; ts: number }>
        : [],
    }
    liveStatusStore.getState().setHealth(snap)

    // Also update legacy shape for backward compat
    const legacy: HealthStatus = {
      status: snap.health_status,
      model_loaded: snap.model_loaded,
      model_type: snap.model_type || '',
      summary: snap.health_summary,
      inference_count: snap.inference_count,
      is_inferencing: snap.is_inferencing,
    }
    liveStatusStore.getState().setHealthLegacy(legacy)
    liveStatusStore.getState().setConnectionStatus('connected')
  }

  function onError(_err: Error) {
    liveStatusStore.getState().incrementFailures()
    liveStatusStore.getState().setConnectionStatus('connecting')
    checkReload()
  }

  function onOpen() {
    liveStatusStore.getState().reset()
    liveStatusStore.getState().setConnectionStatus('connected')
  }

  function onClose() {
    if (_stopped) return
    // SSE disconnected — start fallback poll
    liveStatusStore.getState().setConnectionStatus('connecting')
    stopFallbackPoll()
    startFallbackPoll()
  }

  // Try SSE first, fallback to poll immediately if it fails
  const stream = createSSEStream({
    url: '/health/stream',
    onEvent: onHealthEvent,
    onOpen,
    onClose,
    onError,
    reconnect: true,
    maxReconnects: Infinity,
    baseReconnectMs: 3000,
    maxReconnectMs: 15_000,
  })

  stream.start()

  // Start the fallback poll only if the SSE stream hasn't delivered a health
  // snapshot within one poll interval (grace period) — avoids a duplicate
  // /health/detailed request on every mount while the SSE stream is healthy.
  fallbackDelayTimer = setTimeout(() => {
    fallbackDelayTimer = null
    if (!_receivedHealthEvent) startFallbackPoll()
  }, FALLBACK_POLL_MS)

  return () => {
    _stopped = true
    stream.stop()
    stopFallbackPoll()
  }
}

/**
 * React hook — returns live server status.
 * Subscribes to the Zustand store and re-renders on changes.
 */
export function useLiveStatus() {
  const connectionStatus = useLiveStatusStore((s) => s.connectionStatus)
  const health = useLiveStatusStore((s) => s.health)
  const healthLegacy = useLiveStatusStore((s) => s.healthLegacy)
  const lastUpdate = useLiveStatusStore((s) => s.lastUpdate)
  const failureCount = useLiveStatusStore((s) => s.failureCount)
  const ready = useLiveStatusStore((s) => s.ready)

  return {
    /** SSE connection status */
    connectionStatus,
    /** Full live health snapshot (from SSE) */
    health,
    /** Legacy HealthStatus shape for backward compat */
    healthLegacy,
    /** When the last health update arrived */
    lastUpdate,
    /** Consecutive failures */
    failureCount,
    /** Whether the server is reachable */
    connected: connectionStatus === 'connected',
    /** Whether SSE is actively streaming */
    live: connectionStatus === 'connected' && health !== null,
    /** True once health endpoint first responds — gates feature polling hooks */
    ready,
  }
}

/**
 * Convenience hook — returns true once the server's health endpoint first responds.
 * Use this to gate feature polling hooks that would otherwise 404 during startup.
 */
export function useApiReady(): boolean {
  return useLiveStatusStore((s) => s.ready)
}

// Re-export the store selector for non-hook usage
import { useStore } from 'zustand'

function useLiveStatusStore<T>(selector: (s: LiveStatusState) => T): T {
  return useStore(liveStatusStore, selector)
}

export { useLiveStatusStore }
