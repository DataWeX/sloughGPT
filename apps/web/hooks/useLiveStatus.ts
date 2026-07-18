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

import { useEffect, useRef, useState } from 'react'
import { create } from 'zustand'
import { createSSEStream, type SSEEnvelope } from '@/lib/sse-client'
import { modelController, type HealthStatus } from '@/lib/model-controller'
import { PUBLIC_API_URL } from '@/lib/config'

export type ConnectionStatus = 'connected' | 'connecting' | 'offline' | 'reloading'

export interface LiveHealthSnapshot {
  model_loaded: boolean
  model_loading: boolean
  model_type: string | null
  soul: string | null
  is_inferencing: boolean
  inference_count: number
  uptime_seconds: number
  request_count: number
  error_count: number
  tokens_per_sec: number
  avg_latency_ms: number
  cpu_percent: number | null
  memory_percent: number | null
  health_score: number
  health_status: string
  health_summary: string
  diagnoses: Array<{ check: string; severity: string; score: number; message: string }>
  num_parameters: number | null
  quantization: unknown | null
  training_pool: unknown | null
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

  // Actions
  setConnectionStatus: (s: ConnectionStatus) => void
  setHealth: (h: LiveHealthSnapshot) => void
  setHealthLegacy: (h: HealthStatus | 'offline' | null) => void
  setFailureCount: (n: number) => void
  incrementFailures: () => void
  reset: () => void
}

const FALLBACK_POLL_MS = 8000
const MAX_FAILURES_BEFORE_RELOAD = 6
const RELOAD_DELAY_MS = 2000

export const liveStatusStore = create<LiveStatusState>((set) => ({
  connectionStatus: 'connecting',
  health: null,
  healthLegacy: null,
  lastUpdate: null,
  failureCount: 0,
  lastError: null,

  setConnectionStatus: (connectionStatus) => set({ connectionStatus }),
  setHealth: (health) => set({ health, lastUpdate: Date.now(), failureCount: 0, lastError: null }),
  setHealthLegacy: (healthLegacy) => set({ healthLegacy }),
  setFailureCount: (failureCount) => set({ failureCount }),
  incrementFailures: () => set((s) => ({ failureCount: s.failureCount + 1 })),
  reset: () => set({ connectionStatus: 'connecting', health: null, healthLegacy: null, lastUpdate: null, failureCount: 0, lastError: null }),
}))

/**
 * Subscribe to the live health SSE stream.
 * Call once at app root. The store updates automatically.
 */
export function initLiveStatus(): () => void {
  const store = liveStatusStore.getState()
  let fallbackTimer: ReturnType<typeof setInterval> | null = null
  let _stopped = false

  function startFallbackPoll() {
    if (fallbackTimer || _stopped) return
    const poll = async () => {
      if (_stopped) return
      try {
        const h = await modelController.getHealth()
        if (h && h !== null) {
          liveStatusStore.getState().setHealthLegacy(h)
          liveStatusStore.getState().setConnectionStatus('connected')
          // Convert to LiveHealthSnapshot shape
          const snap: Partial<LiveHealthSnapshot> = {
            model_loaded: h.model_loaded,
            model_loading: h.model_loading ?? false,
            model_type: h.model_type,
            soul: h.soul_name ?? null,
            inference_count: h.inference_count ?? 0,
          }
          liveStatusStore.getState().setHealth(snap as LiveHealthSnapshot)
        } else {
          liveStatusStore.getState().setHealthLegacy('offline')
          liveStatusStore.getState().incrementFailures()
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
    const d = envelope.data as Partial<LiveHealthSnapshot>
    const snap: LiveHealthSnapshot = {
      model_loaded: Boolean(d.model_loaded),
      model_loading: Boolean(d.model_loading),
      model_type: d.model_type ?? null,
      soul: d.soul ?? null,
      is_inferencing: Boolean(d.is_inferencing),
      inference_count: Number(d.inference_count) || 0,
      uptime_seconds: Number(d.uptime_seconds) || 0,
      request_count: Number(d.request_count) || 0,
      error_count: Number(d.error_count) || 0,
      tokens_per_sec: Number(d.tokens_per_sec) || 0,
      avg_latency_ms: Number(d.avg_latency_ms) || 0,
      cpu_percent: d.cpu_percent != null ? Number(d.cpu_percent) : null,
      memory_percent: d.memory_percent != null ? Number(d.memory_percent) : null,
      health_score: Number(d.health_score) || 0,
      health_status: String(d.health_status || 'unknown'),
      health_summary: String(d.health_summary || ''),
      diagnoses: Array.isArray(d.diagnoses) ? d.diagnoses as Array<{ check: string; severity: string; score: number; message: string }> : [],
      num_parameters: d.num_parameters != null ? Number(d.num_parameters) : null,
      quantization: d.quantization ?? null,
      training_pool: d.training_pool ?? null,
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
    stopFallbackPoll() // SSE working, no need for fallback
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

  // Also start a fallback poll in case SSE endpoint doesn't exist yet
  startFallbackPoll()

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
  }
}

// Re-export the store selector for non-hook usage
function useLiveStatusStore<T>(selector: (s: LiveStatusState) => T): T {
  const [value, setValue] = useState(() => selector(liveStatusStore.getState()))
  useEffect(() => {
    return liveStatusStore.subscribe((state) => {
      const next = selector(state)
      setValue(next)
    })
  }, [selector])
  return value
}

export { useLiveStatusStore }
