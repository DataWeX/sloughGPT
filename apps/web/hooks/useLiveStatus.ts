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
import type { LiveHealthSnapshot } from '@/lib/health-blocks'
import { blocksFromDetailed, blocksFromSse, snapshotFromBlocks } from '@/lib/health-blocks'
import { PUBLIC_API_URL } from '@/lib/config'
import { trackEvent } from '@/lib/dev-log'
import { logStateEvent } from '@/lib/state-events'

export type ConnectionStatus = 'connected' | 'connecting' | 'offline' | 'reloading' | 'error'

// Health blocks are owned by @/lib/health-blocks — re-exported here so existing
// consumers keep one import path while the transports share one normalizer.
export type {
  HookStatus,
  LiveHealthSnapshot,
  StartupStage,
  StagedLoaderStatus,
} from '@/lib/health-blocks'

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
const MAX_RELOADS = 3
const RELOAD_WINDOW_MS = 120_000 // 2 minutes

/**
 * Map the full /health/detailed response onto the live snapshot shape.
 * Used by the HTTP fallback poll so every field survives an SSE outage.
 *
 * Thin wrapper over the shared transport adapters: the payload shape is read in
 * `blocksFromDetailed` and the snapshot is written in `snapshotFromBlocks`, so
 * this function cannot drift from the SSE path.
 */
export function mapDetailedToSnapshot(d: DetailedHealth): LiveHealthSnapshot {
  return snapshotFromBlocks(blocksFromDetailed(d))
}

export const liveStatusStore = createStore<LiveStatusState>((set) => ({
  connectionStatus: 'connecting',
  health: null,
  healthLegacy: null,
  lastUpdate: null,
  failureCount: 0,
  lastError: null,
  ready: false,

  setConnectionStatus: (connectionStatus) =>
    set((s) => {
      if (s.connectionStatus !== connectionStatus) {
        trackEvent('connection_status_changed', { from: s.connectionStatus, to: connectionStatus })
        logStateEvent('connection_status_changed', {
          kind: 'connection',
          from: s.connectionStatus,
          to: connectionStatus,
        })
      }
      return { connectionStatus }
    }),
  setHealth: (health) =>
    set((s) => {
      const prev = s.health
      const prevStage = prev?.startup_stage ?? 'unknown'
      const nextStage = health.startup_stage ?? 'unknown'
      if (prevStage !== nextStage) {
        trackEvent('startup_stage_changed', { from: prevStage, to: nextStage })
        logStateEvent('startup_stage_changed', {
          kind: 'startup',
          from: prevStage,
          to: nextStage,
          data: {
            model_loaded: health.model_loaded,
            progress: health.startup_model_progress,
          },
        })
      }
      if ((prev?.model_loaded ?? false) !== health.model_loaded) {
        logStateEvent('health_model_loaded_changed', {
          kind: 'health',
          from: String(prev?.model_loaded ?? false),
          to: String(health.model_loaded),
          data: { model_type: health.model_type ?? undefined },
        })
      }
      return {
        health,
        lastUpdate: Date.now(),
        failureCount: 0,
        lastError: null,
        ready: s.ready || true,
      }
    }),
  setHealthLegacy: (healthLegacy) => set({ healthLegacy }),
  setFailureCount: (failureCount) => set({ failureCount }),
  incrementFailures: () => set((s) => ({ failureCount: s.failureCount + 1 })),
  setReady: (ready) => set({ ready }),
  reset: () =>
    set({
      connectionStatus: 'connecting',
      health: null,
      healthLegacy: null,
      lastUpdate: null,
      failureCount: 0,
      lastError: null,
      ready: false,
    }),
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
          const hadFailures = liveStatusStore.getState().failureCount > 0
          liveStatusStore.getState().setHealthLegacy({
            status: 'healthy',
            model_loaded: h.model_loaded,
            model_type: h.model_type || '',
            summary: '',
            inference_count: h.inference_count,
            is_inferencing: h.inference?.is_inferencing,
          })
          liveStatusStore.getState().setConnectionStatus('connected')
          // Convert full detailed health to the live snapshot shape
          const snap = mapDetailedToSnapshot(h)
          liveStatusStore.getState().setHealth(snap)
          if (hadFailures) {
            logStateEvent('health_fallback_recovered', {
              kind: 'health',
              message: 'health_fallback_recovered connected',
            })
          }
        } else {
          const first = liveStatusStore.getState().failureCount === 0
          liveStatusStore.getState().setHealthLegacy('offline')
          liveStatusStore.getState().incrementFailures()
          if (first) {
            logStateEvent('health_fallback_empty', {
              kind: 'health',
              message: 'health_fallback_empty offline',
            })
          }
          checkReload()
        }
      } catch {
        const first = liveStatusStore.getState().failureCount === 0
        liveStatusStore.getState().setHealthLegacy('offline')
        liveStatusStore.getState().incrementFailures()
        if (first) {
          logStateEvent('health_fallback_error', {
            kind: 'health',
            message: 'health_fallback_error offline',
          })
        }
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
      // Reload-loop protection: track reloads in sessionStorage.
      // If we've reloaded MAX_RELOADS times within RELOAD_WINDOW_MS,
      // stop reloading and show an error instead of creating an infinite loop.
      const now = Date.now()
      const storageKey = 'slo-reload-count'
      const storageTimeKey = 'slo-reload-window-start'
      let reloadCount = parseInt(sessionStorage.getItem(storageKey) || '0', 10)
      let windowStart = parseInt(sessionStorage.getItem(storageTimeKey) || '0', 10)

      if (!windowStart || now - windowStart > RELOAD_WINDOW_MS) {
        // New window — reset counter
        reloadCount = 0
        windowStart = now
      }

      if (reloadCount >= MAX_RELOADS) {
        // Too many reloads — show error state, don't reload again
        liveStatusStore.getState().setConnectionStatus('error')
        return
      }

      reloadCount++
      sessionStorage.setItem(storageKey, String(reloadCount))
      sessionStorage.setItem(storageTimeKey, String(windowStart))

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
    // Same contract as the HTTP fallback poll: lift the payload into blocks,
    // then flatten once. @/lib/health-blocks owns the field list for both
    // transports, so SSE can no longer drop a field HTTP sends (bandwidth)
    // while HTTP drops one SSE sends (diagnoses).
    const snap = snapshotFromBlocks(blocksFromSse(envelope.data))
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
    logStateEvent('sse_error', {
      kind: 'sse',
      message: `sse_error ${_err?.message ?? 'stream failed'}`,
      data: { message: _err?.message ?? 'stream failed' },
    })
    checkReload()
  }

  function onOpen() {
    liveStatusStore.getState().reset()
    liveStatusStore.getState().setConnectionStatus('connected')
    logStateEvent('sse_open', { kind: 'sse', message: 'sse_open connected' })
  }

  function onClose() {
    if (_stopped) return
    // SSE disconnected — start fallback poll
    liveStatusStore.getState().setConnectionStatus('connecting')
    logStateEvent('sse_close', { kind: 'sse', message: 'sse_close → fallback poll' })
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
    /** Current startup stage (init/critical/ready/background) */
    startupStage: health?.startup_stage ?? 'unknown',
    /** Elapsed seconds since startup began */
    startupElapsed: health?.startup_elapsed ?? 0,
    /** Model load progress (0.0 to 1.0) */
    startupModelProgress: health?.startup_model_progress ?? 0,
    /** Model load progress message */
    startupModelProgressMessage: health?.startup_model_progress_message ?? '',
    /** Startup hook statuses */
    startupHooks: health?.startup_hooks ?? {},
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
