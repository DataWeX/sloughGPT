'use client'
import { create } from 'zustand'

export type ApiStatus = 'connected' | 'connecting' | 'offline' | 'reloading'

export interface ConnectionDiagnostic {
  /** The endpoint that failed */
  endpoint: string
  /** Error message */
  error: string
  /** HTTP status code (0 = connection refused / timeout) */
  status: number
  /** Timeout in ms that was used */
  timeoutMs: number
  /** Timestamp of the failure */
  timestamp: number
  /** Whether this was a timeout vs connection refused */
  kind: 'timeout' | 'connection_refused' | 'http_error' | 'unknown'
}

export interface HealthSummaryData {
  score: number
  status: string
  summary: string
  model_loaded: boolean
  model_loading: boolean
  model_type: string | null
  soul: string | null
  uptime_seconds: number
  request_count: number
  error_count: number
  tokens_per_sec: number
  cpu_percent: number | null
  memory_percent: number | null
}

interface ApiMonitorState {
  status: ApiStatus
  lastOnline: number | null
  lastOffline: number | null
  healthSummary: HealthSummaryData | null
  /** Last N connection failures for diagnostic display */
  recentFailures: ConnectionDiagnostic[]
  /** Number of consecutive failures */
  failureCount: number
  /** When the server was last seen online */
  lastSuccessEndpoint: string | null
  setStatus: (status: ApiStatus) => void
  setHealthSummary: (data: HealthSummaryData | null) => void
  addFailure: (diag: ConnectionDiagnostic) => void
  clearFailures: () => void
}

const MAX_RECENT_FAILURES = 10

export const useApiMonitor = create<ApiMonitorState>((set) => ({
  status: 'connecting',
  lastOnline: null,
  lastOffline: null,
  healthSummary: null,
  recentFailures: [],
  failureCount: 0,
  lastSuccessEndpoint: null,
  setStatus: (status) =>
    set((s) => ({
      status,
      lastOnline: status === 'connected' ? Date.now() : s.lastOnline,
      lastOffline: status !== 'connected' ? Date.now() : s.lastOffline,
    })),
  setHealthSummary: (data) => set({ healthSummary: data }),
  addFailure: (diag) =>
    set((s) => ({
      recentFailures: [diag, ...s.recentFailures].slice(0, MAX_RECENT_FAILURES),
      failureCount: s.failureCount + 1,
    })),
  clearFailures: () => set({ recentFailures: [], failureCount: 0 }),
}))

/**
 * Sync liveStatusStore → apiMonitorStore.
 * This keeps useBackendWatcher consumers working while the SSE stream
 * provides the actual data.
 */
if (typeof window !== 'undefined') {
  import('@/hooks/useLiveStatus').then(({ liveStatusStore }) => {
    liveStatusStore.subscribe((live) => {
      const monitor = useApiMonitor.getState()
      // Sync connection status
      if (live.connectionStatus !== monitor.status) {
        monitor.setStatus(live.connectionStatus)
      }
      // Sync health summary
      if (live.health) {
        monitor.setHealthSummary({
          score: live.health.health_score,
          status: live.health.health_status,
          summary: live.health.health_summary,
          model_loaded: live.health.model_loaded,
          model_loading: live.health.model_loading,
          model_type: live.health.model_type,
          soul: live.health.soul,
          uptime_seconds: live.health.uptime_seconds,
          request_count: live.health.request_count,
          error_count: live.health.error_count,
          tokens_per_sec: live.health.tokens_per_sec,
          cpu_percent: live.health.cpu_percent,
          memory_percent: live.health.memory_percent,
        })
      }
      // Sync failure count
      if (live.failureCount > 0) {
        useApiMonitor.setState({ failureCount: live.failureCount })
      }
    })
  })
}
