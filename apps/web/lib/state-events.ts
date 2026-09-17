'use client'

/**
 * State-event logging — single entry point for connection/startup/sse/
 * health/overlay transitions in the web UI.
 *
 * Writes to three sinks (all inside the error-handler + logging logic):
 *   1. error-store `stateEvents` ring buffer — visible in
 *      ErrorDiagnosticsPanel ("State events") and DebugOverlay.
 *   2. `trackEvent` — batched POST to /errors/logs/ingest → backend
 *      OutputBuffer → /system/stream + monitoring OutputCard.
 *   3. `logger` (dev console) — immediate console output during development.
 *
 * Callers: useLiveStatus (connection/startup/sse/health), api-monitor-store
 * (api connection/failures), StartupOverlay (overlay visibility), ErrorLifecycle
 * (lifecycle init). The store dedups consecutive identical transitions within
 * 5s so 3s health polls and 8s fallback polls can't flood the buffer.
 */

import { useErrorStore, type StateEventKind } from './error-store'
import { logger, trackEvent } from './dev-log'

export type { StateEventKind }
export type { StateEvent } from './error-store'

export interface LogStateEventOpts {
  kind?: StateEventKind
  from?: string
  to?: string
  message?: string
  data?: Record<string, unknown>
}

/** Log a state transition. Never throws — logging must not break the UI. */
export function logStateEvent(event: string, opts: LogStateEventOpts = {}): string {
  try {
    const id = useErrorStore.getState().logStateEvent(event, opts)
    try {
      trackEvent(event, {
        ...(opts.from !== undefined ? { from: opts.from } : {}),
        ...(opts.to !== undefined ? { to: opts.to } : {}),
        ...(opts.data ?? {}),
      })
    } catch {
      /* backend ingest is best-effort */
    }
    if (process.env.NODE_ENV === 'development') {
      try {
        logger.debug(`[state] ${event}`, {
          ...(opts.from !== undefined ? { from: opts.from } : {}),
          ...(opts.to !== undefined ? { to: opts.to } : {}),
          ...(opts.data ?? {}),
        })
      } catch {
        /* console must never throw */
      }
    }
    return id
  } catch {
    return ''
  }
}
