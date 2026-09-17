'use client'

/**
 * State-event logging — single entry point for connection/startup/sse/
 * health/overlay transitions in the web UI.
 *
 * STRUCTURED ENVELOPE (across the board — every UI event shares it):
 *   { id, kind, event, message, from?, to?, timestamp, data? }
 *   - event:     snake_case verb phrase, e.g. `training_started`
 *   - kind:      one of the catalog below (inferred from prefix when omitted)
 *   - message:   human one-liner; transitions render `{event} {from} → {to}`
 *   - from/to:   only for transitions (connection, stage, mode changes)
 *   - data:      flat primitives only (string/number/boolean) — no nesting,
 *                so the record survives the backend ingest (`context`) as-is
 *   - timestamp: ms epoch
 *
 * KIND CATALOG (prefix → kind, implemented in `lib/event-kinds.ts`):
 *   connection: connection_*        startup: startup_*          sse: sse_*
 *   health:     health_*            overlay: overlay_*          api: api_*
 *   auth:       auth_*, workspace_* chat: session_*, stream_*, chat_*,
 *               conversion_*, regenerate_*, soul_*
 *   training:   training_*, checkpoint_*, dataset_*, knowledge_*
 *   model:      model_*             system: vm_*, shell_*, webhook_*,
 *               operation_*, download_*, error_lifecycle_*
 *   ui:         route_*, locale_*, theme_*, palette_*, mode_*, settings_*,
 *               feedback_*
 *
 * SINKS (all inside the error-handler + logging logic):
 *   1. error-store `stateEvents` ring buffer (200, 5s consecutive-dedup) —
 *      visible in ErrorDiagnosticsPanel ("State events") and DebugOverlay.
 *   2. `trackEvent` — batched POST to /errors/logs/ingest → backend
 *      OutputBuffer → /system/stream + monitoring OutputCard. Since
 *      `WebLogger.trackEvent` mirrors into the same buffer, EVERY UI event
 *      (even direct trackEvent callers) lands in the timeline with zero
 *      per-callsite changes.
 *   3. `logger` (dev console) — immediate console output during development.
 */

import { useErrorStore, type StateEventKind } from './error-store'
import { inferStateKind as inferKind } from './event-kinds'

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
    const kind = opts.kind ?? inferKind(event)
    const payload = {
      ...(opts.from !== undefined ? { from: opts.from } : {}),
      ...(opts.to !== undefined ? { to: opts.to } : {}),
      ...(opts.data ?? {}),
    }
    const id = useErrorStore.getState().logStateEvent(event, { ...opts, kind })
    // Fire-and-forget: backend ingest + dev console via lazy dev-log import.
    // Lazy (not static) so this module stays importable even when test files
    // partially mock `@/lib/dev-log` — missing exports are skipped via
    // optional chaining instead of failing module collection.
    void import('./dev-log').then(
      (m) => {
        try {
          m.trackEvent?.(event, { kind, ...payload })
        } catch {
          /* backend ingest is best-effort */
        }
        try {
          if (process.env.NODE_ENV === 'development') {
            m.logger?.debug?.(`[state] ${event}`, payload)
          }
        } catch {
          /* console must never throw */
        }
      },
      () => {
        /* dev-log unavailable — store buffer already written */
      },
    )
    return id
  } catch {
    return ''
  }
}
