'use client'

import { createStore } from 'zustand/vanilla'
import { useStore } from 'zustand'
import { extractErrorMessage } from '@/lib/error-utils'
import { inferStateKind, type StateEventKind } from '@/lib/event-kinds'
import { useBannerStore } from '@/lib/banner-store'
import { useToastStore } from '@/lib/toast-store'

export type { StateEventKind }
export { inferStateKind, isStateEventKind } from '@/lib/event-kinds'

export type ErrorSeverity = 'error' | 'warning' | 'info'

/** Where the error came from — drives fan-out policy (banner vs silent). */
export type ErrorSurface = 'global' | 'page' | 'section' | 'component' | 'api' | 'chat'

export interface AppError {
  id: string
  title: string
  message: string
  severity: ErrorSeverity
  source?: string
  timestamp: number
  dismissible?: boolean
  requestId?: string
  surface?: ErrorSurface
  journeyId?: string
  /** How many times this exact error occurred (deduped within 30s window) */
  count: number
}

export interface ActivityEntry {
  id: string
  message: string
  severity: ErrorSeverity
  source?: string
  timestamp: number
}

export interface StateEvent {
  id: string
  kind: StateEventKind
  event: string
  message: string
  from?: string
  to?: string
  timestamp: number
  data?: Record<string, unknown>
}

const DEDUP_WINDOW_MS = 30_000
const MAX_ERRORS = 20
const MAX_ACTIVITY = 50
const MAX_STATE_EVENTS = 200
const STATE_EVENT_DEDUP_MS = 5_000

function fingerprint(message: string): string {
  // Normalize: lowercase, replace numbers with N, replace long hex strings with ID
  let norm = message.toLowerCase()
  norm = norm.replace(/\d+/g, 'N')
  norm = norm.replace(/[a-f0-9]{8,}/g, 'ID')
  // Simple hash (djb2) — no crypto dependency needed
  let hash = 5381
  for (let i = 0; i < norm.length; i++) {
    hash = ((hash << 5) + hash + norm.charCodeAt(i)) | 0
  }
  return (hash >>> 0).toString(36).slice(0, 12)
}

/**
 * `extractErrorMessage` is contract-bound to return a string — but the error
 * funnel must be TOTAL: it runs from unhandled-rejection paths (e.g. the
 * dynamic import in apps/lib/request.ts fires after a test/teardown, where a
 * partial test mock of error-utils can leak in), and a handler that throws
 * turns one handled failure into an unhandled crash that takes the whole
 * vitest run down. Guard once, here, so no fan-out step can re-introduce it.
 */
function safeMessage(err: unknown): string {
  return extractErrorMessage(err) ?? ''
}

function extractErrorTitle(err: unknown): string {
  if (err && typeof err === 'object') {
    const e = err as Record<string, unknown>
    if (typeof e.title === 'string') return e.title
    const msg = safeMessage(e)
    if (msg.includes('404') || msg.includes('Not Found')) return 'Not Found'
    if (msg.includes('401') || msg.includes('Unauthorized')) return 'Unauthorized'
    if (msg.includes('403') || msg.includes('Forbidden')) return 'Forbidden'
    if (msg.includes('500')) return 'Server Error'
    if (msg.includes('timeout') || msg.includes('Timeout')) return 'Timeout'
    if (msg.includes('network') || msg.includes('Network') || msg.includes('fetch'))
      return 'Network Error'
    if (msg.includes('CORS') || msg.includes('cors')) return 'CORS Error'
    if (msg.includes('ECONNREFUSED')) return 'Connection Refused'
  }
  if (err instanceof Error) {
    const n = err.name
    if (n === 'TypeError') return 'Type Error'
    if (n === 'ReferenceError') return 'Reference Error'
    if (n === 'SyntaxError') return 'Syntax Error'
    return n
  }
  return 'Error'
}

function getSeverity(err: unknown, explicitSev?: ErrorSeverity): ErrorSeverity {
  if (explicitSev) return explicitSev
  const msg = safeMessage(err).toLowerCase()
  if (msg.includes('not found') || msg.includes('404')) return 'warning'
  if (msg.includes('unauthorized') || msg.includes('401')) return 'warning'
  if (msg.includes('forbidden') || msg.includes('403')) return 'warning'
  if (msg.includes('timeout')) return 'warning'
  if (msg.includes('network') || msg.includes('cors') || msg.includes('connection'))
    return 'warning'
  return 'error'
}

/**
 * Single funnel fan-out. Runs once per NEW error (deduped repeats skip it):
 *   1. structured state event (`error_reported`) — breadcrumbs/diagnostics
 *   2. global banner — journey blockers only (page/global surfaces)
 *   3. optional toast — transient, only when the producer asks
 *   4. backend batch — dynamic import keeps `db`/indexedDB out of the static
 *      graph (error-store is imported by node-env tests and lib code).
 */
function fanOut(error: AppError, fp: string, opts: { banner?: boolean; toast?: boolean }): void {
  errorStore.getState().logStateEvent('error_reported', {
    kind: 'system',
    message: error.title,
    data: {
      surface: error.surface,
      journeyId: error.journeyId,
      severity: error.severity,
      source: error.source,
      requestId: error.requestId,
    },
  })

  const wantsBanner =
    opts.banner ??
    (error.severity === 'error' && (error.surface === 'page' || error.surface === 'global'))
  if (wantsBanner) {
    useBannerStore.getState().showBanner({
      tone: 'destructive',
      title: error.title,
      message: error.source ? `via ${error.source}` : undefined,
      key: `err_${fp}`,
    })
  }

  if (opts.toast) {
    useToastStore.getState().addToast(error.title, 'error')
  }

  if (typeof window !== 'undefined') {
    import('@/lib/error-reporter')
      .then((m) =>
        m.reportError(error.message, error.source ?? 'web', {
          metadata: {
            surface: error.surface,
            journeyId: error.journeyId,
            severity: error.severity,
            requestId: error.requestId,
          },
        }),
      )
      .catch(() => {
        /* reporter is best-effort — never break the caller */
      })
  }
}

interface ErrorStore {
  errors: AppError[]
  recentActivity: ActivityEntry[]
  /** Total error count (including deduped) */
  totalErrorCount: number
  /** Ring buffer of connection/startup/sse/health/overlay state transitions (breadcrumbs) */
  stateEvents: StateEvent[]
  addError: (
    err: unknown,
    opts?: {
      source?: string
      title?: string
      severity?: ErrorSeverity
      dismissible?: boolean
      requestId?: string
      surface?: ErrorSurface
      journeyId?: string
      /** Force the global banner on/off; default: on for page/global errors. */
      banner?: boolean
      /** Also raise a transient toast (defaults to off — banner covers blockers). */
      toast?: boolean
    },
  ) => string
  dismissError: (id: string) => void
  clearErrors: () => void
  getErrors: () => AppError[]
  hasErrors: () => boolean
  logStateEvent: (
    event: string,
    opts?: {
      kind?: StateEventKind
      from?: string
      to?: string
      message?: string
      data?: Record<string, unknown>
    },
  ) => string
  clearStateEvents: () => void
  getStateEvents: () => StateEvent[]
}

const errorStore = createStore<ErrorStore>((set, get) => ({
  errors: [],
  recentActivity: [],
  totalErrorCount: 0,
  stateEvents: [],

  addError: (err, opts = {}) => {
    const { source, severity: sev, dismissible = true, requestId, surface, journeyId } = opts
    const title = opts.title || extractErrorTitle(err)
    const message = safeMessage(err)
    const severity = getSeverity(err, sev)
    const now = Date.now()

    // Deduplication: same fingerprint within 30s → increment count
    const fp = fingerprint(message)
    const { errors } = get()
    const existing = errors.find(
      (e) =>
        e.source === source && fingerprint(e.message) === fp && now - e.timestamp < DEDUP_WINDOW_MS,
    )

    if (existing) {
      const updated = { ...existing, count: existing.count + 1, timestamp: now }
      set((prev) => ({
        errors: [updated, ...prev.errors.filter((e) => e.id !== existing.id)],
      }))
      return existing.id
    }

    // New error
    const id = `err_${now}_${Math.random().toString(36).slice(2, 6)}`
    const error: AppError = {
      id,
      title,
      message,
      severity,
      source,
      timestamp: now,
      dismissible,
      requestId,
      surface,
      journeyId,
      count: 1,
    }

    // Activity entry (compact ticker feed)
    const activity: ActivityEntry = {
      id: `act_${now}_${Math.random().toString(36).slice(2, 6)}`,
      message: title !== 'Error' ? title : message.slice(0, 60),
      severity,
      source,
      timestamp: now,
    }

    set((prev) => ({
      errors: [error, ...prev.errors].slice(0, MAX_ERRORS),
      recentActivity: [activity, ...prev.recentActivity].slice(0, MAX_ACTIVITY),
      totalErrorCount: prev.totalErrorCount + 1,
    }))

    fanOut(error, fp, opts)
    return id
  },

  dismissError: (id) => {
    set((prev) => ({ errors: prev.errors.filter((e) => e.id !== id) }))
  },

  clearErrors: () => {
    set({ errors: [], recentActivity: [], totalErrorCount: 0 })
  },

  getErrors: () => get().errors,

  hasErrors: () => get().errors.length > 0,

  logStateEvent: (event, opts = {}) => {
    const now = Date.now()
    const kind = opts.kind ?? inferStateKind(event)
    const from = opts.from
    const to = opts.to
    const message =
      opts.message ??
      (from !== undefined || to !== undefined ? `${event} ${from ?? '?'} → ${to ?? '?'}` : event)

    // Dedup: skip consecutive identical transitions within a short window
    // (health polls every 3s / fallback every 8s must not flood the buffer).
    const [latest] = get().stateEvents
    if (
      latest &&
      latest.event === event &&
      latest.from === from &&
      latest.to === to &&
      now - latest.timestamp < STATE_EVENT_DEDUP_MS
    ) {
      return latest.id
    }

    const id = `ste_${now}_${Math.random().toString(36).slice(2, 6)}`
    const entry: StateEvent = {
      id,
      kind,
      event,
      message,
      from,
      to,
      timestamp: now,
      ...(opts.data ? { data: opts.data } : {}),
    }
    set((prev) => ({
      stateEvents: [entry, ...prev.stateEvents].slice(0, MAX_STATE_EVENTS),
    }))
    return id
  },

  clearStateEvents: () => {
    set({ stateEvents: [] })
  },

  getStateEvents: () => get().stateEvents,
}))

export const useErrorStore = Object.assign(
  <T>(selector: (state: ErrorStore) => T): T => useStore(errorStore, selector),
  { getState: errorStore.getState },
)

export function addGlobalError(
  err: unknown,
  source?: string,
  requestId?: string,
  opts?: {
    title?: string
    severity?: ErrorSeverity
    surface?: ErrorSurface
    journeyId?: string
    banner?: boolean
    toast?: boolean
  },
) {
  return useErrorStore.getState().addError(err, { source, requestId, ...opts })
}
