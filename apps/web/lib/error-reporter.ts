/**
 * Frontend error reporter — captures JS errors and sends them to the backend.
 *
 * Catches:
 *   - window.onerror (unhandled exceptions)
 *   - unhandledrejection (unhandled promise rejections)
 *
 * Hydration errors are handled by SuppressDevOverlay (persists to localStorage).
 * Non-hydration runtime errors are batched and POSTed to /errors/log.
 */

const BATCH_INTERVAL_MS = 5000
const MAX_BATCH_SIZE = 10

import { chatDB } from '@/lib/db'

const API_URL =
  (typeof window !== 'undefined' &&
    (window as any).__NEXT_PUBLIC_API_URL) ||
  process.env.NEXT_PUBLIC_API_URL ||
  'http://localhost:8000'

interface ErrorReport {
  message: string
  source: string
  stack?: string | null
  url?: string
  line?: number
  col?: number
  timestamp: string
  metadata?: Record<string, unknown>
}

let batch: ErrorReport[] = []
let timer: ReturnType<typeof setTimeout> | null = null

function flush() {
  if (batch.length === 0) return
  const payload = batch
  batch = []
  timer = null

  fetch(`${API_URL}/errors/log`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ errors: payload }),
    keepalive: true,
  }).catch(() => {
    /* silent — don't loop on network errors */
  })
}

function schedule() {
  if (timer) return
  timer = setTimeout(flush, BATCH_INTERVAL_MS)
}

function push(report: ErrorReport) {
  batch.push(report)
  if (batch.length >= MAX_BATCH_SIZE) {
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
    flush()
  } else {
    schedule()
  }
}

export function reportError(
  message: string,
  source: string = 'web',
  extra?: Partial<ErrorReport>,
) {
  push({
    message,
    source,
    timestamp: new Date().toISOString(),
    url: typeof window !== 'undefined' ? window.location.href : undefined,
    ...extra,
  })
}

function handleOnError(
  event: Event | string,
  source?: string,
  lineno?: number,
  colno?: number,
  error?: Error,
) {
  const message = typeof event === 'string' ? event : event.type
  push({
    message,
    source: 'window.onerror',
    stack: error?.stack || null,
    url: source || window.location.href,
    line: lineno,
    col: colno,
    timestamp: new Date().toISOString(),
  })
}

function handleRejection(event: PromiseRejectionEvent) {
  const reason = event.reason
  const message =
    reason?.message || reason?.toString?.() || 'Unhandled promise rejection'
  push({
    message,
    source: 'unhandledrejection',
    stack: reason?.stack || null,
    url: window.location.href,
    timestamp: new Date().toISOString(),
  })
}

let initialized = false

export function initErrorReporter() {
  if (initialized || typeof window === 'undefined') return
  initialized = true

  window.addEventListener('error', handleOnError as any)
  window.addEventListener('unhandledrejection', handleRejection as any)

  // Persist critical unhandled errors to Dexie for crash recovery
  // (hydration errors are handled separately by ErrorLifecycle)
  window.addEventListener('error', (event) => {
    try {
      const msg = (event as ErrorEvent).message
      if (!msg || msg.toLowerCase().includes('hydrat') || msg.includes('did not match')) return
      chatDB.addError(msg.slice(0, 500), 'unhandled').catch(() => {})
    } catch {}
  })

  // Flush remaining errors on page unload
  window.addEventListener('beforeunload', flush)
}
