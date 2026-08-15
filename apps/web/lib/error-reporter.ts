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

import { chatDB } from '@/lib/db'
import { PUBLIC_API_URL } from '@/lib/config'

const BATCH_INTERVAL_MS = 5000
const MAX_BATCH_SIZE = 10
const DEDUP_WINDOW_MS = 5000

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

interface ErrorReporterDeps {
  apiUrl?: string
  persist?: (message: string, source: string) => Promise<unknown>
}

const EXTENSION_RE =
  /metamask|chrome-extension|moz-extension|safari-web-extension|webextension|extension.*inject|content.?script/i

function isExtensionError(message: string, url?: string): boolean {
  return EXTENSION_RE.test(message) || !!(url && EXTENSION_RE.test(url))
}

/**
 * Batches runtime errors and POSTs them to the backend, with per-instance
 * state so it is testable without module-level resets. The default app
 * singleton is exposed via {@link reportError} and {@link initErrorReporter}.
 */
export class ErrorReporter {
  private _apiUrl: string
  private _persist: (message: string, source: string) => Promise<unknown>
  private _logger: { warn: (msg: string, ctx?: unknown) => void } | null = null
  private _batch: ErrorReport[] = []
  private _timer: ReturnType<typeof setTimeout> | null = null
  private _recentMessages = new Map<string, number>() // message -> last-sent timestamp
  private _initialized = false

  constructor(deps: ErrorReporterDeps = {}) {
    this._apiUrl = deps.apiUrl ?? PUBLIC_API_URL
    this._persist = deps.persist ?? ((message, source) => chatDB.addError(message, source))
  }

  /** Queue a report; flushes immediately at MAX_BATCH_SIZE, else on timer. */
  report(message: string, source: string = 'web', extra?: Partial<ErrorReport>): void {
    this._push({
      message,
      source,
      timestamp: new Date().toISOString(),
      url: typeof window !== 'undefined' ? window.location.href : undefined,
      ...extra,
    })
  }

  /** Attach window listeners; no-op in non-browser environments or if already run. */
  init(): void {
    if (this._initialized || typeof window === 'undefined') return
    this._initialized = true

    window.addEventListener('error', this._handleOnError)
    window.addEventListener('unhandledrejection', this._handleRejection)

    // Persist critical unhandled errors to Dexie for crash recovery
    // (hydration errors are handled separately by ErrorLifecycle)
    window.addEventListener('error', (event) => {
      try {
        const msg = (event as ErrorEvent).message
        if (!msg || msg.toLowerCase().includes('hydrat') || msg.includes('did not match')) return
        this._persist(msg.slice(0, 500), 'unhandled').catch((e: unknown) => {
          this._getLogger()?.warn('error-reporter: failed to persist error to IndexedDB', { error: e })
        })
      } catch {
        this._getLogger()?.warn('error-reporter: failed to read error event', {})
      }
    })

    // Flush remaining errors on page unload
    window.addEventListener('beforeunload', () => this._flush())
  }

  private _getLogger(): { warn: (msg: string, ctx?: unknown) => void } | null {
    if (!this._logger) {
      try { this._logger = require('@/lib/dev-log').logger } catch { /* fallback: no-op */ }
    }
    return this._logger
  }

  private _flush(): void {
    if (this._batch.length === 0) return
    const payload = this._batch
    this._batch = []
    this._timer = null

    fetch(`${this._apiUrl}/errors/log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ errors: payload }),
      keepalive: true,
    }).catch(() => {
      /* silent — don't loop on network errors */
    })
  }

  private _schedule(): void {
    if (this._timer) return
    this._timer = setTimeout(() => this._flush(), BATCH_INTERVAL_MS)
  }

  private _push(report: ErrorReport): void {
    // Dedup: same message within window → skip
    const now = Date.now()
    const lastSent = this._recentMessages.get(report.message)
    if (lastSent !== undefined && now - lastSent < DEDUP_WINDOW_MS) return
    this._recentMessages.set(report.message, now)
    // Prune old entries periodically
    if (this._recentMessages.size > 100) {
      for (const [msg, ts] of this._recentMessages) {
        if (now - ts > DEDUP_WINDOW_MS * 2) this._recentMessages.delete(msg)
      }
    }

    this._batch.push(report)
    if (this._batch.length >= MAX_BATCH_SIZE) {
      if (this._timer) {
        clearTimeout(this._timer)
        this._timer = null
      }
      this._flush()
    } else {
      this._schedule()
    }
  }

  private _handleOnError = (event: ErrorEvent): void => {
    this._push({
      message: event.message || event.type,
      source: 'window.onerror',
      stack: event.error?.stack || null,
      url: event.filename || window.location.href,
      line: event.lineno,
      col: event.colno,
      timestamp: new Date().toISOString(),
    })
  }

  private _handleRejection = (event: PromiseRejectionEvent): void => {
    const reason = event.reason
    const message =
      reason?.message || reason?.toString?.() || 'Unhandled promise rejection'
    this._push({
      message,
      source: 'unhandledrejection',
      stack: reason?.stack || null,
      url: window.location.href,
      timestamp: new Date().toISOString(),
    })
  }
}

const _defaultReporter = new ErrorReporter()

export function reportError(
  message: string,
  source: string = 'web',
  extra?: Partial<ErrorReport>,
): void {
  _defaultReporter.report(message, source, extra)
}

export function initErrorReporter(): void {
  _defaultReporter.init()
}
