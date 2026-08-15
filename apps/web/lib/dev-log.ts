/**
 * WebLogger — structured logging for the browser frontend.
 *
 * Mirrors the Python ``domains.logging.WebLogger`` pattern.
 * Routes to console.debug/log/warn/error based on level.
 * Production: only warnings and errors are emitted.
 * Development: all levels are emitted.
 *
 * All log records are batched and forwarded to the backend via
 * POST /errors/logs/ingest, where they flow into the OutputBuffer
 * and appear in the SSE /system/stream + monitoring page OutputCard.
 *
 * Usage:
 *   import { logger } from '@/lib/dev-log'
 *
 *   logger.info('message sent', { session_id: 'abc' })
 *   logger.error('stream failed', { exception: 'AbortError' })
 *   logger.child('chat').info('user typed')
 */

import { PUBLIC_API_URL } from '@/lib/config'

type LogLevel = 'debug' | 'info' | 'warning' | 'error' | 'critical'

interface LogContext {
  [key: string]: unknown
}

interface LogRecord {
  level: LogLevel
  logger: string
  message: string
  timestamp: number
  context?: LogContext
  exception?: string
}

const LEVEL_ORDER: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warning: 2,
  error: 3,
  critical: 4,
}

const CONSOLE_METHOD: Record<LogLevel, 'debug' | 'log' | 'warn' | 'error'> = {
  debug: 'debug',
  info: 'log',
  warning: 'warn',
  error: 'error',
  critical: 'error',
}

const IS_DEV = process.env.NODE_ENV === 'development'

// ── Batch transport to backend ────────────────────────────────────────

const BATCH_INTERVAL_MS = 5000
const MAX_BATCH_SIZE = 20
const MIN_FLUSH_INTERVAL_MS = 3000 // don't flush more than once per 3s
let _logBatch: LogRecord[] = []
let _logTimer: ReturnType<typeof setTimeout> | null = null
let _lastFlushAt = 0

function _getApiUrl(): string {
  return PUBLIC_API_URL
}

function _flushLogs() {
  if (_logBatch.length === 0) return
  const now = Date.now()
  if (now - _lastFlushAt < MIN_FLUSH_INTERVAL_MS) {
    // Too soon — re-schedule instead of flushing
    if (!_logTimer) _logTimer = setTimeout(_flushLogs, MIN_FLUSH_INTERVAL_MS)
    return
  }
  const payload = _logBatch
  _logBatch = []
  _logTimer = null
  _lastFlushAt = now

  try {
    void fetch(`${_getApiUrl()}/errors/logs/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ logs: payload }),
      keepalive: true,
    }).catch(() => {
      /* silent — don't loop on network errors */
    })
  } catch {
    /* fetch unavailable or returned a non-promise — drop the flush */
  }
}

function _scheduleFlush() {
  if (_logTimer) return
  _logTimer = setTimeout(_flushLogs, BATCH_INTERVAL_MS)
}

function _enqueueLog(record: LogRecord) {
  // Always forward warnings+; in dev, forward everything
  if (!IS_DEV && LEVEL_ORDER[record.level] < LEVEL_ORDER.warning) return

  _logBatch.push(record)
  if (_logBatch.length >= MAX_BATCH_SIZE) {
    if (_logTimer) {
      clearTimeout(_logTimer)
      _logTimer = null
    }
    _flushLogs()
  } else {
    _scheduleFlush()
  }
}

// Flush remaining logs on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', _flushLogs)
}

// ── WebLogger class ──────────────────────────────────────────────────

export class WebLogger {
  private _name: string
  private _level: LogLevel
  private _context: LogContext

  constructor(name = 'slo.web', level: LogLevel = 'info', context: LogContext = {}) {
    this._name = name
    this._level = level
    this._context = context
  }

  get name() { return this._name }
  get level() { return this._level }
  set level(v: LogLevel) { this._level = v }

  setContext(ctx: LogContext) {
    Object.assign(this._context, ctx)
  }

  clearContext() {
    this._context = {}
  }

  child(suffix: string, context: LogContext = {}): WebLogger {
    return new WebLogger(
      `${this._name}.${suffix}`,
      this._level,
      { ...this._context, ...context },
    )
  }

  debug(message: string, context?: LogContext) { this._emit('debug', message, context) }
  info(message: string, context?: LogContext) { this._emit('info', message, context) }
  warning(message: string, context?: LogContext) { this._emit('warning', message, context) }
  error(message: string, opts?: { exception?: string } & LogContext) {
    const { exception, ...ctx } = opts || {}
    this._emit('error', message, ctx, exception)
  }
  critical(message: string, opts?: { exception?: string } & LogContext) {
    const { exception, ...ctx } = opts || {}
    this._emit('critical', message, ctx, exception)
  }

  /** Serialize a record to JSON (for transport / storage). */
  toJSON(record: LogRecord): string {
    return JSON.stringify(record, null, 0)
  }

  /** Deserialize a JSON string back into a LogRecord. */
  fromJSON(raw: string): LogRecord {
    try {
      return JSON.parse(raw)
    } catch {
      return { timestamp: Date.now(), level: 'error', logger: 'unknown', message: `Failed to parse log: ${raw.slice(0, 100)}`, context: {} }
    }
  }

  /** Flush any buffered logs to the backend immediately. */
  flush() {
    _flushLogs()
  }

  // ── Internal ──────────────────────────────────────────────────────

  private _emit(level: LogLevel, message: string, context?: LogContext, exception?: string) {
    if (LEVEL_ORDER[level] < LEVEL_ORDER[this._level]) return

    const record: LogRecord = {
      level,
      logger: this._name,
      message,
      timestamp: Date.now() / 1000,
      context: { ...this._context, ...context },
    }
    if (exception) record.exception = exception

    // Console output (always)
    const method = CONSOLE_METHOD[level]
    const prefix = `[${this._name}]`
    console[method](prefix, message, record)

    // Forward to backend
    _enqueueLog(record)
  }
}

// ── Singleton ────────────────────────────────────────────────────────

export const logger = new WebLogger('slo.web')

/**
 * Logs only in development — keeps production consoles clean for expected API/network failures.
 * @deprecated Use ``logger`` instead.
 */
export function devDebug(...args: unknown[]) {
  if (IS_DEV) {
    console.debug('[slo]', ...args)
  }
}
