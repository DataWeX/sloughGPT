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
 *   logger.error('Could not stream', { exception: 'AbortError' })
 *   logger.child('chat').info('user typed')
 */

import { PUBLIC_API_URL } from '@/lib/config'

export type LogTag =
  | 'REQ' | 'AUTH' | 'MODEL' | 'SOUL' | 'TRAIN' | 'INFRA'
  | 'START' | 'SLOW' | 'ERROR' | 'WARN' | 'OK'
  | 'CHAT' | 'IDLE' | 'DOWNLOAD' | 'INFERENCE' | 'WORKFLOW'
  | 'UI' | 'SYSTEM' | 'WEB'

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

const EVENT_TAG_MAP: Partial<Record<string, LogTag>> = {
  model_:       'MODEL',
  training_:    'TRAIN',
  session_:     'CHAT',
  stream_:      'CHAT',
  chat_:        'CHAT',
  webhook_:     'WORKFLOW',
  download_:    'DOWNLOAD',
  auth_:        'AUTH',
  vm_:          'INFRA',
  shell_:       'INFRA',
  soul_:        'SOUL',
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

/**
 * Batched, rate-limited HTTP transport for log records.
 *
 * Owns the buffered records, the flush timer, and the min-interval gate.
 * Each ``WebLogger`` references one transport; production loggers share the
 * module-wide default while tests can pass a fresh instance to keep batches
 * isolated instead of mutating one shared module-level pool.
 *
 * Side effects:
 *   - POSTs buffered records to ``<PUBLIC_API_URL>/errors/logs/ingest``
 *     when flushed; failures are swallowed (never retried in a loop).
 */
export class LogTransport {
  private _batch: LogRecord[] = []
  private _timer: ReturnType<typeof setTimeout> | null = null
  private _lastFlushAt = 0

  enqueue(record: LogRecord): void {
    this._batch.push(record)
    if (this._batch.length >= MAX_BATCH_SIZE) {
      if (this._timer) {
        clearTimeout(this._timer)
        this._timer = null
      }
      this.flush()
    } else {
      this._scheduleFlush()
    }
  }

  flush(): void {
    if (this._batch.length === 0) return
    const now = Date.now()
    if (now - this._lastFlushAt < MIN_FLUSH_INTERVAL_MS) {
      if (!this._timer) this._timer = setTimeout(() => this.flush(), MIN_FLUSH_INTERVAL_MS)
      return
    }
    const payload = this._batch
    this._batch = []
    this._timer = null
    this._lastFlushAt = now

    try {
      void fetch(`${PUBLIC_API_URL}/errors/logs/ingest`, {
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

  private _scheduleFlush(): void {
    if (this._timer) return
    this._timer = setTimeout(() => this.flush(), BATCH_INTERVAL_MS)
  }
}

let _sharedTransport: LogTransport | null = null
function _getSharedTransport(): LogTransport {
  if (!_sharedTransport) {
    _sharedTransport = new LogTransport()
    if (typeof window !== 'undefined') {
      window.addEventListener('beforeunload', () => _sharedTransport!.flush())
    }
  }
  return _sharedTransport
}

// ── WebLogger class ──────────────────────────────────────────────────

export class WebLogger {
  private _name: string
  private _level: LogLevel
  private _context: LogContext
  private _transport: LogTransport

  constructor(
    name = 'slo.web',
    level: LogLevel = 'info',
    context: LogContext = {},
    transport?: LogTransport,
  ) {
    this._name = name
    this._level = level
    this._context = context
    this._transport = transport ?? _getSharedTransport()
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
      this._transport,
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
    this._transport.flush()
  }

  // ── Event tracking ────────────────────────────────────────────────

  static inferTag(event: string): LogTag {
    for (const [prefix, tag] of Object.entries(EVENT_TAG_MAP)) {
      if (tag && event.startsWith(prefix)) return tag
    }
    return 'UI'
  }

  trackEvent(event: string, data?: Record<string, unknown>) {
    const tag = (data?.tag as LogTag) || WebLogger.inferTag(event)
    const summary = data
      ? Object.entries(data)
          .filter(([k]) => k !== 'tag')
          .map(([k, v]) => `${k}=${v}`)
          .join(' ')
      : ''

    const record: LogRecord = {
      level: 'info',
      logger: this._name,
      message: summary ? `${event} ${summary}` : event,
      timestamp: Date.now() / 1000,
      context: { ...data, tag },
    }
    this._transport.enqueue(record)

    if (IS_DEV) {
      console.debug(`[${this._name}]`, event, data)
    }
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

    // Production only forwards warnings+; dev forwards everything
    if (!IS_DEV && LEVEL_ORDER[level] < LEVEL_ORDER.warning) return
    this._transport.enqueue(record)
  }
}

/** @deprecated Use WebLogger directly. */
export const WebEventLogger = WebLogger

const _defaultEventLogger = new WebLogger('slo.web.ui')
export const trackEvent = (event: string, data?: Record<string, unknown>) =>
  _defaultEventLogger.trackEvent(event, data)

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
