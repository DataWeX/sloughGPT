/**
 * WebLogger — structured logging for the browser frontend.
 *
 * Mirrors the Python ``domains.logging.WebLogger`` pattern.
 * Routes to console.debug/log/warn/error based on level.
 * Production: only warnings and errors are emitted.
 * Development: all levels are emitted.
 *
 * Usage:
 *   import { logger } from '@/lib/dev-log'
 *
 *   logger.info('message sent', { session_id: 'abc' })
 *   logger.error('stream failed', { exception: 'AbortError' })
 *   logger.child('chat').info('user typed')
 */

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

export class WebLogger {
  private _name: string
  private _level: LogLevel
  private _context: LogContext

  constructor(name = 'man.web', level: LogLevel = 'info', context: LogContext = {}) {
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
    return JSON.parse(raw)
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

    // In production, only emit warnings and above
    if (!IS_DEV && LEVEL_ORDER[level] < LEVEL_ORDER.warning) return

    const method = CONSOLE_METHOD[level]
    const prefix = `[${this._name}]`
    console[method](prefix, message, record)
  }
}

// ── Singleton ────────────────────────────────────────────────────────

export const logger = new WebLogger('man.web')

/**
 * Logs only in development — keeps production consoles clean for expected API/network failures.
 * @deprecated Use ``logger`` instead.
 */
export function devDebug(...args: unknown[]) {
  if (IS_DEV) {
    console.debug('[man]', ...args)
  }
}
