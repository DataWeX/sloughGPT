// ═══════════════════════════════════════════════════════════════════════════
// Transport ports — the three things this directory needs from a HOST app.
// ═══════════════════════════════════════════════════════════════════════════
//
// The transport used to import all three straight from apps/web (PUBLIC_API_URL
// from config, useAuthStore, logger) — 9 upward imports, which meant the
// transport depended on ONE app and could not be reused by another host.
// Dependency must run host → transport, so these are supplied here instead.
//
// IMPORTANT: every value is read at CALL time (`_getBaseUrl()`), never captured
// at construction. `apps/lib/api-client.ts` builds `apiClient` at module load,
// and ESM evaluates this directory BEFORE the host's wiring runs — a captured
// value would freeze the pre-wiring default for the app's lifetime.
//
// Defaults are deliberately inert: an unwired host gets same-origin paths, no
// auth header and no logging, instead of a crash.
// ═══════════════════════════════════════════════════════════════════════════

export interface TransportLogger {
  /** Method shorthand on purpose: bivariant params let any host logger (e.g. the
   *  app's WebLogger with its own LogContext type) satisfy this structurally. */
  debug(message: string, context?: unknown): void
  warning(message: string, context?: unknown): void
  error(message: string, context?: unknown): void
}

interface TransportPorts {
  getBaseUrl: () => string
  getToken: () => string | null
  logger: TransportLogger
}

const noop = (): void => {}
const silentLogger: TransportLogger = { debug: noop, warning: noop, error: noop }

let ports: TransportPorts = {
  getBaseUrl: () => '',
  getToken: () => null,
  logger: silentLogger,
}

/** Wire the host into the transport. Called once by apps/web's facade. */
export function configureTransport(next: Partial<TransportPorts>): void {
  ports = { ...ports, ...next }
}

export function _getBaseUrl(): string {
  return ports.getBaseUrl()
}

export function _getToken(): string | null {
  return ports.getToken()
}

export function _log(): TransportLogger {
  return ports.logger
}
