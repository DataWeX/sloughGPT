/**
 * Extract a human-readable error message from an unknown caught value.
 * Handles Error instances, strings, and common API error shapes (FastAPI, etc.).
 */
export function extractErrorMessage(err: unknown, fallback = 'Unknown error'): string {
  if (typeof err === 'string') return err
  if (err instanceof Error) return err.message
  if (err && typeof err === 'object') {
    const e = err as Record<string, unknown>
    if (typeof e.message === 'string') return e.message
    if (typeof e.error === 'string') return e.error
    if (typeof e.detail === 'string') return e.detail
    if (typeof e.msg === 'string') return e.msg
    try { return JSON.stringify(err).slice(0, 300) } catch { return fallback }
  }
  if (err != null) return String(err)
  return fallback
}

/**
 * Format an error for toast display.
 */
export function formatToastError(err: unknown, prefix: string): string {
  return `${prefix}: ${extractErrorMessage(err)}`
}
