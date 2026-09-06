/**
 * Extract a human-readable error message from an unknown caught value.
 * Handles Error instances, strings, and common API error shapes (FastAPI, etc.).
 */
export function extractErrorMessage(err: unknown, fallback = 'Unknown error'): string {
  if (typeof err === 'string') return err
  if (err instanceof Error) return err.message || err.name || fallback
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

/**
 * Patterns that indicate minified/irrelevant stack frames.
 */
const NOISE_PATTERNS = [
  /^\s*at\s+webpack/,
  /^\s*at\s+_next\//,
  /^\s*at\s+https?:\/\/localhost/,
  /^\s*at\s+https?:\/\/127\./,
  /^\s*at\s+Object\.<anonymous>/,
  /^\s*at\s+Module\._compile/,
  /^\s*at\s+node:/,
  /^\s*at\s+internal/,
  /^\s*at\s+eval\s/,
]

/**
 * Extract the first meaningful frame(s) from a stack trace.
 * Returns up to `maxFrames` clean lines like: `ComponentName (file:line:col)`.
 */
export function formatStackTrace(stack: string | undefined, maxFrames = 5): string[] {
  if (!stack) return []

  const lines = stack.split('\n')
  const frames: string[] = []

  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed.startsWith('at ')) continue
    if (NOISE_PATTERNS.some(p => p.test(trimmed))) continue
    // Clean up webpack chunk hashes: "at eu (http://localhost:3000/_next/static/chunks/abc.js:1:12345)"
    const cleaned = trimmed
      .replace(/https?:\/\/[^)]+\.js:\d+:\d+/g, (m) => {
        // Extract just filename and position
        const match = m.match(/([^/]+\.js):(\d+):(\d+)/)
        if (match) return `${match[1]}:${match[2]}:${match[3]}`
        return m
      })
      .replace(/^\s*at\s+/, '')
    frames.push(cleaned)
    if (frames.length >= maxFrames) break
  }

  return frames
}

/**
 * Get the error type name (e.g. "TypeError", "ReferenceError") or null if generic.
 */
export function getErrorType(error: Error): string | null {
  if (!error.name || error.name === 'Error') return null
  return error.name
}
