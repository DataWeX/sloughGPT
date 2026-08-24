'use client'

/**
 * Convert a timestamp to a human-readable relative time string.
 * Accepts unix seconds, milliseconds, ISO date strings, or Date objects.
 */
export function timeAgo(ts: number | string | Date | undefined | null): string {
  if (ts == null) return 'never'

  let diffMs: number

  if (ts instanceof Date) {
    diffMs = Date.now() - ts.getTime()
  } else if (typeof ts === 'string') {
    try {
      const d = new Date(ts)
      if (isNaN(d.getTime())) return ''
      diffMs = Date.now() - d.getTime()
    } catch {
      return ''
    }
  } else if (ts === 0) {
    return 'never'
  } else if (ts > 1e12) {
    diffMs = Date.now() - ts
  } else {
    diffMs = Date.now() - ts * 1000
  }

  if (diffMs < 0) return 'just now'

  const mins = Math.floor(diffMs / 60000)
  const hrs = Math.floor(mins / 60)
  const days = Math.floor(hrs / 24)

  if (days > 0) return `${days}d ago`
  if (hrs > 0) return `${hrs}h ago`
  if (mins > 0) return `${mins}m ago`
  return 'just now'
}
