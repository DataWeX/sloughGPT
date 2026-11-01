/**
 * Shared formatting utilities for mobile screens.
 */

const MS_PER_MINUTE = 60_000
const MS_PER_HOUR = 60 * MS_PER_MINUTE
const MS_PER_DAY = 24 * MS_PER_HOUR

/**
 * Convert a timestamp to a human-readable relative time string.
 * Accepts unix seconds, milliseconds, or ISO date strings.
 * Falls back to locale date after 7 days.
 */
export function formatTimeAgo(ts: number | string): string {
  let diffMs: number
  let absMs: number

  if (typeof ts === 'string') {
    const d = new Date(ts)
    if (isNaN(d.getTime())) return ''
    absMs = d.getTime()
  } else if (ts > 1e12) {
    absMs = ts
  } else {
    absMs = ts * 1000
  }

  diffMs = Date.now() - absMs
  if (diffMs < 0) return 'just now'

  const mins = Math.floor(diffMs / MS_PER_MINUTE)
  const hrs = Math.floor(mins / 60)
  const days = Math.floor(hrs / 24)

  if (days >= 7) return new Date(absMs).toLocaleDateString()
  if (days > 0) return `${days}d ago`
  if (hrs > 0) return `${hrs}h ago`
  if (mins > 0) return `${mins}m ago`
  return 'just now'
}
