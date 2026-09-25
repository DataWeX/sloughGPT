/**
 * Shared formatting utilities for mobile screens.
 */

const MS_PER_MINUTE = 60_000
const MS_PER_HOUR = 60 * MS_PER_MINUTE
const MS_PER_DAY = 24 * MS_PER_HOUR

/**
 * Parse a timestamp into a valid Date, or null when it is missing or
 * unparsable (e.g. the legacy backend format `…+00:00Z`).
 * Numbers are treated as epoch milliseconds, matching `new Date(n)`.
 */
function toDate(value: number | string | Date | null | undefined): Date | null {
  if (value == null || value === '') return null
  const d = value instanceof Date ? value : new Date(value)
  return isNaN(d.getTime()) ? null : d
}

/**
 * Locale date+time, or '' when the input cannot be parsed.
 * Never renders "Invalid Date".
 */
export function formatDateTime(value: number | string | Date | null | undefined): string {
  const d = toDate(value)
  return d ? d.toLocaleString() : ''
}

/**
 * Locale date only, or '' when the input cannot be parsed.
 * Never renders "Invalid Date".
 */
export function formatDate(value: number | string | Date | null | undefined): string {
  const d = toDate(value)
  return d ? d.toLocaleDateString() : ''
}

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

  if (days >= 7) return formatDate(absMs)
  if (days > 0) return `${days}d ago`
  if (hrs > 0) return `${hrs}h ago`
  if (mins > 0) return `${mins}m ago`
  return 'just now'
}
