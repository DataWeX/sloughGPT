/**
 * Shared time formatting utilities.
 *
 * Consolidates duplicate formatTimestamp, formatTime, and formatDate
 * implementations across the codebase.
 *
 * Every formatter is total: null, undefined, and unparsable input renders
 * `''` instead of "Invalid Date" or a thrown RangeError. Use {@link toDate}
 * when you need the parsed Date yourself.
 */

/** Anything a timestamp formatter accepts. */
export type DateInput = Date | string | number | null | undefined

const MS_PER_SECOND = 1000
const MS_PER_MINUTE = 60 * MS_PER_SECOND
const MS_PER_HOUR = 60 * MS_PER_MINUTE
const MS_PER_DAY = 24 * MS_PER_HOUR

/**
 * Parse a timestamp into a valid `Date`, or `null` when it is missing or
 * unparsable (e.g. the legacy `…+00:00Z` backend format, which `new Date`
 * reports as `Invalid Date`).
 */
export function toDate(value: DateInput): Date | null {
  if (value == null || value === '') return null
  let d: Date
  if (value instanceof Date) d = value
  else if (typeof value === 'number') d = new Date(value)
  else d = new Date(value)
  return Number.isNaN(d.getTime()) ? null : d
}

/**
 * Format a timestamp as a relative time string ("just now", "5m ago", "2h ago").
 */
export function formatRelativeTime(date: DateInput): string {
  const d = toDate(date)
  if (!d) return ''
  const now = Date.now()
  const diffMs = now - d.getTime()
  const diffSec = Math.floor(diffMs / MS_PER_SECOND)
  const diffMins = Math.floor(diffMs / MS_PER_MINUTE)
  const diffHours = Math.floor(diffMs / MS_PER_HOUR)
  const diffDays = Math.floor(diffMs / MS_PER_DAY)

  if (diffSec < 60) return 'just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  return d.toLocaleDateString()
}

/**
 * Format a timestamp as a short relative string ("5m", "2h", "1d").
 */
export function formatShortRelative(date: DateInput): string {
  const d = toDate(date)
  if (!d) return ''
  const now = Date.now()
  const diffMs = now - d.getTime()
  const diffMins = Math.floor(diffMs / MS_PER_MINUTE)
  const diffHours = Math.floor(diffMs / MS_PER_HOUR)
  const diffDays = Math.floor(diffMs / MS_PER_DAY)

  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m`
  if (diffHours < 24) return `${diffHours}h`
  if (diffDays < 30) return `${diffDays}d`
  return d.toLocaleDateString()
}

/**
 * Format a timestamp as a full date+time string.
 */
export function formatDateTime(date: DateInput): string {
  const d = toDate(date)
  if (!d) return ''
  return d.toLocaleString()
}

/**
 * Format a timestamp as a short date string ("Jan 5, 2026").
 */
export function formatShortDate(date: DateInput): string {
  const d = toDate(date)
  if (!d) return ''
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

/**
 * Format a timestamp as a date+time with short month ("Jan 5, 2:30 PM").
 */
export function formatDateTimeShort(date: DateInput): string {
  const d = toDate(date)
  if (!d) return ''
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/**
 * Format a timestamp as a full date+time with short month ("Jan 5, 2026, 2:30 PM").
 */
export function formatDateTimeFull(date: DateInput): string {
  const d = toDate(date)
  if (!d) return ''
  return d.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/**
 * Format a timestamp as time only with seconds ("2:30:45 PM").
 */
export function formatTimeWithSeconds(date: DateInput): string {
  const d = toDate(date)
  if (!d) return ''
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

/**
 * Format a timestamp as time only ("2:30 PM").
 */
export function formatTimeShort(date: DateInput): string {
  const d = toDate(date)
  if (!d) return ''
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/**
 * Format a timestamp as US-style short date+time ("Jan 5, 2:30 PM").
 */
export function formatDateTimeUS(date: DateInput): string {
  const d = toDate(date)
  if (!d) return ''
  return d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

/**
 * Format a number of seconds as mm:ss.
 */
export function formatSeconds(seconds: number): string {
  if (!Number.isFinite(seconds)) return '0:00'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}
