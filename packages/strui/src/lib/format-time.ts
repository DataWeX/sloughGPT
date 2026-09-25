/**
 * Total date formatting helpers for shared UI.
 *
 * Every formatter returns `''` for null, undefined, or unparsable input
 * (e.g. the legacy backend format `…+00:00Z`, which `new Date` reports as
 * `Invalid Date`) instead of rendering "Invalid Date".
 */

export type DateInput = Date | string | number | null | undefined

export function toDate(value: DateInput): Date | null {
  if (value == null || value === '') return null
  const d = value instanceof Date ? value : new Date(value)
  return Number.isNaN(d.getTime()) ? null : d
}

/** Locale date+time ("Jan 5, 2026, 2:30:45 PM"), or `''`. */
export function formatDateTime(value: DateInput): string {
  const d = toDate(value)
  return d ? d.toLocaleString() : ''
}

/** Locale date only ("1/5/2026"), or `''`. */
export function formatDate(value: DateInput): string {
  const d = toDate(value)
  return d ? d.toLocaleDateString() : ''
}
