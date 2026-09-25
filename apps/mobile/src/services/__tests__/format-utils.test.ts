import { formatDate, formatDateTime, formatTimeAgo } from '../format-utils'

const VALID = '2026-01-05T12:30:00.000Z'
const LEGACY_BAD = '2026-09-24T08:48:15.788175+00:00Z'

describe('formatDate', () => {
  it('formats a valid timestamp like Date#toLocaleDateString', () => {
    expect(formatDate(VALID)).toBe(new Date(VALID).toLocaleDateString())
    expect(formatDate(new Date(VALID))).toBe(new Date(VALID).toLocaleDateString())
  })

  it('returns "" for the legacy +00:00Z backend format instead of "Invalid Date"', () => {
    expect(formatDate(LEGACY_BAD)).toBe('')
  })

  it('returns "" for missing and unparsable input', () => {
    expect(formatDate(null)).toBe('')
    expect(formatDate(undefined)).toBe('')
    expect(formatDate('')).toBe('')
    expect(formatDate('not a date')).toBe('')
    expect(formatDate(Number.NaN)).toBe('')
  })

  it('never returns "Invalid Date"', () => {
    for (const bad of [LEGACY_BAD, null, undefined, '', 'garbage', Number.NaN]) {
      expect(formatDate(bad as never)).not.toBe('Invalid Date')
      expect(formatDateTime(bad as never)).not.toBe('Invalid Date')
    }
  })
})

describe('formatDateTime', () => {
  it('formats a valid timestamp like Date#toLocaleString', () => {
    expect(formatDateTime(VALID)).toBe(new Date(VALID).toLocaleString())
  })

  it('returns "" for the legacy +00:00Z backend format', () => {
    expect(formatDateTime(LEGACY_BAD)).toBe('')
  })
})

describe('formatTimeAgo', () => {
  it('returns "" for an unparsable string rather than "Invalid Date"', () => {
    expect(formatTimeAgo('not a date')).toBe('')
  })

  it('handles the legacy +00:00Z format without throwing or printing Invalid Date', () => {
    const out = formatTimeAgo(LEGACY_BAD)
    expect(out).not.toContain('Invalid')
  })

  it('still formats valid relative times', () => {
    expect(formatTimeAgo(Date.now() - 30_000)).toBe('just now')
    const weekOld = Date.now() - 8 * 24 * 60 * 60 * 1000
    expect(formatTimeAgo(weekOld)).toBe(new Date(weekOld).toLocaleDateString())
  })
})
