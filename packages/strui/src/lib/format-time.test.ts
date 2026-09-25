import { describe, expect, it } from 'vitest'

import { formatDate, formatDateTime, toDate } from './format-time'

const VALID = '2026-01-05T12:30:00.000Z'
const LEGACY_BAD = '2026-09-24T08:48:15.788175+00:00Z'

describe('toDate', () => {
  it('parses valid timestamps, epoch milliseconds, and Date instances', () => {
    expect(toDate(VALID)?.toISOString()).toBe(VALID)
    expect(toDate(0)?.getTime()).toBe(0)
    const now = new Date()
    expect(toDate(now)?.getTime()).toBe(now.getTime())
  })

  it('returns null for the legacy +00:00Z backend format', () => {
    expect(toDate(LEGACY_BAD)).toBeNull()
  })

  it('returns null for null, undefined, empty, and garbage input', () => {
    expect(toDate(null)).toBeNull()
    expect(toDate(undefined)).toBeNull()
    expect(toDate('')).toBeNull()
    expect(toDate('not a date')).toBeNull()
    expect(toDate(Number.NaN)).toBeNull()
  })
})

describe('formatDateTime / formatDate', () => {
  it('matches native locale formatting for valid input', () => {
    expect(formatDateTime(VALID)).toBe(new Date(VALID).toLocaleString())
    expect(formatDate(VALID)).toBe(new Date(VALID).toLocaleDateString())
  })

  it('returns "" instead of "Invalid Date" for unparsable input', () => {
    for (const bad of [LEGACY_BAD, null, undefined, '', 'garbage', Number.NaN]) {
      expect(formatDateTime(bad)).toBe('')
      expect(formatDate(bad)).toBe('')
      expect(formatDateTime(bad)).not.toBe('Invalid Date')
      expect(formatDate(bad)).not.toBe('Invalid Date')
    }
  })
})
