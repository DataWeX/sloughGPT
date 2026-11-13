import { describe, it, expect } from 'vitest'
import { formatDuration, formatElapsed } from './formatDuration'

describe('formatDuration', () => {
  it('formats seconds under a minute', () => {
    expect(formatDuration(0)).toBe('0s')
    expect(formatDuration(42)).toBe('42s')
  })

  it('formats minutes and seconds', () => {
    expect(formatDuration(61)).toBe('1m 01s')
    expect(formatDuration(9 * 60 + 5)).toBe('9m 05s')
  })

  it('formats hours and minutes', () => {
    expect(formatDuration(3600)).toBe('1h 00m')
    expect(formatDuration(7200 + 35 * 60)).toBe('2h 35m')
  })

  it('handles null and non-finite input', () => {
    expect(formatDuration(null)).toBe('--')
    expect(formatDuration(Number.POSITIVE_INFINITY)).toBe('--')
    expect(formatDuration(Number.NaN)).toBe('--')
  })
})

describe('formatElapsed', () => {
  it('formats a valid range', () => {
    const start = new Date('2026-01-05T12:00:00Z')
    const end = new Date('2026-01-05T12:01:30Z')
    expect(formatElapsed(start, end)).toBe('1m 30s')
    expect(formatElapsed(start.getTime(), end.getTime())).toBe('1m 30s')
  })

  it('returns "" instead of "NaNm NaNs" for unparsable input', () => {
    expect(formatElapsed('not-a-date')).toBe('')
    expect(formatElapsed('2026-01-05T12:00:00+00:00Z')).toBe('')
    expect(formatElapsed(null)).toBe('')
    expect(formatElapsed(undefined)).toBe('')
    expect(formatElapsed(new Date('nope'))).toBe('')
  })

  it('returns "" when end precedes start', () => {
    expect(formatElapsed(new Date('2026-01-05T12:01:00Z'), new Date('2026-01-05T12:00:00Z'))).toBe(
      '',
    )
  })
})
