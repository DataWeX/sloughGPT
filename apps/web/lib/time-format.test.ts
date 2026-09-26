import { describe, it, expect, vi, afterEach } from 'vitest'
import {
  formatRelativeTime,
  formatShortRelative,
  formatDateTime,
  formatShortDate,
  formatLocaleDate,
  formatMonthDay,
  formatDateTimeShort,
  formatDateTimeFull,
  formatTimeWithSeconds,
  formatTimeShort,
  formatDateTimeUS,
  formatSeconds,
  toDate,
} from './time-format'

describe('formatRelativeTime', () => {
  afterEach(() => vi.restoreAllMocks())

  it('returns "just now" for dates less than 60s ago', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-10T12:00:00Z'))
    expect(formatRelativeTime(new Date('2026-09-10T11:59:30Z'))).toBe('just now')
    expect(formatRelativeTime(new Date('2026-09-10T12:00:00Z'))).toBe('just now')
  })

  it('returns minutes ago for 1-59 minutes', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-10T12:00:00Z'))
    expect(formatRelativeTime(new Date('2026-09-10T11:55:00Z'))).toBe('5m ago')
    expect(formatRelativeTime(new Date('2026-09-10T11:01:00Z'))).toBe('59m ago')
  })

  it('returns hours ago for 1-23 hours', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-10T12:00:00Z'))
    expect(formatRelativeTime(new Date('2026-09-10T10:00:00Z'))).toBe('2h ago')
    expect(formatRelativeTime(new Date('2026-09-09T13:00:00Z'))).toBe('23h ago')
  })

  it('returns days ago for 1-6 days', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-10T12:00:00Z'))
    expect(formatRelativeTime(new Date('2026-09-09T12:00:00Z'))).toBe('1d ago')
    expect(formatRelativeTime(new Date('2026-09-04T12:00:00Z'))).toBe('6d ago')
  })

  it('returns locale date string for 7+ days', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-10T12:00:00Z'))
    const result = formatRelativeTime(new Date('2026-09-01T12:00:00Z'))
    expect(result).toMatch(/\d{1,2}\/\d{1,2}\/\d{4}/)
  })

  it('accepts string dates', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-10T12:00:00Z'))
    expect(formatRelativeTime('2026-09-10T11:55:00Z')).toBe('5m ago')
  })

  it('accepts numeric timestamps', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-10T12:00:00Z'))
    expect(formatRelativeTime(new Date('2026-09-10T11:55:00Z').getTime())).toBe('5m ago')
  })
})

describe('formatShortRelative', () => {
  afterEach(() => vi.restoreAllMocks())

  it('returns empty string for undefined', () => {
    expect(formatShortRelative(undefined)).toBe('')
  })

  it('returns empty string for invalid date', () => {
    expect(formatShortRelative('invalid')).toBe('')
  })

  it('returns "Just now" for <1 minute', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-10T12:00:00Z'))
    expect(formatShortRelative(new Date('2026-09-10T11:59:30Z'))).toBe('Just now')
  })

  it('returns short formats for minutes/hours/days', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-10T12:00:00Z'))
    expect(formatShortRelative(new Date('2026-09-10T11:55:00Z'))).toBe('5m')
    expect(formatShortRelative(new Date('2026-09-10T10:00:00Z'))).toBe('2h')
    expect(formatShortRelative(new Date('2026-09-09T12:00:00Z'))).toBe('1d')
  })

  it('returns locale date for 30+ days', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-10T12:00:00Z'))
    const result = formatShortRelative(new Date('2026-08-01T12:00:00Z'))
    expect(result).toMatch(/\d{1,2}\/\d{1,2}\/\d{4}/)
  })
})

describe('formatDateTime', () => {
  it('formats a Date object', () => {
    const d = new Date('2026-01-05T14:30:00Z')
    const result = formatDateTime(d)
    expect(result).toBeTruthy()
    expect(typeof result).toBe('string')
  })

  it('accepts string dates', () => {
    const result = formatDateTime('2026-01-05T14:30:00Z')
    expect(result).toBeTruthy()
  })
})

describe('formatShortDate', () => {
  it('formats as "Jan 5, 2026"', () => {
    const result = formatShortDate(new Date('2026-01-05T12:00:00Z'))
    expect(result).toContain('Jan')
    expect(result).toContain('5')
    expect(result).toContain('2026')
  })

  it('accepts string dates', () => {
    const result = formatShortDate('2026-01-05T12:00:00Z')
    expect(result).toContain('Jan')
  })
})

describe('formatDateTimeShort', () => {
  it('includes month, day, hour, minute', () => {
    const result = formatDateTimeShort(new Date('2026-01-05T14:30:00Z'))
    expect(result).toContain('Jan')
    expect(result).toContain('5')
  })

  it('returns empty string for undefined/null (backend strips null fields)', () => {
    expect(formatDateTimeShort(undefined)).toBe('')
    expect(formatDateTimeShort(null)).toBe('')
    expect(formatDateTimeShort('')).toBe('')
  })

  it('returns empty string for invalid dates instead of throwing', () => {
    expect(formatDateTimeShort('not-a-date')).toBe('')
    expect(formatDateTimeShort(new Date('invalid'))).toBe('')
  })
})

describe('formatDateTimeFull', () => {
  it('includes year, month, day, hour, minute', () => {
    const result = formatDateTimeFull(new Date('2026-01-05T14:30:00Z'))
    expect(result).toContain('Jan')
    expect(result).toContain('5')
    expect(result).toContain('2026')
  })
})

describe('formatTimeWithSeconds', () => {
  it('includes seconds', () => {
    const result = formatTimeWithSeconds(new Date('2026-01-05T14:30:45Z'))
    expect(result).toBeTruthy()
    expect(typeof result).toBe('string')
  })

  it('accepts numeric timestamps', () => {
    const result = formatTimeWithSeconds(new Date('2026-01-05T14:30:45Z').getTime())
    expect(result).toBeTruthy()
  })
})

describe('formatTimeShort', () => {
  it('returns time without seconds', () => {
    const result = formatTimeShort(new Date('2026-01-05T14:30:00Z'))
    expect(result).toBeTruthy()
    expect(typeof result).toBe('string')
  })
})

describe('formatDateTimeUS', () => {
  it('uses en-US locale', () => {
    const result = formatDateTimeUS(new Date('2026-01-05T14:30:00Z'))
    expect(result).toContain('Jan')
    expect(result).toContain('5')
  })

  it('accepts numeric timestamps', () => {
    const result = formatDateTimeUS(new Date('2026-01-05T14:30:00Z').getTime())
    expect(result).toContain('Jan')
  })
})

describe('formatSeconds', () => {
  it('formats 0 as "0:00"', () => {
    expect(formatSeconds(0)).toBe('0:00')
  })

  it('formats seconds only', () => {
    expect(formatSeconds(45)).toBe('0:45')
  })

  it('formats minutes and seconds', () => {
    expect(formatSeconds(90)).toBe('1:30')
  })

  it('pads single-digit seconds', () => {
    expect(formatSeconds(61)).toBe('1:01')
  })

  it('formats large values', () => {
    expect(formatSeconds(3661)).toBe('61:01')
  })

  it('truncates fractional seconds', () => {
    expect(formatSeconds(90.9)).toBe('1:30')
  })
})

describe('toDate', () => {
  it('parses valid ISO strings, numbers, and Date objects', () => {
    expect(toDate('2026-09-24T09:47:33.835728Z')?.getTime()).toBe(1790243253835)
    expect(toDate(0)?.getTime()).toBe(0)
    const d = new Date('2026-01-05T12:00:00Z')
    expect(toDate(d)).toBe(d)
  })

  it('returns null for missing or unparsable input', () => {
    expect(toDate(null)).toBeNull()
    expect(toDate(undefined)).toBeNull()
    expect(toDate('')).toBeNull()
    expect(toDate('not-a-date')).toBeNull()
    expect(toDate(new Date('nope'))).toBeNull()
  })

  it('returns null for the legacy "+00:00Z" backend format', () => {
    // Regression: this is what produced "Invalid Date" on the souls page.
    expect(new Date('2026-09-24T09:47:33.835728+00:00Z').getTime()).toBeNaN()
    expect(toDate('2026-09-24T09:47:33.835728+00:00Z')).toBeNull()
  })

  it('treats epoch-seconds numbers as seconds', () => {
    expect(toDate(1767225600)?.toISOString()).toBe('2026-01-01T00:00:00.000Z')
    expect(toDate(1767225600.5)?.toISOString()).toBe('2026-01-01T00:00:00.500Z')
  })

  it('keeps epoch-milliseconds numbers as milliseconds', () => {
    expect(toDate(1767225600000)?.toISOString()).toBe('2026-01-01T00:00:00.000Z')
    expect(toDate(Date.now())?.getTime()).toBeGreaterThan(1e12)
  })

  it('parses 10-digit epoch-seconds strings (backend str(time.time()))', () => {
    // Regression: adapters page fed these to new Date() → Invalid Date → blank.
    expect(toDate('1767225600')?.toISOString()).toBe('2026-01-01T00:00:00.000Z')
    expect(toDate('1767225600.5')?.toISOString()).toBe('2026-01-01T00:00:00.500Z')
    expect(formatLocaleDate('1767225600')).not.toBe('')
  })

  it('does not treat year-only strings as epoch seconds', () => {
    expect(toDate('2026')?.toISOString()).toBe('2026-01-01T00:00:00.000Z')
  })
})

describe('formatters never render "Invalid Date"', () => {
  const legacy = '2026-09-24T09:47:33.835728+00:00Z'
  const formatters = {
    formatRelativeTime,
    formatShortRelative,
    formatDateTime,
    formatShortDate,
    formatDateTimeShort,
    formatDateTimeFull,
    formatTimeWithSeconds,
    formatTimeShort,
    formatDateTimeUS,
  } as const

  it.each(Object.entries(formatters))('%s returns "" for legacy timestamps', (_name, fn) => {
    expect(fn(legacy)).toBe('')
  })

  it.each(Object.entries(formatters))('%s returns "" for null/undefined', (_name, fn) => {
    expect(fn(null)).toBe('')
    expect(fn(undefined)).toBe('')
  })

  it.each(Object.entries(formatters))('%s still formats valid dates', (_name, fn) => {
    expect(fn('2026-01-05T14:30:00Z')).not.toBe('')
    expect(fn(new Date('2026-01-05T14:30:00Z'))).not.toBe('')
  })
})

describe('formatSeconds guards non-finite input', () => {
  it('returns "0:00" for NaN/Infinity', () => {
    expect(formatSeconds(Number.NaN)).toBe('0:00')
    expect(formatSeconds(Number.POSITIVE_INFINITY)).toBe('0:00')
  })
})

describe('formatLocaleDate / formatMonthDay', () => {
  const valid = new Date('2026-01-05T12:30:00Z')

  it('formats valid dates', () => {
    expect(formatLocaleDate(valid)).toBe(valid.toLocaleDateString())
    expect(formatMonthDay(valid, 'en-US')).toBe('Jan 5')
    expect(formatMonthDay(valid)).toBe(
      valid.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
    )
  })

  it('returns "" for invalid and missing input', () => {
    for (const bad of [null, undefined, '', 'nope', '2026-01-05T12:00:00+00:00Z', new Date('x')]) {
      expect(formatLocaleDate(bad)).toBe('')
      expect(formatMonthDay(bad)).toBe('')
      expect(formatMonthDay(bad, 'en-US')).toBe('')
    }
  })
})
