import { describe, it, expect } from 'vitest'
import { formatDuration } from './formatDuration'

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
