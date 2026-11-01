import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest'
import { formatDate, truncateMessage } from './conversations-utils'

describe('formatDate', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('returns empty string for falsy or invalid dates', () => {
    expect(formatDate(undefined)).toBe('')
    expect(formatDate('')).toBe('')
    expect(formatDate('not-a-date')).toBe('')
  })

  it('returns "Just now" for timestamps under a minute old', () => {
    expect(formatDate('2025-12-31T23:59:30Z')).toBe('Just now')
  })

  it('returns minutes for timestamps under an hour old', () => {
    expect(formatDate('2025-12-31T23:55:00Z')).toBe('5m')
  })

  it('returns hours for timestamps under a day old', () => {
    expect(formatDate('2025-12-31T21:00:00Z')).toBe('3h')
  })

  it('returns days for timestamps under a week old', () => {
    expect(formatDate('2025-12-30T00:00:00Z')).toBe('2d')
  })

  it('returns a locale date for older timestamps', () => {
    expect(formatDate('2025-12-01T00:00:00Z')).toMatch(/2025/)
  })
})

describe('truncateMessage', () => {
  it('returns a placeholder for empty content', () => {
    expect(truncateMessage('')).toBe('Empty conversation')
    expect(truncateMessage(undefined as unknown as string)).toBe('Empty conversation')
  })

  it('uses only the first line', () => {
    expect(truncateMessage('hello world\nsecond line')).toBe('hello world')
  })

  it('truncates content longer than the max length with an ellipsis', () => {
    const long = 'a'.repeat(100)
    const result = truncateMessage(long)
    expect(result).toHaveLength(61)
    expect(result.endsWith('…')).toBe(true)
    expect(result.slice(0, 60)).toBe('a'.repeat(60))
  })

  it('leaves short content unchanged', () => {
    expect(truncateMessage('short message')).toBe('short message')
  })

  it('honors a custom max length', () => {
    expect(truncateMessage('abcdefghij', 5)).toBe('abcde…')
  })
})
