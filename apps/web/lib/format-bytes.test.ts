import { describe, it, expect } from 'vitest'
import { formatBytes, formatRelativeTime, MS_PER_MINUTE, MS_PER_HOUR, MS_PER_DAY } from './format-bytes'

describe('formatBytes', () => {
  it('returns — for zero', () => {
    expect(formatBytes(0)).toBe('—')
  })

  it('returns — for falsy', () => {
    expect(formatBytes(null as unknown as number)).toBe('—')
    expect(formatBytes(undefined as unknown as number)).toBe('—')
  })

  it('formats bytes', () => {
    expect(formatBytes(500)).toBe('500 B')
  })

  it('formats kilobytes', () => {
    expect(formatBytes(2048)).toBe('2.0 KB')
  })

  it('formats megabytes', () => {
    expect(formatBytes(5_242_880)).toBe('5.0 MB')
  })

  it('formats gigabytes', () => {
    expect(formatBytes(5_368_709_120)).toBe('5.00 GB')
  })

  it('formats very large values as GB', () => {
    expect(formatBytes(5_497_558_138_880)).toContain('GB')
  })

  it('formats negative values as bytes', () => {
    expect(typeof formatBytes(-1024)).toBe('string')
  })

  it('formats 1 byte', () => {
    expect(formatBytes(1)).toBe('1 B')
  })

  it('formats fractional KB', () => {
    expect(formatBytes(1536)).toBe('1.5 KB')
  })
})

describe('formatRelativeTime', () => {
  const now = Date.now()

  it('returns empty string for zero or invalid input', () => {
    expect(formatRelativeTime(0)).toBe('')
    expect(formatRelativeTime(-5)).toBe('')
  })

  it('says Just now within a minute', () => {
    expect(formatRelativeTime((now - 10 * 1000) / 1000)).toBe('Just now')
  })

  it('formats minutes ago', () => {
    expect(formatRelativeTime((now - 5 * MS_PER_MINUTE) / 1000)).toBe('5m ago')
  })

  it('formats hours ago', () => {
    expect(formatRelativeTime((now - 3 * MS_PER_HOUR) / 1000)).toBe('3h ago')
  })

  it('formats days ago within a week', () => {
    expect(formatRelativeTime((now - 2 * MS_PER_DAY) / 1000)).toBe('2d ago')
  })

  it('falls back to a locale date string past a week', () => {
    const ts = now - 20 * MS_PER_DAY
    const date = new Date(Math.floor(ts / 1000) * 1000).toLocaleDateString()
    expect(formatRelativeTime(ts / 1000)).toBe(date)
  })
})
