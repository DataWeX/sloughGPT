import { describe, it, expect, vi, afterEach } from 'vitest'
import { timeAgo } from './time-ago'

describe('timeAgo', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('returns "never" for null/undefined', () => {
    expect(timeAgo(null)).toBe('never')
    expect(timeAgo(undefined)).toBe('never')
  })

  it('returns "never" for zero', () => {
    expect(timeAgo(0)).toBe('never')
  })

  it('returns "just now" for future timestamps', () => {
    const future = Math.floor(Date.now() / 1000) + 60
    expect(timeAgo(future)).toBe('just now')
  })

  it('formats minutes ago (unix seconds)', () => {
    const now = Math.floor(Date.now() / 1000)
    expect(timeAgo(now - 120)).toBe('2m ago')
  })

  it('formats hours ago (unix seconds)', () => {
    const now = Math.floor(Date.now() / 1000)
    expect(timeAgo(now - 7200)).toBe('2h ago')
  })

  it('formats days ago (unix seconds)', () => {
    const now = Math.floor(Date.now() / 1000)
    expect(timeAgo(now - 172800)).toBe('2d ago')
  })

  it('handles milliseconds (>1e12)', () => {
    const now = Date.now()
    expect(timeAgo(now - 120000)).toBe('2m ago')
  })

  it('handles Date objects', () => {
    const d = new Date(Date.now() - 3600000)
    expect(timeAgo(d)).toBe('1h ago')
  })

  it('handles ISO date strings', () => {
    const d = new Date(Date.now() - 86400000)
    expect(timeAgo(d.toISOString())).toBe('1d ago')
  })

  it('returns empty string for invalid date strings', () => {
    expect(timeAgo('not-a-date')).toBe('')
  })

  it('returns "just now" for very recent timestamps', () => {
    const now = Math.floor(Date.now() / 1000)
    expect(timeAgo(now)).toBe('just now')
  })
})
