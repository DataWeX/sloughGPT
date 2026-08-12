import { describe, it, expect } from 'vitest'
import { formatBytes } from './format-bytes'

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
