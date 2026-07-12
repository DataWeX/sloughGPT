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
})
