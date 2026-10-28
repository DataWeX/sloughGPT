import { describe, it, expect } from 'vitest'
import { computeGenerationEta, formatEtaDuration } from './generation-eta'

describe('computeGenerationEta', () => {
  it('returns null without elapsed time', () => {
    expect(computeGenerationEta({ tokens: 10, elapsedMs: 0 })).toBeNull()
    expect(computeGenerationEta({ tokens: 10, elapsedMs: -5 })).toBeNull()
  })

  it('returns null without tokens', () => {
    expect(computeGenerationEta({ tokens: 0, elapsedMs: 1000 })).toBeNull()
  })

  it('computes cumulative tokens per second', () => {
    const g = computeGenerationEta({ tokens: 50, elapsedMs: 5000 })
    expect(g?.tokensPerSec).toBeCloseTo(10)
  })

  it('projects eta against the max token cap', () => {
    const g = computeGenerationEta({ tokens: 50, elapsedMs: 5000, maxTokens: 200 })
    expect(g?.etaMs).toBeCloseTo(15000)
  })

  it('returns null eta when no cap is set', () => {
    const g = computeGenerationEta({ tokens: 50, elapsedMs: 5000 })
    expect(g?.etaMs).toBeNull()
  })

  it('returns null eta once the cap is reached', () => {
    const g = computeGenerationEta({ tokens: 200, elapsedMs: 5000, maxTokens: 200 })
    expect(g?.etaMs).toBeNull()
  })

  it('returns null eta for non-positive caps', () => {
    expect(computeGenerationEta({ tokens: 1, elapsedMs: 100, maxTokens: 0 })?.etaMs).toBeNull()
  })
})

describe('formatEtaDuration', () => {
  it('formats sub-90s as seconds', () => {
    expect(formatEtaDuration(15000)).toBe('15s')
    expect(formatEtaDuration(89000)).toBe('89s')
  })

  it('formats longer spans as minutes and seconds', () => {
    expect(formatEtaDuration(90000)).toBe('1m 30s')
    expect(formatEtaDuration(150000)).toBe('2m 30s')
  })

  it('never shows zero for a positive eta', () => {
    expect(formatEtaDuration(100)).toBe('1s')
  })
})
