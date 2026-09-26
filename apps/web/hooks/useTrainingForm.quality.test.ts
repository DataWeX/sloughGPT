// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { QUALITY_PRESETS, qualityFor, BUILT_IN_PRESETS, type Quality } from './useTrainingForm'

const ALL: Quality[] = ['low', 'medium', 'high']

describe('QUALITY_PRESETS', () => {
  it('stays inside the API validation bounds', () => {
    ALL.forEach((q) => {
      const size = QUALITY_PRESETS[q]
      expect(size.embed).toBeGreaterThanOrEqual(16)
      expect(size.embed).toBeLessThanOrEqual(1024)
      expect(size.layers).toBeGreaterThanOrEqual(1)
      expect(size.layers).toBeLessThanOrEqual(12)
      expect(size.heads).toBeGreaterThanOrEqual(1)
      expect(size.heads).toBeLessThanOrEqual(16)
    })
  })

  it('gets strictly bigger from low to high', () => {
    expect(QUALITY_PRESETS.low.embed).toBeLessThan(QUALITY_PRESETS.medium.embed)
    expect(QUALITY_PRESETS.medium.embed).toBeLessThan(QUALITY_PRESETS.high.embed)
    expect(QUALITY_PRESETS.low.layers).toBeLessThanOrEqual(QUALITY_PRESETS.medium.layers)
    expect(QUALITY_PRESETS.medium.layers).toBeLessThan(QUALITY_PRESETS.high.layers)
  })
})

describe('qualityFor', () => {
  it('maps every bucket back to its own quality', () => {
    ALL.forEach((q) => {
      const { embed, layers } = QUALITY_PRESETS[q]
      expect(qualityFor(embed, layers)).toBe(q)
    })
  })

  it('returns null for custom dimensions', () => {
    expect(qualityFor(128, 4)).toBeNull()
    expect(qualityFor(64, 4)).toBeNull()
  })
})

describe('built-in presets line up with the buckets', () => {
  it('Native small is low quality', () => {
    const small = BUILT_IN_PRESETS.find((p) => p.name === 'Native small')!
    expect(qualityFor(small.nativeEmbed!, small.nativeLayers!)).toBe('low')
  })

  it('Native large is medium quality', () => {
    const large = BUILT_IN_PRESETS.find((p) => p.name === 'Native large')!
    expect(qualityFor(large.nativeEmbed!, large.nativeLayers!)).toBe('medium')
  })
})
