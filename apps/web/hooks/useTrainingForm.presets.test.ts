// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { BUILT_IN_PRESETS } from './useTrainingForm'

describe('BUILT_IN_PRESETS', () => {
  it('has 5 presets', () => {
    expect(BUILT_IN_PRESETS).toHaveLength(5)
  })

  it('each preset has required fields', () => {
    BUILT_IN_PRESETS.forEach(p => {
      expect(p.name).toBeTruthy()
      expect(p.description).toBeTruthy()
      expect(p.method).toBeTruthy()
      expect(p.epochs).toBeGreaterThanOrEqual(1)
      expect(p.lr).toBeGreaterThan(0)
      expect(p.batchSize).toBeGreaterThanOrEqual(1)
    })
  })

  it('native presets have architecture params', () => {
    const native = BUILT_IN_PRESETS.filter(p => p.method === 'native')
    expect(native.length).toBeGreaterThanOrEqual(1)
    native.forEach(p => {
      expect(p.nativeEmbed).toBeGreaterThanOrEqual(16)
      expect(p.nativeLayers).toBeGreaterThanOrEqual(1)
      expect(p.nativeHeads).toBeGreaterThanOrEqual(1)
      expect(p.nativeBlockSize).toBeGreaterThanOrEqual(8)
    })
  })

  it('Quick test preset uses fast-training defaults', () => {
    const quick = BUILT_IN_PRESETS[0]
    expect(quick.name).toBe('Quick test')
    expect(quick.method).toBe('distill')
    expect(quick.epochs).toBe(3)
  })

  it('Native large preset uses best-quality defaults', () => {
    const large = BUILT_IN_PRESETS.find(p => p.name === 'Native large')!
    expect(large.method).toBe('native')
    expect(large.nativeEmbed).toBe(256)
    expect(large.nativeLayers).toBe(4)
    expect(large.epochs).toBe(300)
    expect(large.lr).toBe(1e-4)
  })
})
