// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { NATIVE_PRESETS } from './useTrainingForm'

describe('NATIVE_PRESETS', () => {
  it('has 4 presets', () => {
    expect(NATIVE_PRESETS).toHaveLength(4)
  })

  it('each preset has required fields', () => {
    NATIVE_PRESETS.forEach(p => {
      expect(p.name).toBeTruthy()
      expect(p.description).toBeTruthy()
      expect(p.params).toBeTruthy()
      expect(p.embed).toBeGreaterThanOrEqual(16)
      expect(p.layers).toBeGreaterThanOrEqual(1)
      expect(p.heads).toBeGreaterThanOrEqual(1)
      expect(p.blockSize).toBeGreaterThanOrEqual(8)
      expect(p.epochs).toBeGreaterThanOrEqual(1)
      expect(p.lr).toBeGreaterThan(0)
      expect(p.batchSize).toBeGreaterThanOrEqual(1)
    })
  })

  it('presets are ordered from small to large', () => {
    const sizes = NATIVE_PRESETS.map(p => p.embed * p.layers)
    for (let i = 1; i < sizes.length; i++) {
      expect(sizes[i]).toBeGreaterThan(sizes[i - 1])
    }
  })

  it('Tiny preset uses fast-training defaults', () => {
    const tiny = NATIVE_PRESETS[0]
    expect(tiny.name).toBe('Tiny')
    expect(tiny.embed).toBe(64)
    expect(tiny.layers).toBe(2)
    expect(tiny.epochs).toBe(50)
  })

  it('Large preset uses best-quality defaults', () => {
    const large = NATIVE_PRESETS[3]
    expect(large.name).toBe('Large')
    expect(large.embed).toBe(256)
    expect(large.layers).toBe(4)
    expect(large.epochs).toBe(300)
    expect(large.lr).toBe(1e-4)
  })
})
