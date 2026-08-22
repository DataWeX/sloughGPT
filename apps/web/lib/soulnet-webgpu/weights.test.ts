import { describe, it, expect } from 'vitest'
import { parseSou, inferArch, guessShapes } from './weights'
import type { SoulTransformerArch } from './weights'
import { makeSou, makeSouV2, makeLstmSou, makeTransformerSou } from './__test-helper'

const META = { version: 3, soul_name: 'test-soul', soul_traits: {}, system_prompt: '', lineage: 'unit' }

describe('parseSou', () => {
  it('parses v3 metadata fields', () => {
    const sou = makeSou({ ...META, soul_name: 'friendly', system_prompt: 'be nice', step: 42 }, [])
    const cp = parseSou(sou)
    expect(cp.metadata.soul_name).toBe('friendly')
    expect(cp.metadata.system_prompt).toBe('be nice')
    expect(cp.metadata.step).toBe(42)
  })

  it('parses v3 param names, shapes and float32 data in order', () => {
    const p0 = new Float32Array([1, 2, 3])
    const p1 = new Float32Array([0.5, -1.25, 2, 3])
    const sou = makeSou(META, [
      { name: 'p0', shape: [3], data: p0 },
      { name: 'p1', shape: [2, 2], data: p1 },
    ])
    const cp = parseSou(sou)
    expect(Object.keys(cp.weights)).toEqual(['p0', 'p1'])
    expect(Array.from(cp.weights['p0'] as Float32Array)).toEqual([1, 2, 3])
    expect(Array.from(cp.weights['p1'] as Float32Array)).toEqual([0.5, -1.25, 2, 3])
    expect(cp.totalElements).toBe(7)
  })

  it('parses a standard LSTM checkpoint', () => {
    const sou = makeLstmSou({ e: 2, h: 4, v: 3, nl: 2 })
    const cp = parseSou(sou)
    expect(Object.keys(cp.weights).length).toBe(2 + 2 * 4 + 2)
    expect(cp.weights['p0' as const].length).toBe(3 * 2)
    expect(cp.weights['p2' as const].length).toBe(4 * 4 * 2)
    expect(cp.weights['p9' as const].length).toBe(4 * 4)
    expect(cp.weights['p10' as const].length).toBe(3 * 4)
    expect(cp.weights['p11' as const].length).toBe(3)
  })

  it('parses v2 legacy JSON weights', () => {
    const sou = makeSouV2(META, '{"p0":[1,2,3],"p1":[4,5]}')
    const cp = parseSou(sou)
    expect(cp.metadata.soul_name).toBe('test-soul')
    expect(Array.from(cp.weights['p0' as const])).toEqual([1, 2, 3])
    expect(Array.from(cp.weights['p1' as const])).toEqual([4, 5])
    expect(cp.totalElements).toBe(5)
  })

  it('v2 with empty weights yields empty dict', () => {
    const sou = makeSouV2(META, '{}')
    const cp = parseSou(sou)
    expect(Object.keys(cp.weights)).toEqual([])
    expect(cp.totalElements).toBe(0)
  })

  it('throws on invalid magic', () => {
    const bad = new Uint8Array([1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]).buffer
    expect(() => parseSou(bad)).toThrow(/Invalid \.sou magic/)
  })

  it('throws on non-JSON metadata', () => {
    const enc = new TextEncoder()
    const meta = 'not json'
    const bytes = new Uint8Array(4 + 4 + 4 + meta.length)
    bytes.set([0x53, 0x4f, 0x55, 0x4c], 0)
    const dv = new DataView(bytes.buffer)
    dv.setUint32(4, 3, true)
    dv.setUint32(8, meta.length, true)
    bytes.set(enc.encode(meta), 12)
    expect(() => parseSou(bytes.buffer)).toThrow()
  })
})

describe('inferArch', () => {
  it('detects lstm arch from new-format checkpoint', () => {
    const sou = makeLstmSou({ e: 2, h: 4, v: 3, nl: 1 })
    const arch = inferArch(sou)
    expect(arch.archType).toBe('lstm')
    expect(arch.embedDim).toBe(2)
    expect(arch.hiddenDim).toBe(4)
    expect(arch.vocabSize).toBe(3)
    expect(arch.numLayers).toBe(1)
  })

  it('detects transformer arch from checkpoint', () => {
    const sou = makeTransformerSou({ e: 4, L: 2, ff: 8, v: 3 })
    const arch = inferArch(sou)
    expect(arch.archType).toBe('transformer')
    expect(arch.embedDim).toBe(4)
    expect(arch.vocabSize).toBe(3)
    expect(arch.numLayers).toBe(2)
    // Transformer-specific fields
    const t = arch as SoulTransformerArch
    expect(t.numHeads).toBe(4)      // from metadata
    expect(t.numKVHeads).toBe(4)    // from metadata
    expect(t.dimFF).toBe(8)         // inferred from w1 shape (p7 = [8, 4])
    expect(t.maxSeqLen).toBe(256)   // from metadata
    expect(t.eps).toBe(1e-5)        // from metadata
  })

  it('throws on invalid magic', () => {
    const bad = new Uint8Array([9, 9, 9, 9, 0, 0, 0, 0]).buffer
    expect(() => inferArch(bad)).toThrow(/Invalid \.sou magic/)
  })
})

describe('guessShapes', () => {
  it('maps old-format params to named layers', () => {
    const e = 2, h = 3, v = 4
    const p0 = new Float32Array(v * e)
    const p1 = new Float32Array(4 * h * e)
    const p2 = new Float32Array(4 * h * h)
    const p3 = new Float32Array(v * h)
    const p4 = new Float32Array(v)
    const sou = makeSou(META, [
      { name: 'p0', shape: [v, e], data: p0 },
      { name: 'p1', shape: [4 * h, e], data: p1 },
      { name: 'p2', shape: [4 * h, h], data: p2 },
      { name: 'p3', shape: [v, h], data: p3 },
      { name: 'p4', shape: [v], data: p4 },
    ])
    const cp = parseSou(sou)
    const mapped = guessShapes(cp.weights, e, h, v, 1)
    expect(Object.keys(mapped)).toEqual(['embed.weight', 'lstm.W_ih', 'lstm.W_hh', 'fc_out.weight', 'fc_out.bias'])
    expect(mapped['fc_out.weight'].shape).toEqual([v, h])
    expect(mapped['fc_out.bias'].shape).toEqual([v])
  })
})
