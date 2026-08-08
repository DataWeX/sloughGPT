import { describe, it, expect, afterEach, vi } from 'vitest'
import { SoulTransformerWebGPU } from './transformer-engine'
import { makeTransformerSou, makeWebGPU } from './__test-helper'
import type { SoulTransformerArch } from './weights'

const ARCH: SoulTransformerArch = {
  archType: 'transformer',
  embedDim: 4,
  numHeads: 1,
  numKVHeads: 1,
  numLayers: 1,
  dimFF: 8,
  vocabSize: 3,
  maxSeqLen: 16,
  eps: 1e-5,
}

const transformerSou = () => makeTransformerSou({ e: 4, L: 1, ff: 8, v: 3 })

describe('SoulTransformerWebGPU', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('throws WebGPU unavailable when no adapter exists', async () => {
    vi.stubGlobal('navigator', {})
    const engine = new SoulTransformerWebGPU()
    await expect(engine.init()).rejects.toThrow('WebGPU unavailable')
  })

  it('load parses checkpoint and marks engine ready', async () => {
    makeWebGPU()
    const engine = new SoulTransformerWebGPU()
    await engine.init()
    const cp = await engine.load(transformerSou(), ARCH)
    expect(cp.metadata.soul_name).toBe('test-soul')
    expect(engine.ready).toBe(true)
  })

  it('load fetches a URL when given a string', async () => {
    makeWebGPU()
    const sou = transformerSou()
    const fetchMock = vi.fn(async () => ({ arrayBuffer: async () => sou }))
    vi.stubGlobal('fetch', fetchMock)
    const engine = new SoulTransformerWebGPU()
    await engine.init()
    await engine.load('https://models.example/baby.sou', ARCH)
    expect(fetchMock).toHaveBeenCalledWith('https://models.example/baby.sou')
    expect(engine.ready).toBe(true)
  })

  it('forward returns vocab-sized logits and advances the KV cache', async () => {
    makeWebGPU(new Float32Array([0, 0, 0, 0]))
    const engine = new SoulTransformerWebGPU()
    await engine.init()
    await engine.load(transformerSou(), ARCH)
    const logits = await engine.forward(1)
    expect(logits).toHaveLength(3)
    const logits2 = await engine.forward(1)
    expect(logits2).toHaveLength(3)
  })

  it('load throws when architecture expects a missing param', async () => {
    makeWebGPU()
    const engine = new SoulTransformerWebGPU()
    await engine.init()
    await expect(engine.load(transformerSou(), { ...ARCH, numLayers: 2 })).rejects.toThrow('Missing p12')
  })

  it('generate yields maxTokens tokens', async () => {
    makeWebGPU(new Float32Array([0, 0, 0, 0]))
    const engine = new SoulTransformerWebGPU()
    await engine.init()
    await engine.load(transformerSou(), ARCH)
    const tokens: string[] = []
    for await (const t of engine.generate('', 3, 0)) tokens.push(t)
    expect(tokens.length).toBe(3)
    expect(tokens.every(t => typeof t === 'string')).toBe(true)
  })

  it('generate continues after resetState', async () => {
    makeWebGPU(new Float32Array([0, 0, 0, 0]))
    const engine = new SoulTransformerWebGPU()
    await engine.init()
    await engine.load(transformerSou(), ARCH)
    await engine.forward(0)
    engine.resetState()
    const tokens: string[] = []
    for await (const t of engine.generate('hi', 2, 0)) tokens.push(t)
    expect(tokens.length).toBe(2)
  })

  it('destroy clears ready', async () => {
    makeWebGPU()
    const engine = new SoulTransformerWebGPU()
    await engine.init()
    await engine.load(transformerSou(), ARCH)
    expect(engine.ready).toBe(true)
    engine.destroy()
    expect(engine.ready).toBe(false)
  })
})
