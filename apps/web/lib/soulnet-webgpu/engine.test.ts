import { describe, it, expect, afterEach, vi } from 'vitest'
import { SoulNetWebGPU, type SoulNetConfig } from './engine'
import { makeLstmSou, makeWebGPU } from './__test-helper'

const CFG: SoulNetConfig = { embedDim: 2, hiddenDim: 2, vocabSize: 3, numLayers: 1, charset: 'abc' }

const lstmSou = (overrides: Record<string, Float32Array> = {}) =>
  makeLstmSou({ e: 2, h: 2, v: 3, nl: 1 }, overrides)

describe('SoulNetWebGPU', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('falls back to CPU-only mode when no adapter exists', async () => {
    vi.stubGlobal('navigator', {})
    const engine = new SoulNetWebGPU()
    await engine.init()
    expect(engine.cpuOnly).toBe(true)
  })

  it('init is idempotent', async () => {
    const fake = makeWebGPU()
    const engine = new SoulNetWebGPU()
    await engine.init()
    await engine.init()
    expect(fake.device.createShaderModule).toHaveBeenCalledTimes(1)
    expect(fake.device.createComputePipeline).toHaveBeenCalledTimes(1)
  })

  it('load parses checkpoint and marks engine ready', async () => {
    makeWebGPU()
    const engine = new SoulNetWebGPU()
    await engine.init()
    const cp = await engine.load(lstmSou(), CFG)
    expect(cp.metadata.soul_name).toBe('test-soul')
    expect(engine.metadata?.version).toBe(3)
    expect(engine.ready).toBe(true)
  })

  it('load fetches a URL when given a string', async () => {
    makeWebGPU()
    const sou = lstmSou()
    const fetchMock = vi.fn(async () => ({ ok: true, status: 200, arrayBuffer: async () => sou }))
    vi.stubGlobal('fetch', fetchMock)
    const engine = new SoulNetWebGPU()
    await engine.init()
    await engine.load('https://models.example/friendly.sou', CFG)
    expect(fetchMock).toHaveBeenCalledWith('https://models.example/friendly.sou')
    expect(engine.ready).toBe(true)
  })

  it('forward returns vocab-sized logits dominated by fc bias', async () => {
    makeWebGPU(new Float32Array([0, 0]))
    const engine = new SoulNetWebGPU()
    await engine.init()
    await engine.load(lstmSou({ p7: new Float32Array([1, 5, 2]) }), CFG)
    const logits = await engine.forward(0)
    expect(logits).toHaveLength(3)
    expect(logits[1]).toBe(5)
  })

  it('generate with temperature 0 yields argmax token repeatedly', async () => {
    makeWebGPU(new Float32Array([0, 0]))
    const engine = new SoulNetWebGPU()
    await engine.init()
    await engine.load(lstmSou({ p7: new Float32Array([1, 5, 2]) }), CFG)
    const tokens: string[] = []
    for await (const t of engine.generate('', 3, 0)) tokens.push(t)
    expect(tokens).toEqual(['b', 'b', 'b'])
  })

  it('generate stops early on eosToken', async () => {
    makeWebGPU(new Float32Array([0, 0]))
    const engine = new SoulNetWebGPU()
    await engine.init()
    // p7 bias makes token 1 (b) the argmax
    await engine.load(lstmSou({ p7: new Float32Array([1, 5, 2]) }), CFG)
    const tokens: string[] = []
    // eosToken=1 means stop when producing 'b' (token 1)
    for await (const t of engine.generate('', 10, 0, 1)) tokens.push(t)
    expect(tokens).toEqual([])
  })

  it('generate with eosToken=0 never stops early', async () => {
    makeWebGPU(new Float32Array([0, 0]))
    const engine = new SoulNetWebGPU()
    await engine.init()
    await engine.load(lstmSou({ p7: new Float32Array([1, 5, 2]) }), CFG)
    const tokens: string[] = []
    for await (const t of engine.generate('', 5, 0, 0)) tokens.push(t)
    expect(tokens).toHaveLength(5)
  })

  it('forward before load throws', async () => {
    makeWebGPU()
    const engine = new SoulNetWebGPU()
    await engine.init()
    await expect(engine.forward(0)).rejects.toThrow('Engine not ready')
  })

  it('resetState is safe before load', async () => {
    makeWebGPU()
    const engine = new SoulNetWebGPU()
    await engine.init()
    expect(() => engine.resetState()).not.toThrow()
  })

  it('destroy releases all created buffers and clears ready', async () => {
    const fake = makeWebGPU(new Float32Array([0, 0]))
    const engine = new SoulNetWebGPU()
    await engine.init()
    await engine.load(lstmSou(), CFG)
    expect(fake.buffers.length).toBeGreaterThan(0)
    engine.destroy()
    expect(engine.ready).toBe(false)
    expect(fake.buffers.every(b => b.destroy.mock.calls.length > 0)).toBe(true)
  })
})
