import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

vi.mock('./engine', () => ({
  SoulNetWebGPU: vi.fn().mockImplementation(() => ({
    init: vi.fn().mockResolvedValue(undefined),
    load: vi.fn().mockResolvedValue({}),
  })),
}))

vi.mock('./transformer-engine', () => ({
  SoulTransformerWebGPU: vi.fn().mockImplementation(() => ({
    init: vi.fn().mockResolvedValue(undefined),
    load: vi.fn().mockResolvedValue({}),
  })),
}))

vi.mock('./weights', () => ({
  parseSou: vi.fn(),
  inferArch: vi.fn(),
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { child: vi.fn(() => ({ info: vi.fn(), warning: vi.fn(), error: vi.fn() })) },
}))

import {
  SoulNetWebGPU,
  SoulTransformerWebGPU,
  createSoulEngine,
  createSoulEngineWorker,
  SoulEngineWorker,
  parseSou,
  inferArch,
} from './index'

describe('soulnet-webgpu index', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('exports SoulNetWebGPU class', () => {
    expect(SoulNetWebGPU).toBeDefined()
    expect(typeof SoulNetWebGPU).toBe('function')
  })

  it('exports SoulTransformerWebGPU class', () => {
    expect(SoulTransformerWebGPU).toBeDefined()
    expect(typeof SoulTransformerWebGPU).toBe('function')
  })

  it('exports parseSou function', () => {
    expect(parseSou).toBeDefined()
    expect(typeof parseSou).toBe('function')
  })

  it('exports inferArch function', () => {
    expect(inferArch).toBeDefined()
    expect(typeof inferArch).toBe('function')
  })

  it('createSoulEngine returns an engine instance', async () => {
    const engine = await createSoulEngine('/models/test.soul', {
      embedDim: 64,
      hiddenDim: 128,
      vocabSize: 50,
      numLayers: 2,
    })
    expect(engine).toBeDefined()
    expect(engine.init).toBeDefined()
    expect(engine.load).toBeDefined()
  })

  it('createSoulEngineWorker returns a SoulEngineWorker', () => {
    const worker = createSoulEngineWorker()
    expect(worker).toBeInstanceOf(SoulEngineWorker)
  })

  it('SoulEngineWorker has expected methods', () => {
    const worker = new SoulEngineWorker()
    expect(typeof worker.init).toBe('function')
    expect(typeof worker.load).toBe('function')
    expect(typeof worker.generate).toBe('function')
    expect(typeof worker.reset).toBe('function')
    expect(typeof worker.destroy).toBe('function')
  })

  it('SoulEngineWorker destroy does not throw', () => {
    const worker = new SoulEngineWorker()
    expect(() => worker.destroy()).not.toThrow()
  })

  it('SoulEngineWorker generate returns async iterator', () => {
    vi.stubGlobal('Worker', vi.fn().mockImplementation(() => ({
      postMessage: vi.fn(),
      onmessage: null,
      terminate: vi.fn(),
    })))
    const worker = new SoulEngineWorker()
    ;(worker as any).worker = {
      postMessage: vi.fn(),
      onmessage: null,
      terminate: vi.fn(),
    }
    const gen = worker.generate('test', 10)
    expect(gen[Symbol.asyncIterator]).toBeDefined()
    expect(typeof gen.next).toBe('function')
    expect(typeof gen.return).toBe('function')
    expect(typeof gen.throw).toBe('function')
  })

  it('SoulEngineWorker destroy clears pending resolvers', () => {
    const worker = new SoulEngineWorker()
    const fakeWorker = {
      postMessage: vi.fn(),
      onmessage: null,
      terminate: vi.fn(),
    }
    ;(worker as any).worker = fakeWorker
    ;(worker as any).resolvers.set('test', vi.fn())
    worker.destroy()
    expect((worker as any).resolvers.size).toBe(0)
    expect(fakeWorker.terminate).toHaveBeenCalled()
  })
})
