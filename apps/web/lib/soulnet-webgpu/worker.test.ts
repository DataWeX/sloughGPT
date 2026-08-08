import { describe, it, expect, afterEach, vi } from 'vitest'
import { SoulEngineWorker, createSoulEngine, createSoulEngineWorker } from './index'
import { makeLstmSou, makeWebGPU, stubWorker } from './__test-helper'
import type { SoulNetConfig } from './engine'

const CFG: SoulNetConfig = { embedDim: 2, hiddenDim: 2, vocabSize: 3, numLayers: 1, charset: 'abc' }

describe('SoulEngineWorker', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('runs the full init/load/reset/generate message protocol', async () => {
    makeWebGPU(new Float32Array([0, 0]))
    const getLatest = stubWorker()
    const w = new SoulEngineWorker()

    const initP = w.init()
    const fw = getLatest()!
    expect(fw.messages[0]).toEqual({ type: 'init' })
    fw.emit({ type: 'ready' })
    await initP

    const loadP = w.load('https://models.example/friendly.sou', CFG)
    expect(fw.messages[1]).toMatchObject({ type: 'loaded', url: 'https://models.example/friendly.sou' })
    fw.emit({ type: 'loaded', metadata: { version: 3 } })
    await expect(loadP).resolves.toEqual({ metadata: { version: 3 } })

    const resetP = w.reset()
    fw.emit({ type: 'reset-done' })
    await resetP

    const gen = w.generate('', 2, 0)
    const tokenP = gen.next()
    fw.emit({ type: 'token', token: 'a' })
    await expect(tokenP).resolves.toEqual({ value: 'a', done: false })
    const doneP = gen.next()
    fw.emit({ type: 'done' })
    await expect(doneP).resolves.toEqual({ value: undefined, done: true })
  })

  it('destroy terminates the worker', async () => {
    makeWebGPU()
    const getLatest = stubWorker()
    const w = new SoulEngineWorker()
    const initP = w.init()
    const fw = getLatest()!
    fw.emit({ type: 'ready' })
    await initP
    w.destroy()
    expect(fw.terminated).toBe(true)
  })

  it('createSoulEngineWorker returns a SoulEngineWorker', () => {
    stubWorker()
    expect(createSoulEngineWorker()).toBeInstanceOf(SoulEngineWorker)
  })
})

describe('createSoulEngine', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('inits, loads a URL, and generates tokens', async () => {
    makeWebGPU(new Float32Array([0, 0]))
    const sou = makeLstmSou({ e: 2, h: 2, v: 3, nl: 1 }, { p7: new Float32Array([1, 5, 2]) })
    vi.stubGlobal('fetch', vi.fn(async () => ({ arrayBuffer: async () => sou })))
    const engine = await createSoulEngine('https://models.example/friendly.sou', CFG)
    expect(engine.ready).toBe(true)
    const tokens: string[] = []
    for await (const t of engine.generate('', 2, 0)) tokens.push(t)
    expect(tokens).toEqual(['b', 'b'])
  })
})

describe('worker.ts protocol', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('answers init/load/generate/reset and errors on unknown types', async () => {
    makeWebGPU(new Float32Array([0, 0]))
    const sou = makeLstmSou({ e: 2, h: 2, v: 3, nl: 1 })
    vi.stubGlobal('fetch', vi.fn(async () => ({ arrayBuffer: async () => sou })))
    const selfStub: {
      postMessage: ReturnType<typeof vi.fn>
      onmessage?: ((e: { data: Record<string, unknown> }) => void) | null
    } = { postMessage: vi.fn() }
    vi.stubGlobal('self', selfStub)

    await import('./worker')
    const onmessage = selfStub.onmessage as (e: { data: Record<string, unknown> }) => Promise<void>
    const post = selfStub.postMessage as ReturnType<typeof vi.fn>

    onmessage({ data: { type: 'init' } })
    await vi.waitFor(() => expect(post).toHaveBeenCalledWith({ type: 'ready' }))

    onmessage({ data: { type: 'load', url: 'https://models.example/friendly.sou', config: CFG } })
    await vi.waitFor(() =>
      expect(post).toHaveBeenCalledWith(expect.objectContaining({ type: 'loaded', metadata: expect.any(Object) })),
    )

    onmessage({ data: { type: 'generate', prompt: '', maxTokens: 1, temperature: 0 } })
    await vi.waitFor(() => expect(post).toHaveBeenCalledWith({ type: 'token', token: expect.any(String) }))
    await vi.waitFor(() => expect(post).toHaveBeenCalledWith({ type: 'done' }))

    onmessage({ data: { type: 'reset' } })
    await vi.waitFor(() => expect(post).toHaveBeenCalledWith({ type: 'reset-done' }))

    onmessage({ data: { type: 'bogus' } })
    await vi.waitFor(() =>
      expect(post).toHaveBeenCalledWith({ type: 'error', message: expect.stringContaining('bogus') }),
    )
  })

  it('errors on generate before init', async () => {
    makeWebGPU(new Float32Array([0, 0]))
    vi.resetModules()
    const selfStub: {
      postMessage: ReturnType<typeof vi.fn>
      onmessage?: ((e: { data: Record<string, unknown> }) => void) | null
    } = { postMessage: vi.fn() }
    vi.stubGlobal('self', selfStub)
    await import('./worker')
    const onmessage = selfStub.onmessage as (e: { data: Record<string, unknown> }) => Promise<void>
    const post = selfStub.postMessage as ReturnType<typeof vi.fn>
    onmessage({ data: { type: 'generate', prompt: 'x', maxTokens: 1, temperature: 0 } })
    await vi.waitFor(() => expect(post).toHaveBeenCalledWith({ type: 'error', message: 'Not initialized' }))
  })
})
