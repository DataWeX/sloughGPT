/**
 * SoulNet WebGPU — Browser-side neural inference.
 *
 * Loads a .sou checkpoint and runs SoulNet LSTM forward passes
 * on the GPU, entirely in the browser tab. No server needed.
 *
 * Example:
 *   import { createSoulEngine } from '@/lib/soulnet-webgpu'
 *
 *   const engine = await createSoulEngine('/models/friendly.soul', {
 *     embedDim: 256, hiddenDim: 512, vocabSize: 50, numLayers: 2
 *   })
 *   for await (const token of engine.generate('hello', 100)) {
 *     text += token
 *   }
 */

import { SoulNetWebGPU } from './engine'
import type { SoulNetConfig } from './engine'
export type { SoulNetConfig } from './engine'
export { SoulNetWebGPU }
import { SoulTransformerWebGPU } from './transformer-engine'
export { SoulTransformerWebGPU }
export { parseSou, inferArch } from './weights'
export type { SoulCheckpoint, SoulMetadata, SoulWeights, SoulNetArch, SoulTransformerArch } from './weights'

/**
 * Create a ready-to-use engine. Convenience wrapper.
 * Calls init() + load() in sequence.
 */
export async function createSoulEngine(
  url: string,
  config: { embedDim: number; hiddenDim: number; vocabSize: number; numLayers: number },
): Promise<SoulNetWebGPU> {
  const engine = new SoulNetWebGPU()
  await engine.init()
  await engine.load(url, config)
  return engine
}

/**
 * Create a worker-backed engine that runs inference off the main thread.
 */
export function createSoulEngineWorker(): SoulEngineWorker {
  return new SoulEngineWorker()
}

interface SoulNetGenerator {
  [Symbol.asyncIterator](): SoulNetGenerator
  next(): Promise<IteratorResult<string>>
  return(): Promise<IteratorResult<string>>
  throw(): Promise<IteratorResult<string>>
}

export class SoulEngineWorker {
  private worker: Worker | null = null
  private resolvers = new Map<string, (data: unknown) => void>()
  private msgId = 0

  /** Start the worker and wait for ready. */
  async init(): Promise<void> {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    return new Promise((resolve) => {
      this.worker!.onmessage = (e: MessageEvent) => {
        if (e.data.type === 'ready') { resolve(); return }
        const { type, ...data } = e.data
        const r = this.resolvers.get(type)
        if (r) { r(data); this.resolvers.delete(type) }
      }
      this.worker!.postMessage({ type: 'init' })
    })
  }

  private _send(type: string, payload: Record<string, unknown> = {}): Promise<unknown> {
    return new Promise((resolve) => {
      this.resolvers.set(type, resolve)
      this.worker!.postMessage({ type, ...payload })
    })
  }

  load(url: string, config: SoulNetConfig): Promise<unknown> {
    return this._send('loaded', { url, config })
  }

  generate(prompt: string, maxTokens: number, temperature = 1.0): SoulNetGenerator {
    const worker = this.worker!
    const resolvers: ((value: IteratorResult<string>) => void)[] = []
    worker.postMessage({ type: 'generate', prompt, maxTokens, temperature })

    worker.onmessage = (e: MessageEvent) => {
      if (e.data.type === 'token') {
        const r = resolvers.shift()
        r?.({ value: e.data.token, done: false })
      } else if (e.data.type === 'done') {
        const r = resolvers.shift()
        r?.({ value: undefined, done: true })
      }
    }

    return {
      [Symbol.asyncIterator]() { return this },
      next() {
        return new Promise<IteratorResult<string>>((resolve) => { resolvers.push(resolve) })
      },
      return() {
        return Promise.resolve<IteratorResult<string>>({ value: undefined, done: true })
      },
      throw() {
        return Promise.resolve<IteratorResult<string>>({ value: undefined, done: true })
      },
    }
  }

  reset(): Promise<unknown> {
    return this._send('reset-done', {})
  }

  destroy(): void {
    this.worker?.terminate()
    this.worker = null
  }
}
