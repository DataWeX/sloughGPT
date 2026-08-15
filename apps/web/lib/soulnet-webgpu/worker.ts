/**
 * SoulNetWebGPU Worker — runs inference off the main thread.
 *
 * Messages:
 *   { type: 'init' } → { type: 'ready' }
 *   { type: 'load', url: string, config: SoulNetConfig } → { type: 'loaded', metadata }
 *   { type: 'generate', prompt: string, maxTokens: number, temperature: number }
 *     → { type: 'token', token: string } (multiple)
 *     → { type: 'done' } / { type: 'error', message: string }
 *   { type: 'reset' } → { type: 'reset-done' }
 */

import { SoulNetWebGPU, SoulNetConfig } from './engine'

export interface WorkerScope {
  postMessage(message: unknown): void
  onmessage: ((e: MessageEvent) => void) | null
}

/** Build a message handler that owns its own engine state. */
export function createWorkerHandler(workerScope: WorkerScope) {
  let engine: SoulNetWebGPU | null = null
  return async (e: MessageEvent): Promise<void> => {
    const msg = e.data
    try {
      switch (msg.type) {
        case 'init': {
          engine = new SoulNetWebGPU()
          await engine.init()
          workerScope.postMessage({ type: 'ready' })
          break
        }
        case 'load': {
          if (!engine) throw new Error('Not initialized')
          const cp = await engine.load(msg.url, msg.config)
          workerScope.postMessage({ type: 'loaded', metadata: cp.metadata })
          break
        }
        case 'generate': {
          if (!engine) throw new Error('Not initialized')
          const gen = engine.generate(msg.prompt, msg.maxTokens, msg.temperature)
          for await (const token of gen) {
            workerScope.postMessage({ type: 'token', token })
          }
          workerScope.postMessage({ type: 'done' })
          break
        }
        case 'reset': {
          engine?.resetState()
          workerScope.postMessage({ type: 'reset-done' })
          break
        }
        default:
          throw new Error(`Unknown message type: ${msg.type}`)
      }
    } catch (err) {
      workerScope.postMessage({ type: 'error', message: (err as Error).message })
    }
  }
}

/** Install the message handler on a worker scope; no-op outside worker contexts. */
export function registerWorker(workerScope: WorkerScope = self): void {
  workerScope.onmessage = createWorkerHandler(workerScope)
}

if (typeof self !== 'undefined') {
  registerWorker()
}
