/**
 * SoulNetWebGPU Worker — runs inference off the main thread.
 *
 * Supports both LSTM and Transformer architectures via auto-detection.
 *
 * Messages:
 *   { type: 'init' } → { type: 'ready' }
 *   { type: 'load', url: string, config?: SoulNetConfig } → { type: 'loaded', metadata }
 *   { type: 'generate', prompt: string, maxTokens: number, temperature: number }
 *     → { type: 'token', token: string } (multiple)
 *     → { type: 'done' } / { type: 'error', message: string }
 *   { type: 'reset' } → { type: 'reset-done' }
 */

import { SoulNetWebGPU, SoulNetConfig } from './engine'
import { SoulTransformerWebGPU } from './transformer-engine'
import { inferArch } from './weights'
import type { SoulTransformerArch } from './weights'

export interface WorkerScope {
  postMessage(message: unknown): void
  onmessage: ((e: MessageEvent) => void) | null
}

type Engine = SoulNetWebGPU | SoulTransformerWebGPU

/** Build a message handler that owns its own engine state. */
export function createWorkerHandler(workerScope: WorkerScope) {
  let engine: Engine | null = null
  let initialized = false

  return async (e: MessageEvent): Promise<void> => {
    const msg = e.data
    try {
      switch (msg.type) {
        case 'init': {
          initialized = true
          workerScope.postMessage({ type: 'ready' })
          break
        }
        case 'load': {
          if (!initialized) throw new Error('Not initialized')

          // Fetch the buffer
          let raw: ArrayBuffer
          if (msg.url) {
            const resp = await fetch(msg.url)
            if (!resp.ok) throw new Error(`HTTP ${resp.status} from ${msg.url}`)
            raw = await resp.arrayBuffer()
          } else if (msg.buffer) {
            raw = msg.buffer
          } else {
            throw new Error('No url or buffer provided')
          }

          // Detect architecture
          const arch = inferArch(raw)

          if (arch.archType === 'transformer') {
            const t = arch as SoulTransformerArch
            const e = new SoulTransformerWebGPU()
            await e.init()
            await e.load(raw, t)
            engine = e
          } else {
            const e = new SoulNetWebGPU()
            await e.init()
            await e.load(raw, {
              embedDim: arch.embedDim,
              hiddenDim: arch.hiddenDim,
              vocabSize: arch.vocabSize,
              numLayers: arch.numLayers,
              ...(msg.config || {}),
            })
            engine = e
          }

          workerScope.postMessage({ type: 'loaded', metadata: engine!.metadata })
          break
        }
        case 'generate': {
          if (!engine) throw new Error('Not initialized')
          const gen = engine.generate(msg.prompt, msg.maxTokens, msg.temperature, msg.eosToken ?? 0)
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
