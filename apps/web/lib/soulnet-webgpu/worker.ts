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

let engine: SoulNetWebGPU | null = null

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data
  try {
    switch (msg.type) {
      case 'init': {
        engine = new SoulNetWebGPU()
        await engine.init()
        self.postMessage({ type: 'ready' })
        break
      }
      case 'load': {
        if (!engine) throw new Error('Not initialized')
        const cp = await engine.load(msg.url, msg.config)
        self.postMessage({ type: 'loaded', metadata: cp.metadata })
        break
      }
      case 'generate': {
        if (!engine) throw new Error('Not initialized')
        const gen = engine.generate(msg.prompt, msg.maxTokens, msg.temperature)
        for await (const token of gen) {
          self.postMessage({ type: 'token', token })
        }
        self.postMessage({ type: 'done' })
        break
      }
      case 'reset': {
        engine?.resetState()
        self.postMessage({ type: 'reset-done' })
        break
      }
      default:
        throw new Error(`Unknown message type: ${msg.type}`)
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: (err as Error).message })
  }
}
