/**
 * Generate Controller — axios-based API for generation endpoints.
 *
 * Usage:
 *   import { generateController } from '@/lib/generate-controller'
 *   const result = await generateController.generate({ prompt: 'Hello' })
 */

import { apiPost, streamSSE } from './http-client'

export interface GenerateRequest {
  prompt: string
  max_new_tokens?: number
  temperature?: number
  top_p?: number
  top_k?: number
  repetition_penalty?: number
  model?: string
}

export interface GenerateResponse {
  text: string
  model?: string
  tokens_generated?: number
}

export const generateController = {
  async generate(req: GenerateRequest): Promise<GenerateResponse> {
    const result = await apiPost<GenerateResponse & { error?: string }>('/inference/generate', req)
    if (result.error) throw new Error(result.error)
    return result as GenerateResponse
  },

  async generateStream(
    req: GenerateRequest,
    onToken: (token: string) => void,
    onDone: () => void,
    onError?: (error: string) => void,
  ): Promise<void> {
    let errored = false
    try {
      for await (const event of streamSSE('/inference/generate/stream', { body: req })) {
        if (event.status === 'error') {
          errored = true
          onError?.(event.message || 'Generation error')
          break
        }
        if (event.data?.token) onToken(event.data.token as string)
        if (event.status === 'complete') break
      }
    } catch (err) {
      errored = true
      onError?.(err instanceof Error ? err.message : 'Connection error')
    }
    if (!errored) onDone()
  },
}
