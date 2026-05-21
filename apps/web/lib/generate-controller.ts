/**
 * Generate Controller — axios-based API for generation endpoints.
 *
 * Usage:
 *   import { generateController } from '@/lib/generate-controller'
 *   const result = await generateController.generate({ prompt: 'Hello' })
 */

import { apiGet, apiPost } from './http-client'

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
    const { PUBLIC_API_URL } = await import('./config')
    const { useAuthStore } = await import('./auth')

    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    const token = useAuthStore.getState().token
    if (token) headers['Authorization'] = `Bearer ${token}`

    const res = await fetch(`${PUBLIC_API_URL}/inference/generate/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(req),
    })

    if (!res.ok || !res.body) {
      onError?.(`HTTP ${res.status}`)
      onDone()
      return
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let done = false
    while (!done) {
      const { value, done: streamDone } = await reader.read()
      done = streamDone
      if (value) {
        const text = decoder.decode(value, { stream: true })
        for (const line of text.split('\n')) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))
            if (data.status === 'error') {
              onError?.(data.message || 'Generation error')
              done = true
            } else if (data.data?.token) {
              onToken(data.data.token)
            }
            if (data.status === 'complete') {
              done = true
            }
          } catch { /* skip malformed lines */ }
        }
      }
    }
    onDone()
  },
}
