/**
 * Infer Controller — unified client for /infer API endpoints.
 *
 * Language-agnostic inference layer: generate, stream, embed, tokenize, health.
 *
 * Usage:
 *   import { inferController } from '@/lib/infer-controller'
 *
 *   // Generate
 *   const result = await inferController.generate({ prompt: 'Hello' })
 *
 *   // Stream
 *   await inferController.generateStream({ prompt: 'Tell me a joke' }, (token) => {
 *     process.stdout.write(token)
 *   })
 *
 *   // Embed
 *   const vec = await inferController.embed({ text: 'hello world' })
 *
 *   // Health
 *   const health = await inferController.health()
 */

import { apiPost, apiGet } from './http-client'

// --- Types ---

export interface InferRequest {
  prompt: string
  max_new_tokens?: number
  temperature?: number
  top_p?: number
  top_k?: number
  repetition_penalty?: number
  model?: string
}

export interface InferResponse {
  text: string
  model: string
  tokens_generated: number
  elapsed_ms: number
}

export interface EmbedRequest {
  text: string
  model?: string
}

export interface EmbedResponse {
  embedding: number[]
  dimensions: number
  model: string
}

export interface TokenizeRequest {
  text: string
  model?: string
}

export interface TokenizeResponse {
  tokens: string[]
  ids: number[]
  count: number
}

export interface DetokenizeRequest {
  ids: number[]
  model?: string
}

export interface DetokenizeResponse {
  text: string
  count: number
}

export interface InferHealthResponse {
  status: string
  model_loaded: boolean
  model_id: string | null
  engine_type: string | null
  has_streaming: boolean
  has_embedding: boolean
}

export interface InferInfoResponse {
  model_id: string
  model_type: string
  num_parameters: number
  vocab_size: number
  max_context: number
  num_layers: number
  has_tokenizer: boolean
  has_streaming: boolean
  has_embedding: boolean
  extra: Record<string, unknown>
}

// --- Controller ---

export const inferController = {
  /**
   * Generate text from a prompt (non-streaming).
   */
  async generate(req: InferRequest): Promise<InferResponse> {
    const result = await apiPost<InferResponse>('/infer', req)
    return result as InferResponse
  },

  /**
   * Stream generated tokens via SSE.
   *
   * Calls onToken for each token, onDone when complete, onError on failure.
   */
  async generateStream(
    req: InferRequest,
    onToken: (token: string) => void,
    onDone: (meta?: { tokens?: number; elapsed_ms?: number }) => void,
    onError?: (error: string) => void,
  ): Promise<void> {
    const { PUBLIC_API_URL } = await import('./config')
    const { useAuthStore } = await import('./auth')

    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    const token = useAuthStore.getState().token
    if (token) headers['Authorization'] = `Bearer ${token}`

    const res = await fetch(`${PUBLIC_API_URL}/infer/stream`, {
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
              onError?.(data.data?.error || data.message || 'Generation error')
              done = true
            } else if (data.data?.token) {
              onToken(data.data.token)
            }
            if (data.status === 'complete') {
              done = true
              onDone(data.meta)
            }
          } catch { /* skip malformed lines */ }
        }
      }
    }
    if (!done) onDone()
  },

  /**
   * Get embedding vector for text.
   */
  async embed(req: EmbedRequest): Promise<EmbedResponse> {
    const result = await apiPost<EmbedResponse>('/infer/embed', req)
    return result as EmbedResponse
  },

  /**
   * Tokenize text into token IDs and strings.
   */
  async tokenize(req: TokenizeRequest): Promise<TokenizeResponse> {
    const result = await apiPost<TokenizeResponse>('/infer/tokenize', req)
    return result as TokenizeResponse
  },

  /**
   * Convert token IDs back to text.
   */
  async detokenize(req: DetokenizeRequest): Promise<DetokenizeResponse> {
    const result = await apiPost<DetokenizeResponse>('/infer/detokenize', req)
    return result as DetokenizeResponse
  },

  /**
   * Get engine health status.
   */
  async health(): Promise<InferHealthResponse> {
    const result = await apiGet<InferHealthResponse>('/infer/health')
    return result as InferHealthResponse
  },

  /**
   * Get loaded model metadata.
   */
  async info(): Promise<InferInfoResponse> {
    const result = await apiGet<InferInfoResponse>('/infer/info')
    return result as InferInfoResponse
  },

  /**
   * Check if the engine is ready (model loaded and healthy).
   */
  async isReady(): Promise<boolean> {
    try {
      const h = await this.health()
      return h.model_loaded && h.status === 'ready'
    } catch {
      return false
    }
  },
}
