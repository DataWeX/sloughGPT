'use client'

import { apiGet, apiPost } from './http-client'

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

export interface EmbedResponse {
  embedding: number[]
  dimensions: number
  model: string
}

export interface TokenizeResponse {
  tokens: string[]
  ids: number[]
  count: number
}

export interface DetokenizeResponse {
  text: string
  count: number
}

export interface InferHealth {
  status: string
  model_loaded: boolean
  model_id?: string
  engine_type?: string
  has_streaming: boolean
  has_embedding: boolean
}

export interface InferInfo {
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

export const inferController = {
  async generate(req: InferRequest): Promise<InferResponse> {
    return apiPost<InferResponse>('/infer', req)
  },

  async health(): Promise<InferHealth> {
    return apiGet<InferHealth>('/infer/health')
  },

  async info(): Promise<InferInfo> {
    return apiGet<InferInfo>('/infer/info')
  },

  async embed(text: string, model?: string): Promise<EmbedResponse> {
    return apiPost<EmbedResponse>('/infer/embed', { text, model })
  },

  async tokenize(text: string, model?: string): Promise<TokenizeResponse> {
    return apiPost<TokenizeResponse>('/infer/tokenize', { text, model })
  },

  async detokenize(ids: number[], model?: string): Promise<DetokenizeResponse> {
    return apiPost<DetokenizeResponse>('/infer/detokenize', { ids, model })
  },
}
