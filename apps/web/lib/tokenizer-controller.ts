/**
 * Tokenizer Controller — tokenizer stats, tokenize/detokenize, vocab, merges.
 *
 * Usage:
 *   import { tokenizerController } from '@/lib/tokenizer-controller'
 *   const stats = await tokenizerController.getStats()
 */

import { apiGet, apiPost } from './http-client'

export interface TokenizerStats {
  vocab_size: number
  base_chars: number
  merged_subwords: number
  special_tokens: number
  total_merges: number
}

export interface TokenizeResult {
  tokens: string[]
  ids: number[]
}

export interface DetokenizeResult {
  text: string
}

export interface VocabEntry {
  id: number
  token: string
  is_special: boolean
}

export interface VocabResponse {
  entries: VocabEntry[]
  total: number
  offset: number
  limit: number
}

export interface MergeResponse {
  merges: string[]
  total: number
}

export interface SampleWord {
  word: string
  ids: number[]
  tokens: string[]
  count: number
}

export interface SampleResponse {
  samples: SampleWord[]
}

export const tokenizerController = {
  async getStats(): Promise<TokenizerStats> {
    return apiGet<TokenizerStats>('/tokenizer/stats')
  },

  async tokenize(text: string): Promise<TokenizeResult> {
    return apiPost<TokenizeResult>('/tokenizer/tokenize', { text })
  },

  async detokenize(ids: number[]): Promise<DetokenizeResult> {
    return apiPost<DetokenizeResult>('/tokenizer/detokenize', { ids })
  },

  async getVocab(limit = 50, offset = 0): Promise<VocabResponse> {
    return apiGet<VocabResponse>(`/tokenizer/vocab?limit=${limit}&offset=${offset}`)
  },

  async getMerges(limit = 30): Promise<MergeResponse> {
    return apiGet<MergeResponse>(`/tokenizer/merges?limit=${limit}`)
  },

  async getSamples(): Promise<SampleResponse> {
    return apiGet<SampleResponse>('/tokenizer/sample')
  },

  async trainShakespeare(vocabSize = 512): Promise<any> {
    return apiPost<any>(`/tokenizer/train-shakespeare?vocab_size=${vocabSize}`)
  },
}
