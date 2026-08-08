/**
 * Tokenizer Controller — axios-based API for tokenizer management.
 *
 * Usage:
 *   import { tokenizerController } from '@/lib/tokenizer-controller'
 *   const stats = await tokenizerController.getStats()
 *   const result = await tokenizerController.tokenize('hello world')
 */

import { apiGet, apiPost } from './http-client'

export interface TokenizerStats {
  vocab_size: number
  base_chars: number
  merged_subwords: number
  special_tokens: number
  total_merges: number
  trained: boolean
}

export interface TokenizeResult {
  tokens: string[]
  ids: number[]
}

export interface VocabEntry {
  id: number
  token: string
  is_special: boolean
}

export interface MergeEntry {
  index: number
  left: string
  right: string
  token: string
}

export interface SampleWord {
  word: string
  ids: number[]
  tokens: string[]
  count: number
}

export const tokenizerController = {
  async getStats(): Promise<TokenizerStats> {
    return apiGet<TokenizerStats>('/tokenizer/stats')
  },

  async tokenize(text: string): Promise<TokenizeResult> {
    return apiPost<TokenizeResult>('/tokenizer/tokenize', { text })
  },

  async detokenize(ids: number[]): Promise<{ text: string }> {
    return apiPost<{ text: string }>('/tokenizer/detokenize', { ids })
  },

  async getVocab(limit = 50, offset = 0): Promise<{ entries: VocabEntry[]; total: number }> {
    return apiGet(`/tokenizer/vocab?limit=${limit}&offset=${offset}`)
  },

  async getMerges(limit = 30): Promise<{ merges: MergeEntry[]; total: number }> {
    return apiGet(`/tokenizer/merges?limit=${limit}`)
  },

  async getSamples(): Promise<{ samples: SampleWord[] }> {
    return apiGet<{ samples: SampleWord[] }>('/tokenizer/sample')
  },

  async pretokenize(text: string): Promise<Record<string, unknown>> {
    return apiPost('/tokenizer/pretokenize', { text })
  },

  async decompose(text: string): Promise<Record<string, unknown>> {
    return apiPost('/tokenizer/decompose', { text })
  },

  async analyze(texts: string[]): Promise<Record<string, unknown>> {
    return apiPost('/tokenizer/analyze', { texts })
  },

  async train(params: { vocab_size?: number; texts?: string[] }): Promise<{ status: string; corpus_size: number; stats: TokenizerStats }> {
    return apiPost('/tokenizer/train', params)
  },
}
