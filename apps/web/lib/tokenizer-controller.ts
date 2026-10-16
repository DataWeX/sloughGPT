/**
 * Tokenizer Controller — axios-based API for tokenizer management.
 *
 * Usage:
 *   import { tokenizerController } from '@/lib/tokenizer-controller'
 *   const stats = await tokenizerController.getStats()
 *   const result = await tokenizerController.tokenize('hello world')
 */

import { apiGet, apiPost } from './http-client'

/** Labels which tokenizer produced a result — a checkpoint's own, or the
 * process-global singleton. Present on every tokenizer response so the UI can
 * never present singleton output as if it were a model's own. */
export interface TokenizerSource {
  kind: 'checkpoint' | 'singleton'
  checkpoint?: string
  path?: string
  variant?: string | null
  origin?: string | null
}

/** Round-trip honesty: does decode(encode(text)) == text (tokenize), and does
 * re-encoding the decoded text reproduce the original ids (detokenize)? */
export interface TokenizerFidelity {
  round_trip?: boolean
  reencode_matches?: boolean
}

export interface TokenizerStats {
  vocab_size: number
  base_chars: number
  merged_subwords: number
  special_tokens: number
  total_merges: number
  trained: boolean
  source?: TokenizerSource
}

export interface TokenizeResult {
  tokens: string[]
  ids: number[]
  source?: TokenizerSource
  fidelity?: TokenizerFidelity
}

export interface DetokenizeResult {
  text: string
  source?: TokenizerSource
  fidelity?: TokenizerFidelity
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
  async getStats(checkpoint?: string): Promise<TokenizerStats> {
    const qs = checkpoint ? `?checkpoint=${encodeURIComponent(checkpoint)}` : ''
    return apiGet<TokenizerStats>(`/tokenizer/stats${qs}`)
  },

  async tokenize(text: string, checkpoint?: string): Promise<TokenizeResult> {
    // checkpoint is only sent when given, so the singleton call shape stays
    // byte-identical to what existing callers/tests expect.
    return apiPost<TokenizeResult>('/tokenizer/tokenize', {
      text,
      ...(checkpoint ? { checkpoint } : {}),
    })
  },

  async detokenize(ids: number[], checkpoint?: string): Promise<DetokenizeResult> {
    return apiPost<DetokenizeResult>('/tokenizer/detokenize', {
      ids,
      ...(checkpoint ? { checkpoint } : {}),
    })
  },

  async getVocab(
    limit = 50,
    offset = 0,
    checkpoint?: string,
  ): Promise<{ entries: VocabEntry[]; total: number; source?: TokenizerSource }> {
    const cp = checkpoint ? `&checkpoint=${encodeURIComponent(checkpoint)}` : ''
    return apiGet(`/tokenizer/vocab?limit=${limit}&offset=${offset}${cp}`)
  },

  async getMerges(
    limit = 30,
    checkpoint?: string,
  ): Promise<{ merges: MergeEntry[]; total: number; source?: TokenizerSource }> {
    const cp = checkpoint ? `&checkpoint=${encodeURIComponent(checkpoint)}` : ''
    return apiGet(`/tokenizer/merges?limit=${limit}${cp}`)
  },

  async getSamples(): Promise<{ samples: SampleWord[] }> {
    return apiGet<{ samples: SampleWord[] }>('/tokenizer/sample')
  },

  /** Checkpoint names the inspector can bind to (for the picker). Thin read
   * of the training listing — the tokenizer feature only needs names. */
  async listCheckpoints(): Promise<string[]> {
    const resp = await apiGet<
      { checkpoints?: Array<{ name?: string }> } | Array<{ name?: string }>
    >('/training/checkpoints')
    const list = Array.isArray(resp) ? resp : (resp?.checkpoints ?? [])
    return list.map((c) => c?.name).filter((n): n is string => Boolean(n))
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

  async train(params: {
    vocab_size?: number
    texts?: string[]
  }): Promise<{ status: string; corpus_size: number; stats: TokenizerStats }> {
    return apiPost('/tokenizer/train', params)
  },
}
