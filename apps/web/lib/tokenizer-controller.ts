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

export interface MergeEntry {
  index: number
  left: string
  right: string
  token: string
}

export interface MergeResponse {
  merges: MergeEntry[]
  total: number
}

export interface SampleWord {
  word: string
  ids: number[]
  tokens: string[]
  count: number
}

export interface PretokenizeSegment {
  text: string
  char_count: number
  pct: number
}

export interface PretokenizeResponse {
  pretokens: string[]
  segments: PretokenizeSegment[]
  count: number
}

export interface DetokenizeResponse {
  text: string
}

export interface DecomposeResponse {
  token: string
  id: number
  type: string
  merge_path: { left: string; right: string; into: string }[]
  depth: number
  base_chars: string[]
}

export interface AnalysisResponse {
  total_chars: number
  total_tokens: number
  compression_ratio: number
  unique_tokens: number
  vocab_utilization: number
  top_tokens: { id: number; token: string; count: number; pct: number }[]
  rare_tokens: { id: number; token: string; count: number }[]
}

export const tokenizerController = {
  async getStats(): Promise<TokenizerStats> {
    return apiGet<TokenizerStats>('/tokenizer/stats')
  },

  async tokenize(text: string): Promise<TokenizeResult> {
    return apiPost<TokenizeResult>('/tokenizer/tokenize', { text })
  },

  async detokenize(ids: number[]): Promise<DetokenizeResponse> {
    return apiPost<DetokenizeResponse>('/tokenizer/detokenize', { ids })
  },

  async getVocab(limit = 50, offset = 0): Promise<VocabResponse> {
    return apiGet<VocabResponse>(`/tokenizer/vocab?limit=${limit}&offset=${offset}`)
  },

  async getMerges(limit = 30): Promise<MergeResponse> {
    return apiGet<MergeResponse>(`/tokenizer/merges?limit=${limit}`)
  },

  async getSamples(): Promise<{ samples: SampleWord[] }> {
    return apiGet('/tokenizer/sample')
  },

  async pretokenize(text: string): Promise<PretokenizeResponse> {
    return apiPost<PretokenizeResponse>('/tokenizer/pretokenize', { text })
  },

  async decomposeToken(token: string): Promise<DecomposeResponse> {
    return apiPost<DecomposeResponse>('/tokenizer/decompose', { text: token })
  },

  async analyzeCorpus(texts: string[]): Promise<AnalysisResponse> {
    return apiPost<AnalysisResponse>('/tokenizer/analyze', { texts })
  },

  async trainTokenizer(vocabSize = 512, texts?: string[]): Promise<any> {
    const body: Record<string, unknown> = { vocab_size: vocabSize }
    if (texts) body.texts = texts
    return apiPost<any>('/tokenizer/train', body)
  },

  /** @deprecated Use trainTokenizer() */
  async train(vocabSize = 512, texts?: string[]): Promise<any> {
    return apiPost<any>('/tokenizer/train', { vocab_size: vocabSize, texts })
  },
}
