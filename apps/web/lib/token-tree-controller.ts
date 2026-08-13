/**
 * Token Tree Controller — axios-based API for the token-tree semantic queries.
 *
 * The token tree is a BPE merge tree with learned embeddings stored as
 * pugqeep cluster points; `similar()` ranks nearest neighbors by generated
 * embedding cosine similarity.
 *
 * Usage:
 *   import { tokenTreeController } from '@/lib/token-tree-controller'
 *   const neighbors = await tokenTreeController.similar('quick', 5)
 */

import { apiGet, apiPost, apiDelete } from './http-client'

export interface TokenTreeStats {
  trained: boolean
  vocab_size: number
  num_merges: number
  num_base_tokens: number
  embedding_points: number
  embedding_compression_ratio: number
  embed_dim: number
  library?: Record<string, unknown>
}

export interface Neighbor {
  id: number
  token: string
  score: number
}

export interface SimilarResult {
  query: string
  neighbors: Neighbor[]
}

export interface EmbeddingInfo {
  token: string
  id: number
  dim: number
  norm: number
  top: [number, number][]
  embedding_points: number
  compression_ratio: number
}

export interface MatrixEnergyToken {
  token: string
  id: number
  norm: number
}

export interface MatrixSummary {
  matrix: [number, number] | null
  norm_min: number
  norm_mean: number
  norm_max: number
  dead_tokens: number
  live_tokens: number
  most_energetic: [string, number, number][]
  least_energetic: [string, number, number][]
}

export interface TrainTreeResult {
  status: string
  vocab_size: number
  embedding_points: number
  embedding_compression_ratio: number
  embed_dim: number
}

export interface EncodeResult {
  tokens: string[]
  ids: number[]
}

export interface PathStep {
  remaining: string
  token: string
  id: number
  consumed: number
}

export interface PathResult {
  steps: PathStep[]
  ids: number[]
}

export interface LineageResult {
  token: string
  leaves: string[]
  tree: string
}

export interface MergeRule {
  rank: number
  left: string
  right: string
  token: string
  count: number
}

export interface VocabEntry {
  id: number
  token: string
  freq: number
  is_special: boolean
  is_merged: boolean
}

export interface VocabPage {
  total: number
  entries: VocabEntry[]
}

export interface SavedTree {
  name: string
  path: string
  vocab_size: number
  num_merges: number
  trained: boolean
  saved_at: number | null
}

export interface CompareSide {
  name: string
  stats: TokenTreeStats
  vocab: Record<string, number>
}

export interface CompareResult {
  a: CompareSide
  b: CompareSide
  shared_tokens: number
  only_a_tokens: number
  only_b_tokens: number
  shared_merges: number
  only_a_merges: number
  only_b_merges: number
  shared_examples: [string, number][]
  only_a_examples: [string, number][]
  only_b_examples: [string, number][]
}

export const tokenTreeController = {
  async getStats(): Promise<TokenTreeStats> {
    return apiGet<TokenTreeStats>('/token-tree/stats')
  },

  async getVocab(limit = 50, offset = 0): Promise<VocabPage> {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
    return apiGet<VocabPage>(`/token-tree/vocab?${params.toString()}`)
  },

  async getMerges(top_n = 20, query = ''): Promise<MergeRule[]> {
    const params = new URLSearchParams({ top_n: String(top_n) })
    if (query) params.set('query', query)
    return apiGet<MergeRule[]>(`/token-tree/merges?${params.toString()}`)
  },

  async listSaved(): Promise<SavedTree[]> {
    const data = await apiGet<{ trees: SavedTree[] }>('/token-tree/saved')
    return data.trees
  },

  async saveTree(name: string): Promise<SavedTree> {
    return apiPost<SavedTree>('/token-tree/save', { name })
  },

  async loadTree(name: string): Promise<SavedTree> {
    return apiPost<SavedTree>('/token-tree/load', { name })
  },

  async deleteSavedTree(name: string): Promise<{ deleted: boolean }> {
    return apiDelete<{ deleted: boolean }>(`/token-tree/saved/${encodeURIComponent(name)}`)
  },

  async train(params: {
    texts?: string[]
    vocab_size?: number
    embed_dim?: number
    min_frequency?: number
  }): Promise<TrainTreeResult> {
    return apiPost<TrainTreeResult>('/token-tree/train', params)
  },

  async similar(token: string, top_k = 5): Promise<SimilarResult> {
    return apiPost<SimilarResult>('/token-tree/similar', { token, top_k })
  },

  async getEmbedding(token: string, top_k = 8): Promise<EmbeddingInfo> {
    return apiPost<EmbeddingInfo>('/token-tree/embedding', { token, top_k })
  },

  async encode(text: string): Promise<EncodeResult> {
    return apiPost<EncodeResult>('/token-tree/encode', { text })
  },

  async path(text: string): Promise<PathResult> {
    return apiPost<PathResult>('/token-tree/path', { text })
  },

  async decode(ids: number[]): Promise<{ text: string }> {
    return apiPost<{ text: string }>('/token-tree/decode', { ids })
  },

  async lineage(token: string): Promise<LineageResult> {
    return apiPost<LineageResult>('/token-tree/lineage', { token })
  },

  async getMatrixSummary(top_k = 8): Promise<MatrixSummary> {
    const params = new URLSearchParams({ top_k: String(top_k) })
    return apiGet<MatrixSummary>(`/token-tree/matrix?${params.toString()}`)
  },

  async compare(a: string, b: string, top_k = 10): Promise<CompareResult> {
    return apiPost<CompareResult>('/token-tree/compare', { a, b, top_k })
  },
}
