'use client'

import { apiGet, apiPost } from './http-client'

export interface VectorStats {
  provider: string
  count: number
}

export interface VectorSearchResult {
  text: string
  score: number
  id: string
}

export const vectorController = {
  async getStats(): Promise<VectorStats> {
    return apiGet<VectorStats>('/vector/stats')
  },

  async init(provider = 'in_memory', dimension = 384): Promise<{ status: string; provider: string; note?: string }> {
    return apiPost('/vector/init', { provider, dimension })
  },

  async upsert(texts: string[], ids?: string[], metadata?: Record<string, unknown>[]): Promise<{ status: string; count: number; elapsed_ms: number }> {
    return apiPost('/vector/upsert', { texts, ids, metadata })
  },

  async search(query: string, topK = 5): Promise<{ results: VectorSearchResult[]; elapsed_ms: number }> {
    return apiPost('/vector/search', { query, top_k: topK })
  },

  async ingestStatus(): Promise<{ status: string }> {
    return apiGet('/vector/ingest/status')
  },
}
