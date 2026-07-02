/**
 * Knowledge Controller — axios-based API for knowledge management.
 *
 * Usage:
 *   import { knowledgeController } from '@/lib/knowledge-controller'
 *   const items = await knowledgeController.list()
 *   await knowledgeController.add('some content')
 */

import { apiGet, apiPost, apiPatch, apiDelete } from './http-client'

export interface KnowledgeItem {
  id: string
  content: string
  topic: string
  source: string
  url: string
  timestamp: number
  importance: number
  score: number
}

export interface KnowledgeSearchResult {
  results: KnowledgeItem[]
  count?: number
}

export interface KnowledgeStats {
  total_items: number
  topics: Record<string, number>
  topic_count: number
  sources: Record<string, number>
  avg_importance: number
  searchable: boolean
}

export interface TopicCount {
  name: string
  count: number
}

export interface TopicsResponse {
  topics: TopicCount[]
  total: number
}

export interface AdapterStatus {
  adapter_exists: boolean
  fact_count: number
  total_facts_available: number
  trained_at?: number
  post_training_loss?: number
  epochs?: number
  lora_rank?: number
}

export interface IngestResult {
  status: string
  new_facts: number
  title: string
  content_length: number
  rejected: boolean
  reason?: string
}

export const knowledgeController = {
  async list(limit = 200, offset = 0): Promise<KnowledgeItem[]> {
    const data = await apiGet<KnowledgeItem[]>(`/knowledge?limit=${limit}&offset=${offset}`)
    return data || []
  },

  async add(content: string, topic = 'general', autoTag = false): Promise<{ status: string; content: string; topic?: string }> {
    return apiPost('/knowledge', { content, topic, auto_tag: autoTag })
  },

  async update(id: string, updates: { content?: string; topic?: string; importance?: number }): Promise<{ status: string }> {
    return apiPatch(`/knowledge/${id}`, updates)
  },

  async delete(id: string): Promise<void> {
    await apiDelete(`/knowledge/${id}`)
  },

  async search(query: string): Promise<KnowledgeSearchResult> {
    return apiGet(`/knowledge/search?query=${encodeURIComponent(query)}`)
  },

  async batchIngest(items: Array<{ content: string; source?: string; tags?: string[] }>): Promise<{ stored: number }> {
    return apiPost('/knowledge/batch', { items })
  },

  async stats(): Promise<KnowledgeStats> {
    return apiGet('/knowledge/stats')
  },

  async topics(): Promise<TopicsResponse> {
    return apiGet('/knowledge/topics')
  },

  async related(id: string, topK = 6): Promise<{ items: KnowledgeItem[]; count: number }> {
    return apiGet(`/knowledge/${id}/related?top_k=${topK}`)
  },

  async batchDelete(ids: string[]): Promise<{ deleted: number }> {
    return apiPost('/knowledge/batch-delete', { ids })
  },

  async suggestTopic(content: string): Promise<{ topic: string; confidence: string }> {
    return apiPost('/knowledge/suggest-topic', { content })
  },

  async trainAdapter(): Promise<{ status: string; fact_count: number; elapsed: number; adapter_status: AdapterStatus }> {
    return apiPost('/knowledge/train-adapter')
  },

  async getAdapterStatus(): Promise<AdapterStatus> {
    return apiGet('/knowledge/adapter-status')
  },

  async context(): Promise<{ context: string; count: number }> {
    return apiGet('/knowledge/context')
  },

  async ingestUrl(url: string): Promise<IngestResult> {
    return apiPost('/knowledge/ingest-url', { url })
  },

  async ingestFile(file: File, topic = 'imported', chunkSize = 500, overlap = 50): Promise<{
    status: string
    stored: number
    total_chunks: number
    topic: string
    filename: string
    file_size: number
  }> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('topic', topic)
    formData.append('chunk_size', String(chunkSize))
    formData.append('overlap', String(overlap))
    const resp = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/knowledge/ingest-file`, {
      method: 'POST',
      body: formData,
    })
    if (!resp.ok) throw new Error(`Upload failed: ${resp.statusText}`)
    return resp.json()
  },

  // ── Practical knowledge operations ──

  async searchFiles(query: string, path = '.', extensions?: string[], topK = 10): Promise<{
    results: Array<{ path: string; line: number; snippet: string; score: number }>
    indexed_files: number
    indexed_chunks: number
  }> {
    return apiPost('/knowledge/search-files', { query, path, extensions, top_k: topK })
  },

  async checkDuplicate(content: string, threshold = 0.85): Promise<{
    is_duplicate: boolean
    best_match: string | null
    score: number
    threshold: number
  }> {
    return apiPost('/knowledge/check-duplicate', { content, threshold })
  },

  async categorize(content: string): Promise<{
    topic: string
    suggestions: Array<{ topic: string; score: number }>
  }> {
    return apiPost('/knowledge/categorize', { content })
  },

  async gaps(): Promise<{
    gaps: Array<{ topic: string; suggestion: string }>
    total_facts: number
    topics: string[]
  }> {
    return apiGet('/knowledge/gaps')
  },

  async bulkIngest(items: string[], topic = 'imported', source = 'bulk', dedupThreshold = 0.85): Promise<{
    status: string
    added: number
    skipped: number
    errors: number
  }> {
    return apiPost('/knowledge/bulk-ingest', { items, topic, source, dedup_threshold: dedupThreshold })
  },
}
