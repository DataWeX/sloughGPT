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
}
