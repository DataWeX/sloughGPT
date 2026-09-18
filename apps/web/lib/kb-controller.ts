'use client'

import { apiGet, apiPost, apiPut, apiDelete } from './http-client'

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

export interface KnowledgeStats {
  total_items: number
  topics: string[]
  avg_importance: number
  sources: Record<string, number>
}

export interface TopicItem {
  name: string
  count: number
}

export interface KnowledgeGaps {
  gaps: string[]
  suggestions: string[]
}

export const kbController = {
  async list(topic?: string, limit = 50, offset = 0): Promise<KnowledgeItem[]> {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
    if (topic) params.set('topic', topic)
    return apiGet<KnowledgeItem[]>(`/knowledge?${params.toString()}`)
  },

  async add(
    content: string,
    topic = 'general',
    source = 'manual',
    importance = 0.7,
    autoTag = false,
  ): Promise<KnowledgeItem> {
    return apiPost<KnowledgeItem>('/knowledge', {
      content,
      topic,
      source,
      importance,
      auto_tag: autoTag,
    })
  },

  async update(
    id: string,
    data: { content?: string; topic?: string; importance?: number },
  ): Promise<KnowledgeItem> {
    return apiPut<KnowledgeItem>(`/knowledge/${id}`, data)
  },

  async remove(id: string): Promise<{ deleted: boolean }> {
    return apiDelete<{ deleted: boolean }>(`/knowledge/${id}`)
  },

  async batchDelete(ids: string[]): Promise<{ deleted: number }> {
    return apiPost<{ deleted: number }>('/knowledge/batch-delete', { ids })
  },

  async search(query: string, limit = 10): Promise<KnowledgeItem[]> {
    const params = new URLSearchParams({ q: query, limit: String(limit) })
    return apiGet<KnowledgeItem[]>(`/knowledge/search?${params.toString()}`)
  },

  async stats(): Promise<KnowledgeStats> {
    return apiGet<KnowledgeStats>('/knowledge/stats')
  },

  async topics(): Promise<TopicItem[]> {
    return apiGet<TopicItem[]>('/knowledge/topics')
  },

  async ingestUrl(url: string, source = 'direct'): Promise<{ status: string; id?: string }> {
    return apiPost<{ status: string; id?: string }>('/knowledge/ingest-url', { url, source })
  },

  async batchIngest(
    items: { content: string; source?: string; tags?: string[] }[],
  ): Promise<{ ingested: number }> {
    return apiPost<{ ingested: number }>('/knowledge/batch', { items })
  },

  async suggestTopic(content: string): Promise<{ topic: string }> {
    return apiPost<{ topic: string }>('/knowledge/suggest-topic', { content })
  },

  async checkDuplicate(
    content: string,
  ): Promise<{ is_duplicate: boolean; similar?: KnowledgeItem[] }> {
    return apiPost<{ is_duplicate: boolean; similar?: KnowledgeItem[] }>(
      '/knowledge/check-duplicate',
      { content },
    )
  },

  async categorize(id: string, topic: string): Promise<{ updated: boolean }> {
    return apiPost<{ updated: boolean }>(`/knowledge/${id}/categorize`, { topic })
  },

  async gaps(): Promise<KnowledgeGaps> {
    return apiGet<KnowledgeGaps>('/knowledge/gaps')
  },

  async context(query: string, topK = 5): Promise<{ context: string; items: KnowledgeItem[] }> {
    const params = new URLSearchParams({ q: query, top_k: String(topK) })
    return apiGet<{ context: string; items: KnowledgeItem[] }>(
      `/knowledge/context?${params.toString()}`,
    )
  },

  async trainAdapter(topK = 10): Promise<{ status: string; job_id?: string }> {
    return apiPost<{ status: string; job_id?: string }>('/knowledge/train-adapter', { top_k: topK })
  },

  async adapterStatus(): Promise<{ trained: boolean; accuracy?: number; last_trained?: number }> {
    return apiGet<{ trained: boolean; accuracy?: number; last_trained?: number }>(
      '/knowledge/adapter-status',
    )
  },

  async related(id: string, limit = 5): Promise<KnowledgeItem[]> {
    return apiGet<KnowledgeItem[]>(`/knowledge/${id}/related?limit=${limit}`)
  },
}
