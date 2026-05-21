/**
 * Knowledge Controller — axios-based API for knowledge management.
 *
 * Usage:
 *   import { knowledgeController } from '@/lib/knowledge-controller'
 *   const items = await knowledgeController.list()
 *   await knowledgeController.add('some content')
 */

import { apiGet, apiPost, apiDelete } from './http-client'

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
}

export const knowledgeController = {
  async list(): Promise<KnowledgeItem[]> {
    const data = await apiGet<KnowledgeItem[]>('/knowledge')
    return data || []
  },

  async add(content: string, topic = 'general'): Promise<{ status: string; content: string }> {
    return apiPost<{ status: string; content: string }>('/knowledge', { content, topic })
  },

  async delete(id: string): Promise<void> {
    await apiDelete(`/knowledge/${id}`)
  },

  async search(query: string): Promise<KnowledgeSearchResult> {
    return apiGet<KnowledgeSearchResult>(`/knowledge/search?query=${encodeURIComponent(query)}`)
  },

  async batchIngest(items: Array<{ content: string; source?: string; tags?: string[] }>): Promise<{ stored: number }> {
    return apiPost<{ stored: number }>('/knowledge/batch', { items })
  },
}
