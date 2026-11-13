/**
 * Memory Controller — axios-based API for the auto-memory layer.
 *
 * The chat loop writes to this store automatically. These methods let the
 * Knowledge page inspect, search, manually store, and clear what the AI
 * remembers across conversations.
 *
 * Usage:
 *   import { memoryController } from '@/lib/memory-controller'
 *   const stats = await memoryController.stats()
 *   const items = await memoryController.list()
 *   await memoryController.clear()
 */

import { apiGet, apiPost, apiDelete, apiPatch } from './http-client'

export interface MemoryItem {
  id: string
  content: string
  topic: string
  source: string
  url: string
  timestamp: number
  importance: number
  score: number
}

export interface MemoryStats {
  enabled: boolean
  total_facts: number
  topics: number
  visited_urls: number
}

export interface MemoryListResponse {
  items: MemoryItem[]
  total: number
}

export interface MemorySearchResponse {
  results: MemoryItem[]
  total: number
}

export interface MemoryStoreResult {
  stored: boolean
  content: string
  topic: string
  source: string
}

export interface MemoryRememberResult {
  stored: boolean
  reason: string
}

export interface MemoryConfigResult {
  enabled: boolean
  min_chars: number
  max_facts: number
  store_path: string
  sync_remember: boolean
  consolidation_threshold: number
  maintenance_interval_minutes: number
  archive_retention_days: number
}

export interface MemoryConfigUpdate {
  enabled?: boolean
  archive_retention_days?: number
}

export interface MemoryClearResult {
  cleared: number
}

export interface MemoryDeleteResult {
  deleted: number
}

export interface MemoryUpdateResult {
  updated: number
  duplicate: boolean
}

export interface MemoryConsolidateResult {
  removed: number
  kept: number
  threshold: number
}

export interface MemoryArchiveRecord {
  ts: number
  task_id: string
  task_type: string
  [key: string]: unknown
}

export interface MemoryArchiveStats {
  path: string
  records: number
  bytes: number
  task_types: Record<string, number>
  oldest_ts: number | null
  newest_ts: number | null
}

export interface MemoryArchiveResponse {
  records: MemoryArchiveRecord[]
  total: number
}

export interface MemoryPruneResult {
  pruned: number
}

export const memoryController = {
  async list(limit = 50): Promise<MemoryListResponse> {
    const data = await apiGet<MemoryListResponse>(`/memory/list?limit=${limit}`)
    return data || { items: [], total: 0 }
  },

  async stats(): Promise<MemoryStats | null> {
    const data = await apiGet<MemoryStats>('/memory/stats')
    return data || null
  },

  async search(query: string, limit = 5): Promise<MemorySearchResponse> {
    const data = await apiGet<MemorySearchResponse>(`/memory/search?q=${encodeURIComponent(query)}&limit=${limit}`)
    return data || { results: [], total: 0 }
  },

  async store(content: string, topic = 'manual', source = 'api'): Promise<MemoryStoreResult> {
    return apiPost('/memory/store', { content, topic, source })
  },

  async remember(userMessage: string, assistantResponse: string): Promise<MemoryRememberResult> {
    return apiPost('/memory/remember', { user_message: userMessage, assistant_response: assistantResponse })
  },

  async setEnabled(enabled: boolean): Promise<MemoryConfigResult> {
    return apiPost('/memory/config', { enabled })
  },

  async getConfig(): Promise<MemoryConfigResult> {
    return apiGet<MemoryConfigResult>('/memory/config')
  },

  async updateConfig(update: MemoryConfigUpdate): Promise<MemoryConfigResult> {
    return apiPost('/memory/config', update)
  },

  async clear(): Promise<MemoryClearResult> {
    return apiPost('/memory/clear')
  },

  async delete(id: string): Promise<MemoryDeleteResult> {
    return apiDelete(`/memory/${encodeURIComponent(id)}`)
  },

  async update(id: string, content: string, topic?: string, importance?: number): Promise<MemoryUpdateResult> {
    const body: { content: string; topic?: string; importance?: number } = {
      content,
      topic: typeof topic === 'string' && topic.trim() ? topic.trim() : undefined,
    }
    if (typeof importance === 'number' && Number.isFinite(importance)) {
      body.importance = importance
    }
    return apiPatch<MemoryUpdateResult>(`/memory/${encodeURIComponent(id)}`, body)
  },

  async consolidate(threshold?: number): Promise<MemoryConsolidateResult> {
    const query = typeof threshold === 'number' ? `?threshold=${threshold}` : ''
    return apiPost(`/memory/consolidate${query}`)
  },

  async archive(limit = 20): Promise<MemoryArchiveResponse> {
    const data = await apiGet<MemoryArchiveResponse>(`/memory/archive?limit=${limit}`)
    return data || { records: [], total: 0 }
  },

  async archiveStats(): Promise<MemoryArchiveStats | null> {
    const data = await apiGet<MemoryArchiveStats>('/memory/archive/stats')
    return data || null
  },

  async archivePrune(retainDays?: number): Promise<MemoryPruneResult> {
    const query = typeof retainDays === 'number' ? `?retain_days=${retainDays}` : ''
    return apiPost(`/memory/archive/prune${query}`)
  },
}
