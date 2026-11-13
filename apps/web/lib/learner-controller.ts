/**
 * Learner Controller — axios-based API for continual learning.
 *
 * Usage:
 *   import { learnerController } from '@/lib/learner-controller'
 *   const status = await learnerController.status()
 *   await learnerController.search('machine learning')
 */

import { apiGet, apiPost } from './http-client'

export interface LearnerStatus {
  soul_name: string
  total_tokens_ingested: number
  train_steps_completed: number
  current_loss: number
  loss_history: number[]
  buffer_size: number
  buffer_capacity: number
  pending_tokens: number
  arch: string
  n_embed: number
  n_layer: number
  n_head: number
  vocab_size: number
  knowledge: Record<string, unknown>
  feeds_subscribed: number
  filter_stats: Record<string, unknown>
  filter_config: Record<string, unknown>
}

export interface SearchResult {
  tokens_ingested: number
  new_facts: number
  rejected: number
  filter_stats: Record<string, unknown>
}

export interface KnowledgeResult {
  facts: Array<{ content: string; topic: string; source: string; importance: number }>
  total: number
}

export const learnerController = {
  async status(): Promise<LearnerStatus> {
    return apiGet<LearnerStatus>('/learn/status')
  },

  async search(query: string, maxResults = 5): Promise<SearchResult> {
    const params = new URLSearchParams({ query, max_results: String(maxResults) })
    return apiPost(`/learn/search?${params}`)
  },

  async ingestUrl(url: string): Promise<{ status: string; facts_added: number }> {
    const params = new URLSearchParams({ url })
    return apiPost(`/learn/ingest-url?${params}`)
  },

  async ingestText(text: string): Promise<{ status: string; facts_added: number }> {
    return apiPost('/learn/ingest', { text })
  },

  async queryKnowledge(query?: string, limit = 20): Promise<KnowledgeResult> {
    const params = new URLSearchParams()
    if (query) params.set('query', query)
    params.set('top_k', String(limit))
    return apiGet(`/learn/knowledge?${params}`)
  },

  async subscribeFeed(url: string, interval = 3600): Promise<{ status: string }> {
    const params = new URLSearchParams({ action: 'subscribe', url, poll_interval: String(interval) })
    return apiPost(`/learn/feed?${params}`)
  },

  async unsubscribeFeed(url: string): Promise<{ status: string }> {
    const params = new URLSearchParams({ action: 'unsubscribe', url })
    return apiPost(`/learn/feed?${params}`)
  },

  async listFeeds(): Promise<{ feeds: Array<{ url: string; interval: number; last_poll?: string }> }> {
    const params = new URLSearchParams({ action: 'list' })
    return apiPost(`/learn/feed?${params}`)
  },

  async train(): Promise<{ status: string; loss?: number }> {
    return apiPost('/learn/train')
  },

  async evaluate(): Promise<{ metrics: Record<string, unknown> }> {
    return apiPost('/learn/evaluate')
  },

  async deploy(): Promise<{ status: string }> {
    return apiPost('/learn/deploy')
  },
}
