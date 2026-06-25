import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { knowledgeController } from './knowledge-controller'

describe('knowledgeController', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('list GETs /knowledge with limit and offset', async () => {
    apiClient.apiGet.mockResolvedValue([{ id: 'k1', content: 'fact 1' }])
    const result = await knowledgeController.list(10, 5)
    expect(result).toHaveLength(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/knowledge?limit=10&offset=5')
  })

  it('list returns [] when null', async () => {
    apiClient.apiGet.mockResolvedValue(null)
    const result = await knowledgeController.list()
    expect(result).toEqual([])
  })

  it('add POSTs /knowledge with content', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', content: 'hello' })
    const result = await knowledgeController.add('hello', 'general')
    expect(result.status).toBe('ok')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge', { content: 'hello', topic: 'general', auto_tag: false })
  })

  it('add supports autoTag', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', content: 'hello', topic: 'auto-tag' })
    const result = await knowledgeController.add('hello', 'general', true)
    expect(result.topic).toBe('auto-tag')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge', { content: 'hello', topic: 'general', auto_tag: true })
  })

  it('update PATCHes /knowledge/{id}', async () => {
    apiClient.apiPatch.mockResolvedValue({ status: 'updated' })
    const result = await knowledgeController.update('k1', { content: 'new' })
    expect(apiClient.apiPatch).toHaveBeenCalledWith('/knowledge/k1', { content: 'new' })
    expect(result.status).toBe('updated')
  })

  it('delete DELETEs /knowledge/{id}', async () => {
    apiClient.apiDelete.mockResolvedValue(undefined)
    await knowledgeController.delete('k1')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/knowledge/k1')
  })

  it('search GETs /knowledge/search with encoded query', async () => {
    apiClient.apiGet.mockResolvedValue({ results: [{ id: 'k1' }] })
    const result = await knowledgeController.search('hello world')
    expect(result.results).toHaveLength(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/knowledge/search?query=hello%20world')
  })

  it('batchIngest POSTs /knowledge/batch', async () => {
    apiClient.apiPost.mockResolvedValue({ stored: 3 })
    const result = await knowledgeController.batchIngest([{ content: 'a' }, { content: 'b' }])
    expect(result.stored).toBe(3)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge/batch', { items: [{ content: 'a' }, { content: 'b' }] })
  })

  it('stats GETs /knowledge/stats', async () => {
    apiClient.apiGet.mockResolvedValue({ total_items: 10, topics: {}, topic_count: 0, sources: {}, avg_importance: 0, searchable: true })
    const result = await knowledgeController.stats()
    expect(result.total_items).toBe(10)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/knowledge/stats')
  })

  it('topics GETs /knowledge/topics', async () => {
    apiClient.apiGet.mockResolvedValue({ topics: [], total: 0 })
    const result = await knowledgeController.topics()
    expect(result.total).toBe(0)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/knowledge/topics')
  })

  it('related GETs /knowledge/{id}/related', async () => {
    apiClient.apiGet.mockResolvedValue({ items: [], count: 0 })
    const result = await knowledgeController.related('k1', 6)
    expect(result.count).toBe(0)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/knowledge/k1/related?top_k=6')
  })

  it('batchDelete POSTs /knowledge/batch-delete', async () => {
    apiClient.apiPost.mockResolvedValue({ deleted: 2 })
    const result = await knowledgeController.batchDelete(['k1', 'k2'])
    expect(result.deleted).toBe(2)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge/batch-delete', { ids: ['k1', 'k2'] })
  })

  it('suggestTopic POSTs /knowledge/suggest-topic', async () => {
    apiClient.apiPost.mockResolvedValue({ topic: 'AI', confidence: '0.85' })
    const result = await knowledgeController.suggestTopic('neural networks')
    expect(result.topic).toBe('AI')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge/suggest-topic', { content: 'neural networks' })
  })

  it('trainAdapter POSTs /knowledge/train-adapter', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', fact_count: 5, elapsed: 1.2, adapter_status: { adapter_exists: true, fact_count: 5 } })
    const result = await knowledgeController.trainAdapter()
    expect(result.status).toBe('ok')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge/train-adapter')
  })

  it('getAdapterStatus GETs /knowledge/adapter-status', async () => {
    apiClient.apiGet.mockResolvedValue({ adapter_exists: true, fact_count: 5 })
    const result = await knowledgeController.getAdapterStatus()
    expect(result.fact_count).toBe(5)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/knowledge/adapter-status')
  })

  it('context GETs /knowledge/context', async () => {
    apiClient.apiGet.mockResolvedValue({ context: 'string', count: 1 })
    const result = await knowledgeController.context()
    expect(result.count).toBe(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/knowledge/context')
  })

  it('ingestUrl POSTs /knowledge/ingest-url', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', new_facts: 1, title: 'Page', content_length: 100, rejected: false })
    const result = await knowledgeController.ingestUrl('https://example.com')
    expect(result.status).toBe('ok')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge/ingest-url', { url: 'https://example.com' })
  })
})
