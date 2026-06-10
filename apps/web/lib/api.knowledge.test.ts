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

describe('knowledgeController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /knowledge and returns items', async () => {
    const items = [
      { id: 'k1', content: 'fact 1', topic: 'general', source: 'manual', url: '', timestamp: 1704067200, importance: 0.5, score: 0.0 },
      { id: 'k2', content: 'fact 2', topic: 'science', source: 'manual', url: '', timestamp: 1704153600, importance: 0.5, score: 0.0 },
    ]
    apiClient.apiGet.mockResolvedValue(items)

    const result = await knowledgeController.list()
    expect(result).toHaveLength(2)
    expect(result[0].content).toBe('fact 1')
    expect(result[0].topic).toBe('general')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/knowledge?limit=200&offset=0')
  })
})

describe('knowledgeController.add', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /knowledge with content and topic', async () => {
    const returned = { status: 'stored', content: 'new fact' }
    apiClient.apiPost.mockResolvedValue(returned)

    const item = await knowledgeController.add('new fact', 'custom')
    expect(item.status).toBe('stored')
    expect(item.content).toBe('new fact')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge', { content: 'new fact', topic: 'custom', auto_tag: false })
  })

  it('uses default topic when none specified', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'stored', content: 'plain' })

    await knowledgeController.add('plain')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge', { content: 'plain', topic: 'general', auto_tag: false })
  })
})

describe('knowledgeController.delete', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('DELETEs /knowledge/{id}', async () => {
    apiClient.apiDelete.mockResolvedValue({ error: undefined })

    await knowledgeController.delete('k1')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/knowledge/k1')
  })
})

describe('knowledgeController.search', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /knowledge/search with query param', async () => {
    const data = { results: [{ id: 'k1', content: 'fact about ai', topic: 'general', source: 'manual', url: '', timestamp: 0, importance: 0.5, score: 0.0 }] }
    apiClient.apiGet.mockResolvedValue(data)

    const result = await knowledgeController.search('ai')
    expect(result.results).toHaveLength(1)
    expect(result.results[0].topic).toBe('general')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/knowledge/search?query=ai')
  })

  it('returns empty results when no match', async () => {
    apiClient.apiGet.mockResolvedValue({ results: [] })
    const data = await knowledgeController.search('nonexistent')
    expect(data.results).toEqual([])
  })
})

describe('knowledgeController.batchIngest', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /knowledge/batch with items', async () => {
    apiClient.apiPost.mockResolvedValue({ stored: 2 })

    const result = await knowledgeController.batchIngest([
      { content: 'fact a', source: 'chat' },
      { content: 'fact b' },
    ])
    expect(result.stored).toBe(2)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge/batch', {
      items: [
        { content: 'fact a', source: 'chat' },
        { content: 'fact b' },
      ],
    })
  })
})

describe('knowledgeController.batchDelete', () => {
  beforeEach(() => vi.clearAllMocks())

  it('POSTs to /knowledge/batch-delete with IDs', async () => {
    apiClient.apiPost.mockResolvedValue({ deleted: 2 })

    const result = await knowledgeController.batchDelete(['id1', 'id2'])
    expect(result.deleted).toBe(2)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge/batch-delete', { ids: ['id1', 'id2'] })
  })
})

describe('knowledgeController.suggestTopic', () => {
  beforeEach(() => vi.clearAllMocks())

  it('POSTs to /knowledge/suggest-topic and returns topic', async () => {
    apiClient.apiPost.mockResolvedValue({ topic: 'code', confidence: 'high' })

    const result = await knowledgeController.suggestTopic('def foo(): pass')
    expect(result.topic).toBe('code')
    expect(result.confidence).toBe('high')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge/suggest-topic', { content: 'def foo(): pass' })
  })
})

describe('knowledgeController.related', () => {
  beforeEach(() => vi.clearAllMocks())

  it('GETs /knowledge/{id}/related', async () => {
    const related = { items: [{ id: 'k2', content: 'related fact', topic: 'code', source: '', url: '', timestamp: 0, importance: 0.5, score: 0.0 }], count: 1 }
    apiClient.apiGet.mockResolvedValue(related)

    const result = await knowledgeController.related('k1', 5)
    expect(result.items).toHaveLength(1)
    expect(result.items[0].content).toBe('related fact')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/knowledge/k1/related?top_k=5')
  })
})

describe('knowledgeController.trainAdapter', () => {
  beforeEach(() => vi.clearAllMocks())

  it('POSTs to /knowledge/train-adapter', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'trained', fact_count: 10, elapsed: 5.2, adapter_status: { adapter_exists: true, fact_count: 10 } })

    const result = await knowledgeController.trainAdapter()
    expect(result.status).toBe('trained')
    expect(result.fact_count).toBe(10)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge/train-adapter')
  })
})

describe('knowledgeController.getAdapterStatus', () => {
  beforeEach(() => vi.clearAllMocks())

  it('GETs /knowledge/adapter-status', async () => {
    apiClient.apiGet.mockResolvedValue({ adapter_exists: false, fact_count: 0, total_facts_available: 5 })

    const result = await knowledgeController.getAdapterStatus()
    expect(result.adapter_exists).toBe(false)
    expect(result.total_facts_available).toBe(5)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/knowledge/adapter-status')
  })
})
