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
    expect(apiClient.apiGet).toHaveBeenCalledWith('/knowledge')
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
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge', { content: 'new fact', topic: 'custom' })
  })

  it('uses default topic when none specified', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'stored', content: 'plain' })

    await knowledgeController.add('plain')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/knowledge', { content: 'plain', topic: 'general' })
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
