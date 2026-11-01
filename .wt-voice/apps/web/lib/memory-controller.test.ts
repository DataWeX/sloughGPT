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

import { memoryController } from './memory-controller'

const item = {
  id: 'm1',
  content: 'Likes espresso in the morning',
  topic: 'preferences',
  source: 'task',
  url: '',
  timestamp: 1700000000,
  importance: 0.8,
  score: 0.9,
}

describe('memoryController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /memory/list with limit and returns items', async () => {
    apiClient.apiGet.mockResolvedValue({ items: [item], total: 1 })

    const result = await memoryController.list()
    expect(result.items).toHaveLength(1)
    expect(result.total).toBe(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/memory/list?limit=50')
  })

  it('accepts a custom limit', async () => {
    apiClient.apiGet.mockResolvedValue({ items: [], total: 0 })
    await memoryController.list(10)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/memory/list?limit=10')
  })

  it('returns empty response on missing payload', async () => {
    apiClient.apiGet.mockResolvedValue(undefined)
    const result = await memoryController.list()
    expect(result).toEqual({ items: [], total: 0 })
  })
})

describe('memoryController.stats', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /memory/stats and returns stats', async () => {
    apiClient.apiGet.mockResolvedValue({ enabled: true, total_facts: 42, topics: 4, visited_urls: 7 })

    const result = await memoryController.stats()
    expect(result?.enabled).toBe(true)
    expect(result?.total_facts).toBe(42)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/memory/stats')
  })

  it('returns null on missing payload', async () => {
    apiClient.apiGet.mockResolvedValue(undefined)
    expect(await memoryController.stats()).toBeNull()
  })
})

describe('memoryController.search', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /memory/search with encoded query and limit', async () => {
    apiClient.apiGet.mockResolvedValue({ results: [item], total: 1 })

    const result = await memoryController.search('espresso coffee', 3)
    expect(result.results).toHaveLength(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/memory/search?q=espresso%20coffee&limit=3')
  })

  it('defaults limit to 5', async () => {
    apiClient.apiGet.mockResolvedValue({ results: [], total: 0 })
    await memoryController.search('anything')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/memory/search?q=anything&limit=5')
  })

  it('returns empty on missing payload', async () => {
    apiClient.apiGet.mockResolvedValue(undefined)
    expect(await memoryController.search('x')).toEqual({ results: [], total: 0 })
  })
})

describe('memoryController.store', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs content/topic/source to /memory/store', async () => {
    apiClient.apiPost.mockResolvedValue({ stored: true, content: 'fact', topic: 'manual', source: 'api' })

    const result = await memoryController.store('fact')
    expect(result.stored).toBe(true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/memory/store', { content: 'fact', topic: 'manual', source: 'api' })
  })

  it('honors custom topic and source', async () => {
    apiClient.apiPost.mockResolvedValue({ stored: true, content: 'fact', topic: 'planning', source: 'web' })
    await memoryController.store('fact', 'planning', 'web')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/memory/store', { content: 'fact', topic: 'planning', source: 'web' })
  })
})

describe('memoryController.remember', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs user_message and assistant_response', async () => {
    apiClient.apiPost.mockResolvedValue({ stored: true, reason: 'stored' })

    const result = await memoryController.remember('hello', 'hi there')
    expect(result.stored).toBe(true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/memory/remember', {
      user_message: 'hello',
      assistant_response: 'hi there',
    })
  })
})

describe('memoryController.setEnabled', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs enabled=true to /memory/config', async () => {
    apiClient.apiPost.mockResolvedValue({ enabled: true })

    const result = await memoryController.setEnabled(true)
    expect(result.enabled).toBe(true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/memory/config', { enabled: true })
  })

  it('POSTs enabled=false to /memory/config', async () => {
    apiClient.apiPost.mockResolvedValue({ enabled: false })

    const result = await memoryController.setEnabled(false)
    expect(result.enabled).toBe(false)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/memory/config', { enabled: false })
  })
})

describe('memoryController.getConfig', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /memory/config and returns the snapshot', async () => {
    apiClient.apiGet.mockResolvedValue({ enabled: true, archive_retention_days: 45 })

    const result = await memoryController.getConfig()
    expect(result.archive_retention_days).toBe(45)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/memory/config')
  })
})

describe('memoryController.updateConfig', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs the partial update to /memory/config', async () => {
    apiClient.apiPost.mockResolvedValue({ enabled: true, archive_retention_days: 7 })

    const result = await memoryController.updateConfig({ archive_retention_days: 7 })
    expect(result.archive_retention_days).toBe(7)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/memory/config', { archive_retention_days: 7 })
  })
})

describe('memoryController.clear', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs /memory/clear and returns cleared count', async () => {
    apiClient.apiPost.mockResolvedValue({ cleared: 5 })

    const result = await memoryController.clear()
    expect(result.cleared).toBe(5)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/memory/clear')
  })
})

describe('memoryController.delete', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('DELETEs /memory/{id} and returns deleted count', async () => {
    apiClient.apiDelete.mockResolvedValue({ deleted: 1 })

    const result = await memoryController.delete('m1')
    expect(result.deleted).toBe(1)
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/memory/m1')
  })

  it('encodes the item id', async () => {
    apiClient.apiDelete.mockResolvedValue({ deleted: 0 })
    await memoryController.delete('id with spaces')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/memory/id%20with%20spaces')
  })
})

describe('memoryController.consolidate', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs /memory/consolidate without a threshold by default', async () => {
    apiClient.apiPost.mockResolvedValue({ removed: 2, kept: 3, threshold: 0.8 })

    const result = await memoryController.consolidate()
    expect(result.removed).toBe(2)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/memory/consolidate')
  })

  it('passes an explicit threshold as a query param', async () => {
    apiClient.apiPost.mockResolvedValue({ removed: 0, kept: 4, threshold: 0.5 })
    await memoryController.consolidate(0.5)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/memory/consolidate?threshold=0.5')
  })
})

describe('memoryController.archive', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /memory/archive with limit and returns records', async () => {
    apiClient.apiGet.mockResolvedValue({ records: [{ ts: 1, task_id: 't1', task_type: 'memory.store' }], total: 1 })

    const result = await memoryController.archive()
    expect(result.total).toBe(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/memory/archive?limit=20')
  })

  it('accepts a custom limit', async () => {
    apiClient.apiGet.mockResolvedValue({ records: [], total: 0 })
    await memoryController.archive(5)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/memory/archive?limit=5')
  })

  it('returns empty on missing payload', async () => {
    apiClient.apiGet.mockResolvedValue(undefined)
    expect(await memoryController.archive()).toEqual({ records: [], total: 0 })
  })
})

describe('memoryController.archiveStats', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /memory/archive/stats and returns stats', async () => {
    apiClient.apiGet.mockResolvedValue({ path: '/tmp/facts.jsonl', records: 7, bytes: 2048, task_types: {}, oldest_ts: 1, newest_ts: 2 })

    const result = await memoryController.archiveStats()
    expect(result?.records).toBe(7)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/memory/archive/stats')
  })

  it('returns null on missing payload', async () => {
    apiClient.apiGet.mockResolvedValue(undefined)
    expect(await memoryController.archiveStats()).toBeNull()
  })
})

describe('memoryController.archivePrune', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs /memory/archive/prune without retain_days by default', async () => {
    apiClient.apiPost.mockResolvedValue({ pruned: 4 })

    const result = await memoryController.archivePrune()
    expect(result.pruned).toBe(4)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/memory/archive/prune')
  })

  it('passes retain_days as a query param', async () => {
    apiClient.apiPost.mockResolvedValue({ pruned: 0 })
    await memoryController.archivePrune(7)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/memory/archive/prune?retain_days=7')
  })
})

describe('memoryController.update', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('PATCHes /memory/{id} with content and topic', async () => {
    apiClient.apiPatch.mockResolvedValue({ updated: 1, duplicate: false })

    const result = await memoryController.update('fact_1_abc', 'Likes espresso in the morning', 'drinks')
    expect(result.updated).toBe(1)
    expect(apiClient.apiPatch).toHaveBeenCalledWith('/memory/fact_1_abc', {
      content: 'Likes espresso in the morning',
      topic: 'drinks',
    })
  })

  it('omits topic when not provided', async () => {
    apiClient.apiPatch.mockResolvedValue({ updated: 1, duplicate: false })

    await memoryController.update('fact_1_abc', 'New text')
    expect(apiClient.apiPatch).toHaveBeenCalledWith('/memory/fact_1_abc', {
      content: 'New text',
      topic: undefined,
    })
  })

  it('omits a whitespace-only topic', async () => {
    apiClient.apiPatch.mockResolvedValue({ updated: 0, duplicate: true })

    await memoryController.update('fact_1_abc', 'New text', '   ')
    expect(apiClient.apiPatch).toHaveBeenCalledWith('/memory/fact_1_abc', {
      content: 'New text',
      topic: undefined,
    })
  })

  it('includes importance when provided', async () => {
    apiClient.apiPatch.mockResolvedValue({ updated: 1, duplicate: false })

    await memoryController.update('fact_1_abc', 'New text', 'drinks', 0.9)
    expect(apiClient.apiPatch).toHaveBeenCalledWith('/memory/fact_1_abc', {
      content: 'New text',
      topic: 'drinks',
      importance: 0.9,
    })
  })

  it('omits importance when not a finite number', async () => {
    apiClient.apiPatch.mockResolvedValue({ updated: 1, duplicate: false })

    await memoryController.update('fact_1_abc', 'New text', undefined, NaN)
    expect(apiClient.apiPatch).toHaveBeenCalledWith('/memory/fact_1_abc', {
      content: 'New text',
      topic: undefined,
    })
  })
})
