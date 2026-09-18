import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockApiGet = vi.fn()
const mockApiPost = vi.fn()
const mockApiPut = vi.fn()
const mockApiPatch = vi.fn()
const mockApiDelete = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiPut: (...args: unknown[]) => mockApiPut(...args),
  apiPatch: (...args: unknown[]) => mockApiPatch(...args),
  apiDelete: (...args: unknown[]) => mockApiDelete(...args),
}))

const { kbController } = await import('@/lib/kb-controller')

describe('kbController', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('list calls GET /knowledge with params', async () => {
    mockApiGet.mockResolvedValue([{ id: '1', content: 'test' }])
    const result = await kbController.list('science', 10, 20)
    expect(mockApiGet).toHaveBeenCalledWith('/knowledge?limit=10&offset=20&topic=science')
    expect(result).toHaveLength(1)
  })

  it('list omits topic when undefined', async () => {
    mockApiGet.mockResolvedValue([])
    await kbController.list()
    expect(mockApiGet).toHaveBeenCalledWith('/knowledge?limit=50&offset=0')
  })

  it('add calls POST /knowledge', async () => {
    mockApiPost.mockResolvedValue({ id: '1', content: 'new' })
    await kbController.add('new fact', 'science', 'manual', 0.8, true)
    expect(mockApiPost).toHaveBeenCalledWith('/knowledge', {
      content: 'new fact',
      topic: 'science',
      source: 'manual',
      importance: 0.8,
      auto_tag: true,
    })
  })

  it('add uses defaults', async () => {
    mockApiPost.mockResolvedValue({ id: '1' })
    await kbController.add('fact')
    expect(mockApiPost).toHaveBeenCalledWith('/knowledge', {
      content: 'fact',
      topic: 'general',
      source: 'manual',
      importance: 0.7,
      auto_tag: false,
    })
  })

  it('update calls PATCH /knowledge/:id', async () => {
    mockApiPatch.mockResolvedValue({ id: '1' })
    await kbController.update('1', { content: 'updated', topic: 'new' })
    expect(mockApiPatch).toHaveBeenCalledWith('/knowledge/1', { content: 'updated', topic: 'new' })
  })

  it('remove calls DELETE /knowledge/:id', async () => {
    mockApiDelete.mockResolvedValue({ deleted: true })
    const result = await kbController.remove('1')
    expect(mockApiDelete).toHaveBeenCalledWith('/knowledge/1')
    expect(result.deleted).toBe(true)
  })

  it('batchDelete calls POST /knowledge/batch-delete', async () => {
    mockApiPost.mockResolvedValue({ deleted: 3 })
    const result = await kbController.batchDelete(['1', '2', '3'])
    expect(mockApiPost).toHaveBeenCalledWith('/knowledge/batch-delete', { ids: ['1', '2', '3'] })
    expect(result.deleted).toBe(3)
  })

  it('search calls GET /knowledge/search', async () => {
    mockApiGet.mockResolvedValue([{ id: '1', score: 0.9 }])
    const result = await kbController.search('test query', 5)
    expect(mockApiGet).toHaveBeenCalledWith('/knowledge/search?q=test+query&limit=5')
    expect(result).toHaveLength(1)
  })

  it('stats calls GET /knowledge/stats', async () => {
    mockApiGet.mockResolvedValue({
      total_items: 10,
      topics: ['a'],
      avg_importance: 0.7,
      sources: {},
    })
    const result = await kbController.stats()
    expect(mockApiGet).toHaveBeenCalledWith('/knowledge/stats')
    expect(result.total_items).toBe(10)
  })

  it('topics calls GET /knowledge/topics', async () => {
    mockApiGet.mockResolvedValue([{ name: 'science', count: 5 }])
    const result = await kbController.topics()
    expect(mockApiGet).toHaveBeenCalledWith('/knowledge/topics')
    expect(result[0].name).toBe('science')
  })

  it('ingestUrl calls POST /knowledge/ingest-url', async () => {
    mockApiPost.mockResolvedValue({ status: 'ok', id: '1' })
    await kbController.ingestUrl('https://example.com', 'web')
    expect(mockApiPost).toHaveBeenCalledWith('/knowledge/ingest-url', {
      url: 'https://example.com',
      source: 'web',
    })
  })

  it('batchIngest calls POST /knowledge/batch', async () => {
    mockApiPost.mockResolvedValue({ ingested: 2 })
    const result = await kbController.batchIngest([{ content: 'a' }, { content: 'b' }])
    expect(mockApiPost).toHaveBeenCalledWith('/knowledge/batch', {
      items: [{ content: 'a' }, { content: 'b' }],
    })
    expect(result.ingested).toBe(2)
  })

  it('suggestTopic calls POST /knowledge/suggest-topic', async () => {
    mockApiPost.mockResolvedValue({ topic: 'science' })
    const result = await kbController.suggestTopic('quantum physics is fascinating')
    expect(mockApiPost).toHaveBeenCalledWith('/knowledge/suggest-topic', {
      content: 'quantum physics is fascinating',
    })
    expect(result.topic).toBe('science')
  })

  it('checkDuplicate calls POST /knowledge/check-duplicate', async () => {
    mockApiPost.mockResolvedValue({ is_duplicate: true, similar: [] })
    const result = await kbController.checkDuplicate('some text')
    expect(mockApiPost).toHaveBeenCalledWith('/knowledge/check-duplicate', { content: 'some text' })
    expect(result.is_duplicate).toBe(true)
  })

  it('categorize calls POST /knowledge/:id/categorize', async () => {
    mockApiPost.mockResolvedValue({ updated: true })
    await kbController.categorize('1', 'history')
    expect(mockApiPost).toHaveBeenCalledWith('/knowledge/1/categorize', { topic: 'history' })
  })

  it('gaps calls GET /knowledge/gaps', async () => {
    mockApiGet.mockResolvedValue({ gaps: ['missing topic'], suggestions: ['add more'] })
    const result = await kbController.gaps()
    expect(mockApiGet).toHaveBeenCalledWith('/knowledge/gaps')
    expect(result.gaps).toHaveLength(1)
  })

  it('context calls GET /knowledge/context', async () => {
    mockApiGet.mockResolvedValue({ context: 'relevant info', items: [] })
    const result = await kbController.context('query', 3)
    expect(mockApiGet).toHaveBeenCalledWith('/knowledge/context?q=query&top_k=3')
    expect(result.context).toBe('relevant info')
  })

  it('trainAdapter calls POST /knowledge/train-adapter', async () => {
    mockApiPost.mockResolvedValue({ status: 'started', job_id: 'j1' })
    const result = await kbController.trainAdapter(15)
    expect(mockApiPost).toHaveBeenCalledWith('/knowledge/train-adapter', { top_k: 15 })
    expect(result.job_id).toBe('j1')
  })

  it('adapterStatus calls GET /knowledge/adapter-status', async () => {
    mockApiGet.mockResolvedValue({ trained: true, accuracy: 0.85 })
    const result = await kbController.adapterStatus()
    expect(mockApiGet).toHaveBeenCalledWith('/knowledge/adapter-status')
    expect(result.trained).toBe(true)
  })

  it('related calls GET /knowledge/:id/related', async () => {
    mockApiGet.mockResolvedValue([{ id: '2', content: 'related' }])
    const result = await kbController.related('1', 3)
    expect(mockApiGet).toHaveBeenCalledWith('/knowledge/1/related?limit=3')
    expect(result).toHaveLength(1)
  })

  it('propagates errors', async () => {
    mockApiGet.mockRejectedValue(new Error('network'))
    await expect(kbController.stats()).rejects.toThrow('network')
  })
})
