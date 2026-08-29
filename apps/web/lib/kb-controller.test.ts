import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockApiGet = vi.fn()
const mockApiPost = vi.fn()
const mockApiPut = vi.fn()
const mockApiDelete = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiPut: (...args: unknown[]) => mockApiPut(...args),
  apiDelete: (...args: unknown[]) => mockApiDelete(...args),
}))

const { kbController } = await import('@/lib/kb-controller')

describe('kbController', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('list calls GET /kb with params', async () => {
    mockApiGet.mockResolvedValue([{ id: '1', content: 'test' }])
    const result = await kbController.list('science', 10, 20)
    expect(mockApiGet).toHaveBeenCalledWith('/kb?limit=10&offset=20&topic=science')
    expect(result).toHaveLength(1)
  })

  it('list omits topic when undefined', async () => {
    mockApiGet.mockResolvedValue([])
    await kbController.list()
    expect(mockApiGet).toHaveBeenCalledWith('/kb?limit=50&offset=0')
  })

  it('add calls POST /kb', async () => {
    mockApiPost.mockResolvedValue({ id: '1', content: 'new' })
    await kbController.add('new fact', 'science', 'manual', 0.8, true)
    expect(mockApiPost).toHaveBeenCalledWith('/kb', { content: 'new fact', topic: 'science', source: 'manual', importance: 0.8, auto_tag: true })
  })

  it('add uses defaults', async () => {
    mockApiPost.mockResolvedValue({ id: '1' })
    await kbController.add('fact')
    expect(mockApiPost).toHaveBeenCalledWith('/kb', { content: 'fact', topic: 'general', source: 'manual', importance: 0.7, auto_tag: false })
  })

  it('update calls PUT /kb/:id', async () => {
    mockApiPut.mockResolvedValue({ id: '1' })
    await kbController.update('1', { content: 'updated', topic: 'new' })
    expect(mockApiPut).toHaveBeenCalledWith('/kb/1', { content: 'updated', topic: 'new' })
  })

  it('remove calls DELETE /kb/:id', async () => {
    mockApiDelete.mockResolvedValue({ deleted: true })
    const result = await kbController.remove('1')
    expect(mockApiDelete).toHaveBeenCalledWith('/kb/1')
    expect(result.deleted).toBe(true)
  })

  it('batchDelete calls POST /kb/batch-delete', async () => {
    mockApiPost.mockResolvedValue({ deleted: 3 })
    const result = await kbController.batchDelete(['1', '2', '3'])
    expect(mockApiPost).toHaveBeenCalledWith('/kb/batch-delete', { ids: ['1', '2', '3'] })
    expect(result.deleted).toBe(3)
  })

  it('search calls GET /kb/search', async () => {
    mockApiGet.mockResolvedValue([{ id: '1', score: 0.9 }])
    const result = await kbController.search('test query', 5)
    expect(mockApiGet).toHaveBeenCalledWith('/kb/search?q=test+query&limit=5')
    expect(result).toHaveLength(1)
  })

  it('stats calls GET /kb/stats', async () => {
    mockApiGet.mockResolvedValue({ total_items: 10, topics: ['a'], avg_importance: 0.7, sources: {} })
    const result = await kbController.stats()
    expect(mockApiGet).toHaveBeenCalledWith('/kb/stats')
    expect(result.total_items).toBe(10)
  })

  it('topics calls GET /kb/topics', async () => {
    mockApiGet.mockResolvedValue([{ name: 'science', count: 5 }])
    const result = await kbController.topics()
    expect(mockApiGet).toHaveBeenCalledWith('/kb/topics')
    expect(result[0].name).toBe('science')
  })

  it('ingestUrl calls POST /kb/ingest-url', async () => {
    mockApiPost.mockResolvedValue({ status: 'ok', id: '1' })
    await kbController.ingestUrl('https://example.com', 'web')
    expect(mockApiPost).toHaveBeenCalledWith('/kb/ingest-url', { url: 'https://example.com', source: 'web' })
  })

  it('batchIngest calls POST /kb/batch', async () => {
    mockApiPost.mockResolvedValue({ ingested: 2 })
    const result = await kbController.batchIngest([{ content: 'a' }, { content: 'b' }])
    expect(mockApiPost).toHaveBeenCalledWith('/kb/batch', { items: [{ content: 'a' }, { content: 'b' }] })
    expect(result.ingested).toBe(2)
  })

  it('suggestTopic calls POST /kb/suggest-topic', async () => {
    mockApiPost.mockResolvedValue({ topic: 'science' })
    const result = await kbController.suggestTopic('quantum physics is fascinating')
    expect(mockApiPost).toHaveBeenCalledWith('/kb/suggest-topic', { content: 'quantum physics is fascinating' })
    expect(result.topic).toBe('science')
  })

  it('checkDuplicate calls POST /kb/check-duplicate', async () => {
    mockApiPost.mockResolvedValue({ is_duplicate: true, similar: [] })
    const result = await kbController.checkDuplicate('some text')
    expect(mockApiPost).toHaveBeenCalledWith('/kb/check-duplicate', { content: 'some text' })
    expect(result.is_duplicate).toBe(true)
  })

  it('categorize calls POST /kb/:id/categorize', async () => {
    mockApiPost.mockResolvedValue({ updated: true })
    await kbController.categorize('1', 'history')
    expect(mockApiPost).toHaveBeenCalledWith('/kb/1/categorize', { topic: 'history' })
  })

  it('gaps calls GET /kb/gaps', async () => {
    mockApiGet.mockResolvedValue({ gaps: ['missing topic'], suggestions: ['add more'] })
    const result = await kbController.gaps()
    expect(mockApiGet).toHaveBeenCalledWith('/kb/gaps')
    expect(result.gaps).toHaveLength(1)
  })

  it('context calls GET /kb/context', async () => {
    mockApiGet.mockResolvedValue({ context: 'relevant info', items: [] })
    const result = await kbController.context('query', 3)
    expect(mockApiGet).toHaveBeenCalledWith('/kb/context?q=query&top_k=3')
    expect(result.context).toBe('relevant info')
  })

  it('trainAdapter calls POST /kb/train-adapter', async () => {
    mockApiPost.mockResolvedValue({ status: 'started', job_id: 'j1' })
    const result = await kbController.trainAdapter(15)
    expect(mockApiPost).toHaveBeenCalledWith('/kb/train-adapter', { top_k: 15 })
    expect(result.job_id).toBe('j1')
  })

  it('adapterStatus calls GET /kb/adapter-status', async () => {
    mockApiGet.mockResolvedValue({ trained: true, accuracy: 0.85 })
    const result = await kbController.adapterStatus()
    expect(mockApiGet).toHaveBeenCalledWith('/kb/adapter-status')
    expect(result.trained).toBe(true)
  })

  it('related calls GET /kb/:id/related', async () => {
    mockApiGet.mockResolvedValue([{ id: '2', content: 'related' }])
    const result = await kbController.related('1', 3)
    expect(mockApiGet).toHaveBeenCalledWith('/kb/1/related?limit=3')
    expect(result).toHaveLength(1)
  })

  it('propagates errors', async () => {
    mockApiGet.mockRejectedValue(new Error('network'))
    await expect(kbController.stats()).rejects.toThrow('network')
  })
})
