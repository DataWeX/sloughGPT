import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockApiPost = vi.fn()
const mockApiGet = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiGet: (...args: unknown[]) => mockApiGet(...args),
}))

const { vectorController } = await import('@/lib/vector-controller')

describe('vectorController', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('getStats calls GET /vector/stats', async () => {
    mockApiGet.mockResolvedValue({ provider: 'in_memory', count: 42 })
    const result = await vectorController.getStats()
    expect(mockApiGet).toHaveBeenCalledWith('/vector/stats')
    expect(result).toEqual({ provider: 'in_memory', count: 42 })
  })

  it('init calls POST /vector/init with provider and dimension', async () => {
    mockApiPost.mockResolvedValue({ status: 'ok', provider: 'chromadb' })
    const result = await vectorController.init('chromadb', 512)
    expect(mockApiPost).toHaveBeenCalledWith('/vector/init', { provider: 'chromadb', dimension: 512 })
    expect(result).toEqual({ status: 'ok', provider: 'chromadb' })
  })

  it('init uses default provider and dimension', async () => {
    mockApiPost.mockResolvedValue({ status: 'ok', provider: 'in_memory' })
    await vectorController.init()
    expect(mockApiPost).toHaveBeenCalledWith('/vector/init', { provider: 'in_memory', dimension: 384 })
  })

  it('upsert calls POST /vector/upsert with texts, ids, metadata', async () => {
    mockApiPost.mockResolvedValue({ status: 'ok', count: 3, elapsed_ms: 15 })
    const result = await vectorController.upsert(['a', 'b', 'c'], ['1', '2', '3'], [{ x: 1 }, { x: 2 }, { x: 3 }])
    expect(mockApiPost).toHaveBeenCalledWith('/vector/upsert', { texts: ['a', 'b', 'c'], ids: ['1', '2', '3'], metadata: [{ x: 1 }, { x: 2 }, { x: 3 }] })
    expect(result).toEqual({ status: 'ok', count: 3, elapsed_ms: 15 })
  })

  it('search calls POST /vector/search with query and top_k', async () => {
    mockApiPost.mockResolvedValue({ results: [{ text: 'found', score: 0.95, id: '1' }], elapsed_ms: 5 })
    const result = await vectorController.search('test query', 10)
    expect(mockApiPost).toHaveBeenCalledWith('/vector/search', { query: 'test query', top_k: 10 })
    expect(result.results).toHaveLength(1)
    expect(result.results[0].score).toBe(0.95)
  })

  it('search uses default topK', async () => {
    mockApiPost.mockResolvedValue({ results: [], elapsed_ms: 1 })
    await vectorController.search('query')
    expect(mockApiPost).toHaveBeenCalledWith('/vector/search', { query: 'query', top_k: 5 })
  })

  it('ingestStatus calls GET /vector/ingest/status', async () => {
    mockApiGet.mockResolvedValue({ status: 'idle' })
    const result = await vectorController.ingestStatus()
    expect(mockApiGet).toHaveBeenCalledWith('/vector/ingest/status')
    expect(result).toEqual({ status: 'idle' })
  })

  it('propagates errors', async () => {
    mockApiGet.mockRejectedValue(new Error('conn refused'))
    await expect(vectorController.getStats()).rejects.toThrow('conn refused')
  })
})
