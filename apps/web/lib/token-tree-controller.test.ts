import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

let mockApiGet: ReturnType<typeof vi.fn>
let mockApiPost: ReturnType<typeof vi.fn>
let mockApiDelete: ReturnType<typeof vi.fn>

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiDelete: (...args: unknown[]) => mockApiDelete(...args),
}))

const { tokenTreeController } = await import('@/lib/token-tree-controller')

const SAVED = {
  name: 'the-default',
  path: '/data/token_trees/the-default',
  vocab_size: 53,
  num_merges: 20,
  trained: true,
  saved_at: 1000,
}

describe('tokenTreeController', () => {
  beforeEach(() => {
    mockApiGet = vi.fn()
    mockApiPost = vi.fn()
    mockApiDelete = vi.fn()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('getVocab', () => {
    it('fetches paged vocabulary', async () => {
      mockApiGet.mockResolvedValue({ total: 3, entries: [] })
      const result = await tokenTreeController.getVocab(10, 20)
      expect(mockApiGet).toHaveBeenCalledWith('/token-tree/vocab?limit=10&offset=20')
      expect(result.total).toBe(3)
    })

    it('defaults to limit 50 offset 0', async () => {
      mockApiGet.mockResolvedValue({ total: 0, entries: [] })
      await tokenTreeController.getVocab()
      expect(mockApiGet).toHaveBeenCalledWith('/token-tree/vocab?limit=50&offset=0')
    })
  })

  describe('getMerges', () => {
    it('passes top_n and query', async () => {
      mockApiGet.mockResolvedValue([])
      await tokenTreeController.getMerges(5, 'th')
      expect(mockApiGet).toHaveBeenCalledWith('/token-tree/merges?top_n=5&query=th')
    })

    it('omits query when empty', async () => {
      mockApiGet.mockResolvedValue([])
      await tokenTreeController.getMerges(20)
      expect(mockApiGet).toHaveBeenCalledWith('/token-tree/merges?top_n=20')
    })
  })

  describe('listSaved', () => {
    it('unwraps trees from the response', async () => {
      mockApiGet.mockResolvedValue({ trees: [SAVED] })
      const result = await tokenTreeController.listSaved()
      expect(mockApiGet).toHaveBeenCalledWith('/token-tree/saved')
      expect(result).toEqual([SAVED])
    })

    it('propagates errors', async () => {
      mockApiGet.mockRejectedValue(new Error('fail'))
      await expect(tokenTreeController.listSaved()).rejects.toThrow('fail')
    })
  })

  describe('saveTree', () => {
    it('posts the tree name', async () => {
      mockApiPost.mockResolvedValue(SAVED)
      const result = await tokenTreeController.saveTree('the-default')
      expect(mockApiPost).toHaveBeenCalledWith('/token-tree/save', { name: 'the-default' })
      expect(result.name).toBe('the-default')
    })
  })

  describe('loadTree', () => {
    it('posts the tree name to load', async () => {
      mockApiPost.mockResolvedValue(SAVED)
      const result = await tokenTreeController.loadTree('the-default')
      expect(mockApiPost).toHaveBeenCalledWith('/token-tree/load', { name: 'the-default' })
      expect(result.vocab_size).toBe(53)
    })
  })

  describe('deleteSavedTree', () => {
    it('deletes the named tree', async () => {
      mockApiDelete.mockResolvedValue({ deleted: true })
      const result = await tokenTreeController.deleteSavedTree('the-default')
      expect(mockApiDelete).toHaveBeenCalledWith('/token-tree/saved/the-default')
      expect(result.deleted).toBe(true)
    })

    it('encodes the name in the path', async () => {
      mockApiDelete.mockResolvedValue({ deleted: true })
      await tokenTreeController.deleteSavedTree('a/b')
      expect(mockApiDelete).toHaveBeenCalledWith('/token-tree/saved/a%2Fb')
    })
  })

  describe('getEmbedding', () => {
    it('posts the token and top_k', async () => {
      const embed = {
        token: 'the',
        id: 3,
        dim: 8,
        norm: 1,
        top: [[0, 0.9]],
        embedding_points: 200,
        compression_ratio: 4,
      }
      mockApiPost.mockResolvedValue(embed)
      const result = await tokenTreeController.getEmbedding('the', 2)
      expect(mockApiPost).toHaveBeenCalledWith('/token-tree/embedding', { token: 'the', top_k: 2 })
      expect(result).toEqual(embed)
    })

    it('defaults top_k to 8', async () => {
      mockApiPost.mockResolvedValue({})
      await tokenTreeController.getEmbedding('quick')
      expect(mockApiPost).toHaveBeenCalledWith('/token-tree/embedding', { token: 'quick', top_k: 8 })
    })
  })

  describe('path', () => {
    it('posts text and returns the trace steps', async () => {
      const trace = {
        steps: [{ remaining: 'the</w>', token: 'the', id: 3, consumed: 7 }],
        ids: [3],
      }
      mockApiPost.mockResolvedValue(trace)
      const result = await tokenTreeController.path('the')
      expect(mockApiPost).toHaveBeenCalledWith('/token-tree/path', { text: 'the' })
      expect(result).toEqual(trace)
    })
  })
})
