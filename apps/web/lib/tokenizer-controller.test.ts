import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

let mockApiGet: ReturnType<typeof vi.fn>
let mockApiPost: ReturnType<typeof vi.fn>

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
}))

const { tokenizerController } = await import('@/lib/tokenizer-controller')

describe('tokenizerController', () => {
  beforeEach(() => {
    mockApiGet = vi.fn()
    mockApiPost = vi.fn()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('getStats', () => {
    it('fetches tokenizer stats', async () => {
      mockApiGet.mockResolvedValue({ vocab_size: 256, base_chars: 128, merged_subwords: 128, special_tokens: 0, total_merges: 128, trained: true })
      const result = await tokenizerController.getStats()
      expect(mockApiGet).toHaveBeenCalledWith('/tokenizer/stats')
      expect(result.vocab_size).toBe(256)
      expect(result.trained).toBe(true)
    })

    it('propagates errors', async () => {
      mockApiGet.mockRejectedValue(new Error('fail'))
      await expect(tokenizerController.getStats()).rejects.toThrow('fail')
    })
  })

  describe('tokenize', () => {
    it('posts text for tokenization', async () => {
      mockApiPost.mockResolvedValue({ tokens: ['hello', ' world'], ids: [1, 2] })
      const result = await tokenizerController.tokenize('hello world')
      expect(mockApiPost).toHaveBeenCalledWith('/tokenizer/tokenize', { text: 'hello world' })
      expect(result.tokens).toEqual(['hello', ' world'])
      expect(result.ids).toEqual([1, 2])
    })
  })

  describe('detokenize', () => {
    it('posts ids for detokenization', async () => {
      mockApiPost.mockResolvedValue({ text: 'hello world' })
      const result = await tokenizerController.detokenize([1, 2])
      expect(mockApiPost).toHaveBeenCalledWith('/tokenizer/detokenize', { ids: [1, 2] })
      expect(result.text).toBe('hello world')
    })
  })

  describe('getVocab', () => {
    it('fetches vocab with default params', async () => {
      mockApiGet.mockResolvedValue({ entries: [], total: 100 })
      await tokenizerController.getVocab()
      expect(mockApiGet).toHaveBeenCalledWith('/tokenizer/vocab?limit=50&offset=0')
    })

    it('fetches vocab with custom params', async () => {
      mockApiGet.mockResolvedValue({ entries: [], total: 100 })
      await tokenizerController.getVocab(10, 20)
      expect(mockApiGet).toHaveBeenCalledWith('/tokenizer/vocab?limit=10&offset=20')
    })
  })

  describe('getMerges', () => {
    it('fetches merges with default limit', async () => {
      mockApiGet.mockResolvedValue({ merges: [], total: 50 })
      await tokenizerController.getMerges()
      expect(mockApiGet).toHaveBeenCalledWith('/tokenizer/merges?limit=30')
    })

    it('fetches merges with custom limit', async () => {
      mockApiGet.mockResolvedValue({ merges: [], total: 50 })
      await tokenizerController.getMerges(100)
      expect(mockApiGet).toHaveBeenCalledWith('/tokenizer/merges?limit=100')
    })
  })

  describe('getSamples', () => {
    it('fetches sample words', async () => {
      mockApiGet.mockResolvedValue({ samples: [{ word: 'hello', ids: [1], tokens: ['hel', 'lo'], count: 5 }] })
      const result = await tokenizerController.getSamples()
      expect(mockApiGet).toHaveBeenCalledWith('/tokenizer/sample')
      expect(result.samples).toHaveLength(1)
      expect(result.samples[0].word).toBe('hello')
    })
  })

  describe('train', () => {
    it('trains tokenizer with params', async () => {
      mockApiPost.mockResolvedValue({ status: 'ok', corpus_size: 1000, stats: { vocab_size: 256 } })
      const result = await tokenizerController.train({ vocab_size: 256, texts: ['hello'] })
      expect(mockApiPost).toHaveBeenCalledWith('/tokenizer/train', { vocab_size: 256, texts: ['hello'] })
      expect(result.status).toBe('ok')
      expect(result.corpus_size).toBe(1000)
    })
  })

  describe('pretokenize', () => {
    it('posts text for pretokenization', async () => {
      mockApiPost.mockResolvedValue({ chunks: ['hello', 'world'] })
      const result = await tokenizerController.pretokenize('hello world')
      expect(mockApiPost).toHaveBeenCalledWith('/tokenizer/pretokenize', { text: 'hello world' })
      expect(result).toEqual({ chunks: ['hello', 'world'] })
    })
  })

  describe('decompose', () => {
    it('posts text for decomposition', async () => {
      mockApiPost.mockResolvedValue({ morphemes: ['un', 'believ', 'able'] })
      const result = await tokenizerController.decompose('unbelievable')
      expect(mockApiPost).toHaveBeenCalledWith('/tokenizer/decompose', { text: 'unbelievable' })
      expect(result).toEqual({ morphemes: ['un', 'believ', 'able'] })
    })
  })

  describe('analyze', () => {
    it('posts texts for analysis', async () => {
      mockApiPost.mockResolvedValue({ results: [{}] })
      const result = await tokenizerController.analyze(['hello', 'world'])
      expect(mockApiPost).toHaveBeenCalledWith('/tokenizer/analyze', { texts: ['hello', 'world'] })
      expect(result).toEqual({ results: [{}] })
    })
  })
})
