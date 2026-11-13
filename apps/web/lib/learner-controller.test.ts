import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

let mockApiGet: ReturnType<typeof vi.fn>
let mockApiPost: ReturnType<typeof vi.fn>

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
}))

const { learnerController } = await import('@/lib/learner-controller')

describe('learnerController', () => {
  beforeEach(() => {
    mockApiGet = vi.fn()
    mockApiPost = vi.fn()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('status', () => {
    it('fetches learner status', async () => {
      mockApiGet.mockResolvedValue({
        soul_name: 'test', total_tokens_ingested: 500, train_steps_completed: 10,
        current_loss: 0.5, loss_history: [], buffer_size: 3, buffer_capacity: 100,
        pending_tokens: 0, arch: 'transformer', n_embed: 128, n_layer: 4, n_head: 4,
        vocab_size: 256, knowledge: {}, feeds_subscribed: 2,
        filter_stats: {}, filter_config: {},
      })
      const result = await learnerController.status()
      expect(mockApiGet).toHaveBeenCalledWith('/learn/status')
      expect(result.total_tokens_ingested).toBe(500)
      expect(result.feeds_subscribed).toBe(2)
    })

    it('propagates errors', async () => {
      mockApiGet.mockRejectedValue(new Error('fail'))
      await expect(learnerController.status()).rejects.toThrow('fail')
    })
  })

  describe('search', () => {
    it('posts search query', async () => {
      mockApiPost.mockResolvedValue({ tokens_ingested: 100, new_facts: 5, rejected: 0, filter_stats: {} })
      const result = await learnerController.search('machine learning')
      expect(mockApiPost).toHaveBeenCalledWith('/learn/search?query=machine+learning&max_results=5')
      expect(result.tokens_ingested).toBe(100)
      expect(result.new_facts).toBe(5)
    })

    it('passes custom maxResults', async () => {
      mockApiPost.mockResolvedValue({ tokens_ingested: 0, new_facts: 0, rejected: 0, filter_stats: {} })
      await learnerController.search('test', 10)
      expect(mockApiPost).toHaveBeenCalledWith('/learn/search?query=test&max_results=10')
    })
  })

  describe('ingestUrl', () => {
    it('posts URL for ingestion', async () => {
      mockApiPost.mockResolvedValue({ status: 'ok', facts_added: 3 })
      const result = await learnerController.ingestUrl('https://example.com')
      expect(mockApiPost).toHaveBeenCalledWith('/learn/ingest-url?url=https%3A%2F%2Fexample.com')
      expect(result.facts_added).toBe(3)
    })
  })

  describe('ingestText', () => {
    it('posts text for ingestion', async () => {
      mockApiPost.mockResolvedValue({ status: 'ok', facts_added: 2 })
      const result = await learnerController.ingestText('Some text to learn from')
      expect(mockApiPost).toHaveBeenCalledWith('/learn/ingest', { text: 'Some text to learn from' })
      expect(result.facts_added).toBe(2)
    })
  })

  describe('queryKnowledge', () => {
    it('fetches knowledge without query', async () => {
      mockApiGet.mockResolvedValue({ facts: [], total: 0 })
      await learnerController.queryKnowledge()
      expect(mockApiGet).toHaveBeenCalledWith('/learn/knowledge?top_k=20')
    })

    it('fetches knowledge with query', async () => {
      mockApiGet.mockResolvedValue({ facts: [], total: 0 })
      await learnerController.queryKnowledge('python', 10)
      expect(mockApiGet).toHaveBeenCalledWith('/learn/knowledge?query=python&top_k=10')
    })
  })

  describe('subscribeFeed', () => {
    it('subscribes to feed with default interval', async () => {
      mockApiPost.mockResolvedValue({ status: 'ok' })
      await learnerController.subscribeFeed('https://rss.example.com/feed.xml')
      expect(mockApiPost).toHaveBeenCalledWith('/learn/feed?action=subscribe&url=https%3A%2F%2Frss.example.com%2Ffeed.xml&poll_interval=3600')
    })

    it('subscribes to feed with custom interval', async () => {
      mockApiPost.mockResolvedValue({ status: 'ok' })
      await learnerController.subscribeFeed('https://rss.example.com/feed.xml', 1800)
      expect(mockApiPost).toHaveBeenCalledWith('/learn/feed?action=subscribe&url=https%3A%2F%2Frss.example.com%2Ffeed.xml&poll_interval=1800')
    })
  })

  describe('unsubscribeFeed', () => {
    it('unsubscribes from feed', async () => {
      mockApiPost.mockResolvedValue({ status: 'ok' })
      await learnerController.unsubscribeFeed('https://rss.example.com/feed.xml')
      expect(mockApiPost).toHaveBeenCalledWith('/learn/feed?action=unsubscribe&url=https%3A%2F%2Frss.example.com%2Ffeed.xml')
    })
  })

  describe('listFeeds', () => {
    it('fetches feed list', async () => {
      mockApiPost.mockResolvedValue({ feeds: [{ url: 'https://rss.example.com', interval: 3600 }] })
      const result = await learnerController.listFeeds()
      expect(mockApiPost).toHaveBeenCalledWith('/learn/feed?action=list')
      expect(result.feeds).toHaveLength(1)
      expect(result.feeds[0].url).toBe('https://rss.example.com')
    })
  })

  describe('train', () => {
    it('trains learner', async () => {
      mockApiPost.mockResolvedValue({ status: 'ok', loss: 0.5 })
      const result = await learnerController.train()
      expect(mockApiPost).toHaveBeenCalledWith('/learn/train')
      expect(result.status).toBe('ok')
      expect(result.loss).toBe(0.5)
    })
  })

  describe('evaluate', () => {
    it('evaluates learner', async () => {
      mockApiPost.mockResolvedValue({ metrics: { accuracy: 0.85 } })
      const result = await learnerController.evaluate()
      expect(mockApiPost).toHaveBeenCalledWith('/learn/evaluate')
      expect(result.metrics).toEqual({ accuracy: 0.85 })
    })
  })

  describe('deploy', () => {
    it('deploys learner', async () => {
      mockApiPost.mockResolvedValue({ status: 'deployed' })
      const result = await learnerController.deploy()
      expect(mockApiPost).toHaveBeenCalledWith('/learn/deploy')
      expect(result.status).toBe('deployed')
    })
  })
})
