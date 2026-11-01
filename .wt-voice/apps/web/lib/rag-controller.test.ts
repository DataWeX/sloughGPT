/**
 * Tests for the RAG controller (lib/rag-controller.ts).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock http-client
vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))

import { apiGet, apiPost } from '@/lib/http-client'
import {
  ingestDocument,
  queryRAG,
  verifyRAG,
  listRAGDocuments,
  getRAGStats,
  clearRAG,
} from '@/lib/rag-controller'

const mockApiPost = vi.mocked(apiPost)
const mockApiGet = vi.mocked(apiGet)

beforeEach(() => {
  vi.clearAllMocks()
})

describe('rag-controller', () => {
  describe('ingestDocument', () => {
    it('calls POST /knowledge/rag/ingest with correct params', async () => {
      mockApiPost.mockResolvedValue({ chunk_ids: ['c1', 'c2'], num_chunks: 2, stats: { total_documents: 1, total_chunks: 2, index_size: 10 } })
      const result = await ingestDocument('test content', 'test-source', 'test-topic', 256)
      expect(mockApiPost).toHaveBeenCalledWith('/knowledge/rag/ingest', {
        content: 'test content',
        source: 'test-source',
        topic: 'test-topic',
        chunk_size: 256,
      })
      expect(result.num_chunks).toBe(2)
    })

    it('uses defaults when optional params omitted', async () => {
      mockApiPost.mockResolvedValue({ chunk_ids: ['c1'], num_chunks: 1, stats: {} })
      await ingestDocument('hello')
      expect(mockApiPost).toHaveBeenCalledWith('/knowledge/rag/ingest', {
        content: 'hello',
        source: 'user',
        topic: 'general',
        chunk_size: 512,
      })
    })
  })

  describe('queryRAG', () => {
    it('calls POST /knowledge/rag/query', async () => {
      mockApiPost.mockResolvedValue({ question: 'test', results: [], context: '', num_results: 0 })
      const result = await queryRAG('What is Python?', 3)
      expect(mockApiPost).toHaveBeenCalledWith('/knowledge/rag/query', { question: 'What is Python?', top_k: 3 })
      expect(result.num_results).toBe(0)
    })
  })

  describe('verifyRAG', () => {
    it('calls POST /knowledge/rag/verify', async () => {
      mockApiPost.mockResolvedValue({ verification: {}, confidence: 0.9, is_verified: true })
      const result = await verifyRAG('Python is great', 'What is Python?')
      expect(mockApiPost).toHaveBeenCalledWith('/knowledge/rag/verify', {
        text: 'Python is great',
        question: 'What is Python?',
      })
      expect(result.is_verified).toBe(true)
    })
  })

  describe('listRAGDocuments', () => {
    it('calls GET /knowledge/rag/documents', async () => {
      mockApiGet.mockResolvedValue({
        documents: [{ metadata: {}, chunk_size: 512, num_chunks: 1, added_at: 0 }],
        stats: { total_documents: 1, total_chunks: 1, index_size: 5 },
      })
      const result = await listRAGDocuments()
      expect(mockApiGet).toHaveBeenCalledWith('/knowledge/rag/documents')
      expect(result.documents).toHaveLength(1)
    })
  })

  describe('getRAGStats', () => {
    it('calls GET /knowledge/rag/stats', async () => {
      mockApiGet.mockResolvedValue({ total_documents: 3, total_chunks: 15, index_size: 100 })
      const result = await getRAGStats()
      expect(mockApiGet).toHaveBeenCalledWith('/knowledge/rag/stats')
      expect(result.total_documents).toBe(3)
    })
  })

  describe('clearRAG', () => {
    it('calls POST /knowledge/rag/clear', async () => {
      mockApiPost.mockResolvedValue({ cleared: 5 })
      const result = await clearRAG()
      expect(mockApiPost).toHaveBeenCalledWith('/knowledge/rag/clear')
      expect(result.cleared).toBe(5)
    })
  })
})
