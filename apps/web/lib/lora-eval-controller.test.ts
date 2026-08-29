import { describe, it, expect, vi, beforeEach } from 'vitest'
import { loraEvalController } from './lora-eval-controller'

const mockApiGet = vi.fn()
const mockApiPost = vi.fn()

vi.mock('./http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
}))

import { apiGet, apiPost } from './http-client'

beforeEach(() => {
  vi.clearAllMocks()
})

describe('loraEvalController', () => {
  describe('runEval', () => {
    it('calls run endpoint with adapter path', async () => {
      mockApiPost.mockResolvedValue({ status: 'ok' })
      await loraEvalController.runEval('data/user_adapters/best.npz')
      expect(mockApiPost).toHaveBeenCalledWith('/lora-eval/run', { adapter_path: 'data/user_adapters/best.npz', soul: undefined })
    })
  })

  describe('getHistory', () => {
    it('returns results from nested data', async () => {
      mockApiGet.mockResolvedValue({
        results: [{ adapter_path: 'a.npz', verdict: 'good' }],
      })
      const results = await loraEvalController.getHistory(5)
      expect(results).toHaveLength(1)
      expect(results[0].verdict).toBe('good')
      expect(mockApiGet).toHaveBeenCalledWith('/lora-eval/history?limit=5')
    })

    it('returns flat results', async () => {
      mockApiGet.mockResolvedValue({ results: [{ adapter_path: 'b.npz', verdict: 'ok' }] })
      const results = await loraEvalController.getHistory()
      expect(results).toHaveLength(1)
    })

    it('returns empty array on missing results', async () => {
      mockApiGet.mockResolvedValue({})
      const results = await loraEvalController.getHistory()
      expect(results).toEqual([])
    })

    it('defaults limit to 10', async () => {
      mockApiGet.mockResolvedValue({ results: [] })
      await loraEvalController.getHistory()
      expect(mockApiGet).toHaveBeenCalledWith('/lora-eval/history?limit=10')
    })

    it('encodes special chars in path', async () => {
      mockApiPost.mockResolvedValue({ results: [] })
      await loraEvalController.runEval('path with spaces/file.npz')
      expect(mockApiPost).toHaveBeenCalledWith('/lora-eval/run', { adapter_path: 'path with spaces/file.npz', soul: undefined })
    })
  })
})
