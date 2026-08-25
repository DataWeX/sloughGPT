import { describe, it, expect, vi, beforeEach } from 'vitest'
import { loraEvalController } from './lora-eval-controller'

vi.mock('./http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))

import { apiGet, apiPost } from './http-client'

const mockGet = vi.mocked(apiGet)
const mockPost = vi.mocked(apiPost)

beforeEach(() => {
  vi.clearAllMocks()
})

describe('loraEvalController', () => {
  describe('runEval', () => {
    it('calls run endpoint with adapter path and soul', async () => {
      mockGet.mockResolvedValue({ status: 'compared' })
      const result = await loraEvalController.runEval('data/user_adapters/best.npz', 'assistant')
      expect(mockGet).toHaveBeenCalledWith(expect.stringContaining('/lora-eval/run?adapter_path='))
      expect(result.status).toBe('compared')
    })

    it('works without soul param', async () => {
      mockGet.mockResolvedValue({ status: 'baseline_only' })
      await loraEvalController.runEval('data/user_adapters/best.npz')
      expect(mockGet).toHaveBeenCalledWith(expect.stringContaining('/lora-eval/run?adapter_path='))
    })
  })

  describe('getHistory', () => {
    it('returns results from nested data', async () => {
      mockGet.mockResolvedValue({ results: [{ adapter_path: 'a.npz', verdict: 'good' }] })
      const results = await loraEvalController.getHistory(5)
      expect(results).toHaveLength(1)
      expect(results[0].verdict).toBe('good')
      expect(mockGet).toHaveBeenCalledWith('/lora-eval/history?limit=5')
    })

    it('returns flat array', async () => {
      mockGet.mockResolvedValue([{ adapter_path: 'b.npz', verdict: 'ok' }])
      const results = await loraEvalController.getHistory()
      expect(results).toHaveLength(1)
    })

    it('returns empty array on missing results', async () => {
      mockGet.mockResolvedValue({})
      const results = await loraEvalController.getHistory()
      expect(results).toEqual([])
    })

    it('defaults limit to 10', async () => {
      mockGet.mockResolvedValue({ results: [] })
      await loraEvalController.getHistory()
      expect(mockGet).toHaveBeenCalledWith('/lora-eval/history?limit=10')
    })
  })

  describe('aggregate', () => {
    it('calls aggregate endpoint with params', async () => {
      mockPost.mockResolvedValue({ status: 'aggregated_with_eval' })
      const result = await loraEvalController.aggregate(10, 5)
      expect(mockPost).toHaveBeenCalledWith(expect.stringContaining('/lora-eval/aggregate?'))
      expect(result.status).toBe('aggregated_with_eval')
    })
  })
})
