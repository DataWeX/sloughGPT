import { describe, it, expect, vi, beforeEach } from 'vitest'
import { loraEvalController } from './lora-eval-controller'

vi.mock('./http-client', () => ({
  apiGet: vi.fn(),
}))

import { apiGet } from './http-client'

const mockGet = vi.mocked(apiGet)

beforeEach(() => {
  vi.clearAllMocks()
})

describe('loraEvalController', () => {
  describe('runEval', () => {
    it('calls run endpoint with adapter path', async () => {
      mockGet.mockResolvedValue({ status: 'ok' })
      await loraEvalController.runEval('data/user_adapters/best.npz')
      expect(mockGet).toHaveBeenCalledWith('/lora-eval/run?adapter_path=data%2Fuser_adapters%2Fbest.npz')
    })
  })

  describe('getHistory', () => {
    it('returns results from nested data', async () => {
      mockGet.mockResolvedValue({
        results: [{ adapter_path: 'a.npz', verdict: 'good' }],
      })
      const results = await loraEvalController.getHistory(5)
      expect(results).toHaveLength(1)
      expect(results[0].verdict).toBe('good')
      expect(mockGet).toHaveBeenCalledWith('/lora-eval/history?limit=5')
    })

    it('returns flat results', async () => {
      mockGet.mockResolvedValue({ results: [{ adapter_path: 'b.npz', verdict: 'ok' }] })
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

    it('encodes special chars in path', async () => {
      mockGet.mockResolvedValue({ results: [] })
      await loraEvalController.runEval('path with spaces/file.npz')
      expect(mockGet).toHaveBeenCalledWith('/lora-eval/run?adapter_path=path+with+spaces%2Ffile.npz')
    })
  })
})
