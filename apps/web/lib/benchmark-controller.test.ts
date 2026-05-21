import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { benchmarkController } from './benchmark-controller'

describe('benchmarkController.run', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /benchmark/run', async () => {
    apiClient.apiPost.mockResolvedValue({ model: 'gpt2', perplexity: 12.5, latency_ms: 100, throughput: 50, num_parameters: 124_000_000, memory_mb: 500, throughput_tokens_per_sec: 1000, inference_time_ms: 90 })

    const result = await benchmarkController.run({ model: 'gpt2' })
    expect(result.model).toBe('gpt2')
    expect(result.perplexity).toBe(12.5)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/benchmark/run', { model: 'gpt2', dataset: undefined })
  })
})

describe('benchmarkController.history', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /benchmark/metrics with limit', async () => {
    apiClient.apiGet.mockResolvedValue({ results: [{ model: 'gpt2', latency_ms: 100, throughput: 50, num_parameters: 124_000_000, memory_mb: 500, throughput_tokens_per_sec: 1000, inference_time_ms: 90 }] })

    const result = await benchmarkController.history()
    expect(result).toHaveLength(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/benchmark/metrics?limit=10')
  })

  it('returns empty array on missing results', async () => {
    apiClient.apiGet.mockResolvedValue({})
    const result = await benchmarkController.history(5)
    expect(result).toEqual([])
    expect(apiClient.apiGet).toHaveBeenCalledWith('/benchmark/metrics?limit=5')
  })
})

describe('benchmarkController.quality', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /benchmark/quality', async () => {
    apiClient.apiGet.mockResolvedValue({ coherence_score: 0.9, quality_score: 0.85, repetition_rate: 0.05 })

    const result = await benchmarkController.quality()
    expect(result.coherence_score).toBe(0.9)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/benchmark/quality')
  })
})

describe('benchmarkController.stats', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /benchmark/stats', async () => {
    apiClient.apiGet.mockResolvedValue({ total: 10, avg_tokens: 128 })

    const result = await benchmarkController.stats()
    expect(result.total).toBe(10)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/benchmark/stats')
  })
})
