import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'

const { mockApiGet } = vi.hoisted(() => ({ mockApiGet: vi.fn() }))

vi.mock('@/lib/http-client', () => ({
  apiGet: (...a: any[]) => mockApiGet(...a),
}))

import { useTrainingTrends } from './useTrainingTrends'

const TRENDS = {
  runs: [{ run_id: 'run-abcdef123456', timestamp: 1767225600, model: 'gpt2', dataset: 'd', method: 'finetune', quality_score: 0.8, final_loss: 2.1, perplexity: 8, converged: true, early_stopped: false }],
  models: [{ model: 'gpt2', total_runs: 1, avg_quality: 0.8, best_quality: 0.8, avg_loss: 2.1, latest_quality: 0.8, improving: false }],
  summary: { total_runs: 1, avg_quality: 0.8, best_quality: 0.8, avg_loss: 2.1, trend: 'stable' as const },
}

describe('useTrainingTrends', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockResolvedValue(TRENDS)
  })

  it('loads trends and derives chart points', async () => {
    const { result } = renderHook(() => useTrainingTrends())
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(mockApiGet).toHaveBeenCalledWith('/training/trends')
    expect(result.current.runs).toHaveLength(1)
    expect(result.current.chartPoints[0].name).toBe('gpt2#123456')
    expect(result.current.chartPoints[0].loss).toBe(2.1)
  })

  it('passes model/dataset filters as query params', async () => {
    const { result } = renderHook(() => useTrainingTrends())
    await waitFor(() => expect(result.current.loading).toBe(false))
    result.current.setModelFilter('gpt2')
    await waitFor(() => expect(mockApiGet).toHaveBeenCalledWith('/training/trends?model=gpt2'))
  })
})
