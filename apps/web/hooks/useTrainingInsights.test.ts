import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, waitFor, cleanup } from '@testing-library/react'

const { mockInsights, mockExport, mockUpdate } = vi.hoisted(() => ({
  mockInsights: vi.fn(),
  mockExport: vi.fn(),
  mockUpdate: vi.fn(),
}))

vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    getAdaptiveInsights: mockInsights,
    exportTrainingHistory: mockExport,
    updateTraining: mockUpdate,
  },
}))

import { useTrainingInsights } from './useTrainingInsights'

describe('useTrainingInsights', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockInsights.mockResolvedValue({ total_runs: 2, best_config: { model: 'gpt2' } })
    mockExport.mockResolvedValue({
      outcomes: [
        { run_id: 'a', timestamp: 1767225600, quality_score: 0.8, final_loss: 2.0, model: 'gpt2' },
        { run_id: 'b', timestamp: 1767225700, quality_score: 0.9, final_loss: 1.5, model: 'gpt2' },
      ],
    })
  })

  afterEach(() => { cleanup() })

  it('loads insights and chart points', async () => {
    const { result } = renderHook(() => useTrainingInsights())
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.insights?.total_runs).toBe(2)
    expect(result.current.trendPoints).toHaveLength(2)
    expect(result.current.trendPoints[0].name).toBe('#1')
    expect(result.current.trendPoints[1].loss).toBe(1.5)
  })

  it('nulls loss when non-positive', async () => {
    mockExport.mockResolvedValue({ outcomes: [{ run_id: 'a', timestamp: 1, quality_score: 0.5, final_loss: 0, model: 'm' }] })
    const { result } = renderHook(() => useTrainingInsights())
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.trendPoints[0].loss).toBeNull()
  })

  it('applyRecommendedConfig pushes best config and resets status', async () => {
    const { result } = renderHook(() => useTrainingInsights())
    await waitFor(() => expect(result.current.loading).toBe(false))
    await result.current.applyRecommendedConfig()
    expect(mockUpdate).toHaveBeenCalledWith(expect.objectContaining({ preferred_method: 'finetune' }))
    await waitFor(() => expect(result.current.useConfigStatus).toBe('applied'))
  })
})
