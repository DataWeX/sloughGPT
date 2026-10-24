import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import React from 'react'
import { compareMetric } from './useRunCompare'

const { mockExport, mockCompare } = vi.hoisted(() => ({
  mockExport: vi.fn(),
  mockCompare: vi.fn(),
}))

vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    exportTrainingHistory: mockExport,
    compareTrainingRuns: mockCompare,
  },
}))

import { useRunCompare } from './useRunCompare'

describe('compareMetric', () => {
  it('treats loss/perplexity/time as lower-is-better', () => {
    expect(compareMetric('final_loss', 1, 2)).toBe('a')
    expect(compareMetric('perplexity', 9, 8)).toBe('b')
    expect(compareMetric('training_time_s', 5, 5)).toBe('tie')
  })

  it('treats quality as higher-is-better', () => {
    expect(compareMetric('quality_score', 0.9, 0.8)).toBe('a')
    expect(compareMetric('quality_score', 0.7, 0.8)).toBe('b')
    expect(compareMetric('quality_score', 0.8, 0.8)).toBe('tie')
  })
})

describe('useRunCompare', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockExport.mockResolvedValue({ outcomes: [{ run_id: 'a', model: 'gpt2', quality_score: 0.9 }] })
  })

  it('loads run options on mount', async () => {
    const { result } = renderHook(() => useRunCompare())
    await waitFor(() => expect(result.current.fetchingOptions).toBe(false))
    expect(result.current.runOptions).toEqual([{ run_id: 'a', model: 'gpt2', quality_score: 0.9 }])
  })

  it('comparePair stores the result', async () => {
    const res = { run_a: { run_id: 'a' }, run_b: { run_id: 'b' }, differences: {}, a_wins: 2, b_wins: 1 }
    mockCompare.mockResolvedValue(res)
    const { result } = renderHook(() => useRunCompare())
    await waitFor(() => expect(result.current.fetchingOptions).toBe(false))
    let out: unknown = null
    await result.current.comparePair('a', 'b').then(r => { out = r })
    expect(mockCompare).toHaveBeenCalledWith('a', 'b')
    expect(out).toEqual(res)
    await waitFor(() => expect(result.current.result).toEqual(res))
  })

  it('comparePair returns null without both ids', async () => {
    const { result } = renderHook(() => useRunCompare())
    await waitFor(() => expect(result.current.fetchingOptions).toBe(false))
    expect(await result.current.comparePair('', 'b')).toBeNull()
    expect(mockCompare).not.toHaveBeenCalled()
  })
})
