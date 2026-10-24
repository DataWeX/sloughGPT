import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'

const { mockAnalytics } = vi.hoisted(() => ({ mockAnalytics: vi.fn() }))

vi.mock('@/lib/settings-controller', () => ({
  settingsController: { getTrainingAnalytics: mockAnalytics },
}))

import { useTrainingAnalytics } from './useTrainingAnalytics'

describe('useTrainingAnalytics', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockAnalytics.mockResolvedValue({ total_runs: 3 })
  })

  it('loads analytics on mount', async () => {
    const { result } = renderHook(() => useTrainingAnalytics())
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.analytics).toEqual({ total_runs: 3 })
    expect(result.current.error).toBeNull()
  })

  it('surfaces errors', async () => {
    mockAnalytics.mockRejectedValue(new Error('boom'))
    const { result } = renderHook(() => useTrainingAnalytics())
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.error).toBe('boom')
    expect(result.current.analytics).toBeNull()
  })
})
