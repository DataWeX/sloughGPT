import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { useConsciousnessStats } from './useConsciousnessStats'

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
}))

import { apiGet } from '@/lib/http-client'
const mockApiGet = vi.mocked(apiGet)

describe('useConsciousnessStats', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('returns loading state initially', () => {
    mockApiGet.mockRejectedValue(new Error('not ready'))
    const { result } = renderHook(() => useConsciousnessStats())
    expect(result.current.loading).toBe(true)
    expect(result.current.stats).toBeNull()
    expect(result.current.error).toBeNull()
  })

  it('fetches stats on mount', async () => {
    const mockStats = { total_episodes: 42, avg_qualia: { valence: 0.6 } }
    mockApiGet.mockResolvedValue(mockStats)

    const { result } = renderHook(() => useConsciousnessStats())
    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })
    expect(result.current.stats).toEqual(mockStats)
    expect(result.current.error).toBeNull()
  })

  it('returns raw json when data key missing', async () => {
    const mockStats = { total_episodes: 42 }
    mockApiGet.mockResolvedValue(mockStats)

    const { result } = renderHook(() => useConsciousnessStats())
    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })
    expect(result.current.stats).toEqual(mockStats)
  })

  it('handles non-ok response', async () => {
    mockApiGet.mockRejectedValue(new Error('Failed to fetch stats: 500'))

    const { result } = renderHook(() => useConsciousnessStats())
    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })
    expect(result.current.stats).toBeNull()
    expect(result.current.error).toBe('Failed to fetch stats: 500')
  })

  it('handles network error', async () => {
    mockApiGet.mockRejectedValue(new Error('network'))

    const { result } = renderHook(() => useConsciousnessStats())
    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })
    expect(result.current.stats).toBeNull()
    expect(result.current.error).toBe('network')
  })

  it('refetch reloads stats', async () => {
    const mockStats = { total_episodes: 10 }
    const mockUpdated = { total_episodes: 20 }
    mockApiGet.mockResolvedValueOnce(mockStats)
    mockApiGet.mockResolvedValueOnce(mockUpdated)

    const { result } = renderHook(() => useConsciousnessStats())
    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })
    expect(result.current.stats).toEqual(mockStats)

    await waitFor(async () => {
      await result.current.refetch()
    })
    await waitFor(() => {
      expect(result.current.stats).toEqual(mockUpdated)
    })
  })
})
