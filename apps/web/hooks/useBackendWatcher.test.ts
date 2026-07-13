/**
 * Tests for useBackendWatcher hook.
 */
import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, cleanup } from '@testing-library/react'

const mockSetStatus = vi.fn()
const mockSetHealthSummary = vi.fn()
const mockAddToast = vi.fn()

vi.mock('@/lib/api-monitor-store', () => ({
  useApiMonitor: (selector: (s: any) => any) => selector({ setStatus: mockSetStatus, setHealthSummary: mockSetHealthSummary }),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: { getState: () => ({ addToast: mockAddToast }) },
}))

const mockApiGet = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...a: any[]) => mockApiGet(...a),
}))

import { useBackendWatcher } from './useBackendWatcher'

const POLL_MS = 8000

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.useRealTimers()
})

describe('useBackendWatcher', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    mockApiGet.mockResolvedValue({ score: 100, status: 'healthy', summary: 'All good', model_loaded: true })
  })

  it('polls health/summary on mount and repeats', async () => {
    renderHook(() => useBackendWatcher())
    expect(mockApiGet).toHaveBeenCalledTimes(1)
    expect(mockApiGet.mock.calls[0][0]).toContain('/health/summary')
    await vi.advanceTimersByTimeAsync(POLL_MS + 100)
    expect(mockApiGet).toHaveBeenCalledTimes(2)
  })

  it('sets connected when fetch succeeds', async () => {
    renderHook(() => useBackendWatcher())
    await vi.advanceTimersByTimeAsync(100)
    expect(mockSetStatus).toHaveBeenCalledWith('connected')
  })

  it('never sets reloading for 429 rate limit errors', async () => {
    const rateLimitError = Object.assign(new Error('Too many requests'), { status: 429 })
    mockApiGet.mockRejectedValue(rateLimitError)
    renderHook(() => useBackendWatcher())
    // 10 polls with backoff — should never reload on rate limit
    // First few polls: 8s, then 10s, 12s, 16s, 24s, 40s, ...
    for (let i = 0; i < 10; i++) {
      await vi.advanceTimersByTimeAsync(POLL_MS * 5)
    }
    expect(mockSetStatus).not.toHaveBeenCalledWith('reloading')
    expect(mockSetStatus).not.toHaveBeenCalledWith('connecting')
  })

  it('resets failure count after success following failures', async () => {
    // Chain: success, fail, success
    mockApiGet
      .mockResolvedValueOnce({ score: 100, status: 'healthy', summary: 'Init', model_loaded: true })
      .mockRejectedValueOnce(new Error('down'))
      .mockResolvedValueOnce({ score: 100, status: 'healthy', summary: 'Back', model_loaded: true })
    renderHook(() => useBackendWatcher())
    // Call 1: success immediately on mount
    expect(mockApiGet).toHaveBeenCalledTimes(1)
    // Advance past first poll (8s base, no backoff since failureCount=0)
    await vi.advanceTimersByTimeAsync(POLL_MS + 100)
    // Call 2: failure
    expect(mockApiGet).toHaveBeenCalledTimes(2)
    // Advance past second poll — backoff is 2000ms (failureCount=1) so poll at 10s
    await vi.advanceTimersByTimeAsync(POLL_MS + 2000 + 100)
    // Call 3: success — resets failure count
    expect(mockApiGet).toHaveBeenCalledTimes(3)
    expect(mockSetStatus).toHaveBeenCalledWith('connected')
    expect(mockSetStatus).not.toHaveBeenCalledWith('reloading')
  })

  it('cleans up timer on unmount', async () => {
    const { unmount } = renderHook(() => useBackendWatcher())
    unmount()
    const callCount = mockApiGet.mock.calls.length
    await vi.advanceTimersByTimeAsync(POLL_MS * 2)
    expect(mockApiGet.mock.calls.length).toBe(callCount)
  })

  it('does not reload after single failure', async () => {
    mockApiGet.mockRejectedValue(new Error('down'))
    renderHook(() => useBackendWatcher())
    await vi.advanceTimersByTimeAsync(POLL_MS + 100)
    expect(mockSetStatus).toHaveBeenCalledWith('connecting')
    expect(mockSetStatus).not.toHaveBeenCalledWith('reloading')
  })
})
