/**
 */
import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, cleanup } from '@testing-library/react'

let currentStatus = 'initial'
const mockSetStatus = vi.fn((s: string) => { currentStatus = s })
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
    currentStatus = 'initial'
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

  it('defers reloading until 3 consecutive failures', async () => {
    mockApiGet.mockRejectedValue(new Error('network down'))
    renderHook(() => useBackendWatcher())
    await vi.advanceTimersByTimeAsync(POLL_MS)
    expect(mockSetStatus).not.toHaveBeenCalledWith('reloading')
    mockSetStatus.mockClear()
    await vi.advanceTimersByTimeAsync(POLL_MS)
    expect(mockSetStatus).toHaveBeenCalledWith('reloading')
  })

  it('does not set reloading after success following failures', async () => {
    mockApiGet
      .mockRejectedValueOnce(new Error('down'))
      .mockRejectedValueOnce(new Error('down'))
      .mockResolvedValueOnce({ score: 100, status: 'healthy', summary: 'Back', model_loaded: true })
      .mockRejectedValue(new Error('down'))
    renderHook(() => useBackendWatcher())
    await vi.advanceTimersByTimeAsync(POLL_MS * 2 + 100)
    expect(mockApiGet).toHaveBeenCalledTimes(3)
    expect(mockSetStatus).toHaveBeenCalledWith('connected')
    expect(mockSetStatus).not.toHaveBeenCalledWith('reloading')
  })

  it('schedules page reload after 3 consecutive failures', async () => {
    const reloadSpy = vi.fn()
    Object.defineProperty(window, 'location', {
      value: { reload: reloadSpy },
      writable: true,
    })
    mockApiGet.mockRejectedValue(new Error('down'))
    renderHook(() => useBackendWatcher())
    await vi.advanceTimersByTimeAsync(POLL_MS * 3 + 100)
    expect(mockSetStatus).toHaveBeenCalledWith('reloading')
    // Should have scheduled a page reload but not resumed polling
    await vi.advanceTimersByTimeAsync(POLL_MS)
    expect(mockApiGet).toHaveBeenCalledTimes(3)
  })

  it('shows toast on first degraded status transition', async () => {
    mockApiGet
      .mockResolvedValueOnce({ score: 100, status: 'healthy', summary: 'Ok', model_loaded: true })
      .mockResolvedValueOnce({ score: 60, status: 'degraded', summary: 'High latency', model_loaded: true })
    renderHook(() => useBackendWatcher())
    await vi.advanceTimersByTimeAsync(POLL_MS + 100)
    expect(mockAddToast).toHaveBeenCalledWith('High latency', 'info')
  })

  it('cleans up timer on unmount', async () => {
    const { unmount } = renderHook(() => useBackendWatcher())
    unmount()
    const callCount = mockApiGet.mock.calls.length
    await vi.advanceTimersByTimeAsync(POLL_MS * 2)
    expect(mockApiGet.mock.calls.length).toBe(callCount)
  })

  it('sets reloading after 3 non-ok responses', async () => {
    mockApiGet.mockRejectedValue(new Error('network down'))
    renderHook(() => useBackendWatcher())
    await vi.advanceTimersByTimeAsync(POLL_MS * 3 + 100)
    expect(mockSetStatus).toHaveBeenCalledWith('reloading')
  })
})
