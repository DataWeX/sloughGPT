/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, cleanup } from '@testing-library/react'

let currentStatus = 'initial'
const mockSetStatus = vi.fn((s: string) => { currentStatus = s })
const mockAddToast = vi.fn()

vi.mock('@/lib/api-monitor-store', () => ({
  useApiMonitor: (selector: (s: any) => any) => selector({ setStatus: mockSetStatus }),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: { getState: () => ({ addToast: mockAddToast }) },
}))

import { useBackendWatcher } from './useBackendWatcher'

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.useRealTimers()
})

describe('useBackendWatcher', () => {
  beforeEach(() => {
    currentStatus = 'initial'
    vi.useFakeTimers()
  })

  it('polls health/summary on mount and sets connected', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ score: 100, status: 'healthy', summary: 'All good', model_loaded: true }),
    })
    renderHook(() => useBackendWatcher())
    expect(mockFetch).toHaveBeenCalledTimes(1)
    expect(mockFetch.mock.calls[0][0]).toContain('/health/summary')
    await vi.advanceTimersByTimeAsync(3500)
    expect(mockFetch).toHaveBeenCalledTimes(2)
  })

  it('sets connected when fetch succeeds', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ score: 100, status: 'healthy', summary: 'All good', model_loaded: true }),
    })
    renderHook(() => useBackendWatcher())
    await vi.advanceTimersByTimeAsync(100)
    expect(mockSetStatus).toHaveBeenCalledWith('connected')
  })

  it('sets reloading when fetch fails', async () => {
    mockFetch.mockRejectedValue(new Error('network down'))
    renderHook(() => useBackendWatcher())
    await vi.advanceTimersByTimeAsync(100)
    expect(mockSetStatus).toHaveBeenCalledWith('reloading')
  })

  it('sets connected after offline recovery', async () => {
    mockFetch.mockRejectedValueOnce(new Error('down'))
    const { rerender } = renderHook(() => useBackendWatcher())
    await vi.advanceTimersByTimeAsync(100)
    expect(mockSetStatus).toHaveBeenCalledWith('reloading')
    mockSetStatus.mockClear()
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ score: 100, status: 'healthy', summary: 'Back', model_loaded: true }),
    })
    await vi.advanceTimersByTimeAsync(3500)
    await vi.advanceTimersByTimeAsync(100)
    expect(mockSetStatus).toHaveBeenCalledWith('connected')
  })

  it('shows toast on first degraded status transition', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ score: 100, status: 'healthy', summary: 'Ok', model_loaded: true }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ score: 60, status: 'degraded', summary: 'High latency', model_loaded: true }),
      })
    renderHook(() => useBackendWatcher())
    await vi.advanceTimersByTimeAsync(100)
    await vi.advanceTimersByTimeAsync(3500)
    await vi.advanceTimersByTimeAsync(100)
    expect(mockAddToast).toHaveBeenCalledWith('High latency', 'info')
  })

  it('cleans up timer on unmount', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ score: 100, status: 'healthy', summary: 'Ok', model_loaded: true }),
    })
    const { unmount } = renderHook(() => useBackendWatcher())
    unmount()
    const callCount = mockFetch.mock.calls.length
    await vi.advanceTimersByTimeAsync(10000)
    expect(mockFetch.mock.calls.length).toBe(callCount)
  })

  it('sets reloading when response not ok', async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 503 })
    renderHook(() => useBackendWatcher())
    await vi.advanceTimersByTimeAsync(100)
    expect(mockSetStatus).toHaveBeenCalledWith('reloading')
  })
})
