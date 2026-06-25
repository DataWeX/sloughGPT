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
  })

  it('polls health/summary on mount and repeats', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ score: 100, status: 'healthy', summary: 'All good', model_loaded: true }),
    })
    renderHook(() => useBackendWatcher())
    expect(mockFetch).toHaveBeenCalledTimes(1)
    expect(mockFetch.mock.calls[0][0]).toContain('/health/summary')
    await vi.advanceTimersByTimeAsync(POLL_MS + 100)
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

  it('defers reloading until 3 consecutive failures', async () => {
    mockFetch.mockRejectedValue(new Error('network down'))
    renderHook(() => useBackendWatcher())
    // Immediate check = failure 1. First timer advance = failure 2 (no reloading yet).
    await vi.advanceTimersByTimeAsync(POLL_MS)
    expect(mockSetStatus).not.toHaveBeenCalledWith('reloading')
    // Second timer advance = failure 3 → reloading
    mockSetStatus.mockClear()
    await vi.advanceTimersByTimeAsync(POLL_MS)
    expect(mockSetStatus).toHaveBeenCalledWith('reloading')
  })

  it('does not set reloading after success following failures', async () => {
    mockFetch
      .mockRejectedValueOnce(new Error('down'))
      .mockRejectedValueOnce(new Error('down'))
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ score: 100, status: 'healthy', summary: 'Back', model_loaded: true }),
      })
      .mockRejectedValue(new Error('down'))
    renderHook(() => useBackendWatcher())
    // 2 failures (not enough for reloading), then success
    await vi.advanceTimersByTimeAsync(POLL_MS * 2 + 100)
    expect(mockFetch).toHaveBeenCalledTimes(3)
    // The 3rd call (success) should have set 'connected'
    expect(mockSetStatus).toHaveBeenCalledWith('connected')
    // And NOT set 'reloading'
    expect(mockSetStatus).not.toHaveBeenCalledWith('reloading')
  })

  it('shows reconnect toast after offline recovery', async () => {
    mockFetch
      .mockRejectedValue(new Error('down'))
    renderHook(() => useBackendWatcher())
    // 3 consecutive failures
    await vi.advanceTimersByTimeAsync(POLL_MS * 3 + 100)
    expect(mockSetStatus).toHaveBeenCalledWith('reloading')
    expect(mockSetStatus).not.toHaveBeenCalledWith('connected')

    mockSetStatus.mockClear()
    mockFetch.mockReset()
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ score: 100, status: 'healthy', summary: 'Ok', model_loaded: true }),
    })
    // Next poll succeeds
    await vi.advanceTimersByTimeAsync(POLL_MS)
    expect(mockSetStatus).toHaveBeenCalledWith('connected')
    expect(mockAddToast).toHaveBeenCalledWith('Server reconnected', 'success')
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
    await vi.advanceTimersByTimeAsync(POLL_MS + 100)
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
    await vi.advanceTimersByTimeAsync(POLL_MS * 2)
    expect(mockFetch.mock.calls.length).toBe(callCount)
  })

  it('sets reloading after 3 non-ok responses', async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 503 })
    renderHook(() => useBackendWatcher())
    await vi.advanceTimersByTimeAsync(POLL_MS * 3 + 100)
    expect(mockSetStatus).toHaveBeenCalledWith('reloading')
  })
})
