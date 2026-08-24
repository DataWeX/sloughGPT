import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, cleanup, waitFor } from '@testing-library/react'
import { useConversionStatus, formatStage } from './useConversionStatus'

const mockApiGet = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
}))

describe('useConversionStatus', () => {
  afterEach(cleanup)

  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers({ shouldAdvanceTime: true })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('returns null status when modelId is null', () => {
    const { result } = renderHook(() => useConversionStatus(null))
    expect(result.current.status).toBeNull()
  })

  it('fetches status on mount when modelId provided', async () => {
    mockApiGet.mockResolvedValueOnce({ data: { stage: 'downloading', progress: 50 } })

    const { result } = renderHook(() => useConversionStatus('my-model'))

    await waitFor(() => {
      expect(result.current.status).not.toBeNull()
    })

    expect(mockApiGet).toHaveBeenCalled()
    expect(result.current.status?.stage).toBe('downloading')
    expect(result.current.status?.progress).toBe(50)
  })

  it('polls every 2000ms', async () => {
    mockApiGet.mockResolvedValue({ data: { stage: 'converting', progress: 30 } })

    renderHook(() => useConversionStatus('my-model'))

    await vi.advanceTimersByTimeAsync(2200)
    expect(mockApiGet.mock.calls.length).toBeGreaterThanOrEqual(2)

    await vi.advanceTimersByTimeAsync(2200)
    expect(mockApiGet.mock.calls.length).toBeGreaterThanOrEqual(3)
  })

  it('stops polling when stage is ready', async () => {
    mockApiGet.mockResolvedValue({ data: { stage: 'ready', progress: 100 } })

    renderHook(() => useConversionStatus('my-model'))

    await vi.advanceTimersByTimeAsync(100)
    const countAfterFirst = mockApiGet.mock.calls.length

    await vi.advanceTimersByTimeAsync(2000)
    expect(mockApiGet.mock.calls.length).toBe(countAfterFirst)
  })

  it('stops polling when stage is error', async () => {
    mockApiGet.mockResolvedValue({ data: { stage: 'error', progress: 0, error: 'fail' } })

    renderHook(() => useConversionStatus('my-model'))

    await vi.advanceTimersByTimeAsync(100)
    const countAfterFirst = mockApiGet.mock.calls.length

    await vi.advanceTimersByTimeAsync(2000)
    expect(mockApiGet.mock.calls.length).toBe(countAfterFirst)
  })

  it('handles fetch error gracefully', async () => {
    mockApiGet.mockRejectedValueOnce(new Error('network'))

    const { result } = renderHook(() => useConversionStatus('my-model'))

    await vi.advanceTimersByTimeAsync(200)
    expect(result.current.status).toBeNull()
  })

  it('unwraps nested data field', async () => {
    mockApiGet.mockResolvedValueOnce({ data: { stage: 'loading', progress: 80 } })

    const { result } = renderHook(() => useConversionStatus('my-model'))

    await waitFor(() => {
      expect(result.current.status?.stage).toBe('loading')
    })

    expect(result.current.status?.progress).toBe(80)
  })

  it('cleans up interval on unmount', async () => {
    mockApiGet.mockResolvedValue({ data: { stage: 'converting', progress: 10 } })

    const { unmount } = renderHook(() => useConversionStatus('my-model'))

    await vi.advanceTimersByTimeAsync(100)
    const countAfterFirst = mockApiGet.mock.calls.length

    unmount()

    await vi.advanceTimersByTimeAsync(2000)
    expect(mockApiGet.mock.calls.length).toBe(countAfterFirst)
  })

  it('clears status when modelId changes to null', async () => {
    mockApiGet.mockResolvedValue({ data: { stage: 'converting', progress: 50 } })

    const { result, rerender } = renderHook(
      ({ id }: { id: string | null }) => useConversionStatus(id),
      { initialProps: { id: 'model-a' } as { id: string | null } }
    )

    await waitFor(() => {
      expect(result.current.status?.stage).toBe('converting')
    })

    rerender({ id: null })
    expect(result.current.status).toBeNull()
  })
})

describe('formatStage', () => {
  it('returns label for known stages', () => {
    expect(formatStage('idle')).toBe('Preparing')
    expect(formatStage('downloading')).toBe('Downloading')
    expect(formatStage('converting')).toBe('Converting to .slnc')
    expect(formatStage('protecting')).toBe('Protecting files')
    expect(formatStage('loading')).toBe('Loading into memory')
    expect(formatStage('ready')).toBe('Ready')
    expect(formatStage('error')).toBe('Error')
  })

  it('returns raw stage for unknown values', () => {
    expect(formatStage('unknown')).toBe('unknown')
    expect(formatStage('')).toBe('')
  })
})
