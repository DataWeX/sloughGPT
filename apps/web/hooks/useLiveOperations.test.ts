// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, cleanup } from '@testing-library/react'

let mockStart: ReturnType<typeof vi.fn>
let mockStop: ReturnType<typeof vi.fn>
let _onEvent: ((envelope: Record<string, unknown>) => void) | null = null
let _onClose: (() => void) | null = null
let _onError: (() => void) | null = null

vi.mock('@/lib/sse-client', () => ({
  createSSEStream: (opts: Record<string, unknown>) => {
    _onEvent = opts.onEvent as (e: Record<string, unknown>) => void
    _onClose = opts.onClose as () => void
    _onError = opts.onError as () => void
    return {
      start: mockStart,
      stop: mockStop,
      get connected() { return true },
    }
  },
}))

const mockFetch = vi.fn()
const mockSetState = vi.fn()
vi.mock('@/lib/operations-store', () => ({
  operationsStore: {
    getState: () => ({ fetch: mockFetch }),
    setState: (...args: unknown[]) => mockSetState(...args),
  },
  useOperationsStore: (selector: (s: Record<string, unknown>) => unknown) => {
    const state = {
      operations: [], counts: {}, loading: false, error: null,
      fetch: mockFetch,
      cancel: vi.fn().mockResolvedValue(true),
      cancelAll: vi.fn().mockResolvedValue(0),
    }
    return selector(state)
  },
}))

beforeEach(async () => {
  vi.clearAllMocks()
  vi.useFakeTimers()
  mockStart = vi.fn()
  mockStop = vi.fn()
  _onEvent = null
  _onClose = null
  _onError = null
  mockFetch.mockResolvedValue(undefined)
  vi.resetModules()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

async function importFresh() {
  return await import('./useLiveOperations')
}

describe('useLiveOperations', () => {
  it('calls initOperationsStream on mount', async () => {
    const { useLiveOperations } = await importFresh()
    renderHook(() => useLiveOperations())
    expect(mockStart).toHaveBeenCalled()
  })

  it('cleans up on unmount', async () => {
    const { useLiveOperations } = await importFresh()
    const { unmount } = renderHook(() => useLiveOperations())
    unmount()
    expect(mockStop).toHaveBeenCalled()
  })

  it('returns default state', async () => {
    const { useLiveOperations } = await importFresh()
    const { result } = renderHook(() => useLiveOperations())
    expect(result.current.operations).toEqual([])
    expect(result.current.isActive).toBe(false)
    expect(result.current.loading).toBe(false)
  })
})

describe('initOperationsStream', () => {
  it('creates SSE stream and starts it', async () => {
    const { initOperationsStream } = await importFresh()
    initOperationsStream()
    expect(mockStart).toHaveBeenCalled()
  })

  it('returns cleanup function that decrements ref', async () => {
    const { initOperationsStream } = await importFresh()
    const cleanup1 = initOperationsStream()
    const cleanup2 = initOperationsStream()
    cleanup1()
    expect(mockStop).not.toHaveBeenCalled()
    cleanup2()
    expect(mockStop).toHaveBeenCalled()
  })

  it('handles init event with operations', async () => {
    const { initOperationsStream } = await importFresh()
    initOperationsStream()
    _onEvent!({
      stream: 'operations',
      phase: 'init',
      data: {
        operations: [{ id: 'op1', type: 'training', status: 'running' }],
        counts: { training: 1 },
      },
    })
    expect(mockSetState).toHaveBeenCalledWith(
      expect.objectContaining({
        operations: expect.arrayContaining([
          expect.objectContaining({ id: 'op1' }),
        ]),
      }),
    )
  })

  it('handles registered event by fetching', async () => {
    const { initOperationsStream } = await importFresh()
    initOperationsStream()
    _onEvent!({ stream: 'operations', phase: 'registered', data: {} })
    expect(mockFetch).toHaveBeenCalled()
  })

  it('handles started event by fetching', async () => {
    const { initOperationsStream } = await importFresh()
    initOperationsStream()
    _onEvent!({ stream: 'operations', phase: 'started', data: {} })
    expect(mockFetch).toHaveBeenCalled()
  })

  it('handles finished event by fetching', async () => {
    const { initOperationsStream } = await importFresh()
    initOperationsStream()
    _onEvent!({ stream: 'operations', phase: 'finished', data: {} })
    expect(mockFetch).toHaveBeenCalled()
  })

  it('handles cancelled event by fetching', async () => {
    const { initOperationsStream } = await importFresh()
    initOperationsStream()
    _onEvent!({ stream: 'operations', phase: 'cancelled', data: {} })
    expect(mockFetch).toHaveBeenCalled()
  })

  it('ignores events from other streams', async () => {
    const { initOperationsStream } = await importFresh()
    initOperationsStream()
    _onEvent!({ stream: 'training', phase: 'init', data: {} })
    expect(mockSetState).not.toHaveBeenCalled()
  })

  it('starts fallback poll on close', async () => {
    const { initOperationsStream } = await importFresh()
    initOperationsStream()
    _onClose!()
    expect(mockFetch).toHaveBeenCalled()
    vi.advanceTimersByTime(5000)
    expect(mockFetch.mock.calls.length).toBeGreaterThanOrEqual(2)
  })

  it('starts fallback poll on error', async () => {
    const { initOperationsStream } = await importFresh()
    initOperationsStream()
    _onError!()
    expect(mockFetch).toHaveBeenCalled()
    vi.advanceTimersByTime(5000)
    expect(mockFetch.mock.calls.length).toBeGreaterThanOrEqual(2)
  })

  it('starts fallback poll after grace period if no init', async () => {
    const { initOperationsStream } = await importFresh()
    initOperationsStream()
    vi.advanceTimersByTime(5000)
    expect(mockFetch).toHaveBeenCalled()
  })
})
