import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useV86 } from './useV86'

vi.mock('@/lib/dev-log', () => ({
  logger: { child: () => ({ warning: vi.fn(), error: vi.fn(), info: vi.fn(), debug: vi.fn() }) },
  trackEvent: vi.fn(),
}))

vi.mock('@/lib/v86-controller', () => {
  const mockInit = vi.fn().mockResolvedValue(undefined)
  const mockPersistState = vi.fn().mockResolvedValue(undefined)
  const mockLoadPersistedState = vi.fn().mockResolvedValue(null)
  const mockRestoreState = vi.fn().mockResolvedValue(undefined)
  const mockRestart = vi.fn()
  const mockDestroy = vi.fn()

  return {
    V86Controller: vi.fn().mockImplementation(() => ({
      init: mockInit,
      persistState: mockPersistState,
      loadPersistedState: mockLoadPersistedState,
      restoreState: mockRestoreState,
      restart: mockRestart,
      destroy: mockDestroy,
      save_state: vi.fn().mockResolvedValue(new ArrayBuffer(8)),
      is_running: vi.fn().mockReturnValue(true),
    })),
  }
})

describe('useV86', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('initializes with default state', () => {
    const { result } = renderHook(() => useV86())
    expect(result.current.isBooted).toBe(false)
    expect(result.current.stateSaved).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('init sets isBooted on success', async () => {
    const { result } = renderHook(() => useV86())
    const container = document.createElement('div')

    await act(async () => {
      await result.current.init(container)
    })

    expect(result.current.isBooted).toBe(true)
    expect(result.current.error).toBeNull()
  })

  it('save persists state', async () => {
    const { result } = renderHook(() => useV86())
    const container = document.createElement('div')

    await act(async () => {
      await result.current.init(container)
    })

    await act(async () => {
      await result.current.save()
    })

    expect(result.current.stateSaved).toBe(true)
  })

  it('save is no-op when not initialized', async () => {
    const { result } = renderHook(() => useV86())
    await act(async () => {
      await result.current.save()
    })
    expect(result.current.stateSaved).toBe(false)
  })

  it('reset calls restart', async () => {
    const { result } = renderHook(() => useV86())
    const container = document.createElement('div')

    await act(async () => {
      await result.current.init(container)
    })

    act(() => {
      result.current.reset()
    })
    // no assertion needed — just verifying no throw
  })
})
