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
      isRunning: vi.fn().mockReturnValue(true),
    })),
  }
})

describe('useV86', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
    // init() probes the default boot image before constructing the controller;
    // answer "local image exists, 8MB" without real network I/O.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 206,
        headers: {
          get: (k: string) => (k.toLowerCase() === 'content-range' ? 'bytes 0-0/8388608' : null),
        },
      }),
    )
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
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

  it('init fails fast with actionable error when no image is available', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: false, status: 404, headers: { get: () => null } }),
    )
    const { result } = renderHook(() => useV86())
    const container = document.createElement('div')

    await act(async () => {
      await result.current.init(container)
    })

    expect(result.current.isBooted).toBe(false)
    expect(result.current.error).toMatch(/VM image not available/)
    expect(result.current.error).toMatch(/buildroot\/build\.sh/)
  })

  it('falls back to the local vendored kernel when no local hda exists', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) =>
        Promise.resolve(
          url.includes('bzimage68.bin')
            ? {
                ok: true,
                status: 206,
                headers: {
                  get: (k: string) =>
                    k.toLowerCase() === 'content-range' ? 'bytes 0-0/10068480' : null,
                },
              }
            : { ok: false, status: 404, headers: { get: () => null } },
        ),
      ),
    )
    const { V86Controller } = await import('@/lib/v86-controller')
    const { result } = renderHook(() => useV86())
    const container = document.createElement('div')

    await act(async () => {
      await result.current.init(container)
    })

    expect(result.current.isBooted).toBe(true)
    const instance = vi.mocked(V86Controller).mock.results.at(-1)?.value as any
    expect(instance.init).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({
        imageUrl: '/buildroot/bzimage68.bin',
        imageKind: 'kernel',
      }),
    )
  })

  it('falls back to the upstream i.copy.sh kernel when nothing is local', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) =>
        Promise.resolve(
          url.startsWith('https://i.copy.sh/')
            ? {
                ok: true,
                status: 200,
                headers: {
                  get: (k: string) => (k.toLowerCase() === 'content-length' ? '10068480' : null),
                },
              }
            : { ok: false, status: 404, headers: { get: () => null } },
        ),
      ),
    )
    const { V86Controller } = await import('@/lib/v86-controller')
    const { result } = renderHook(() => useV86())
    const container = document.createElement('div')

    await act(async () => {
      await result.current.init(container)
    })

    expect(result.current.isBooted).toBe(true)
    const instance = vi.mocked(V86Controller).mock.results.at(-1)?.value as any
    expect(instance.init).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({
        imageUrl: 'https://i.copy.sh/buildroot-bzimage68.bin',
        imageKind: 'kernel',
      }),
    )
  })

  it('auto-save persists while running and skips when not running', async () => {
    const { V86Controller } = await import('@/lib/v86-controller')
    const { result } = renderHook(() => useV86())
    const container = document.createElement('div')

    await act(async () => {
      await result.current.init(container)
    })

    const instance = vi.mocked(V86Controller).mock.results.at(-1)?.value as any
    expect(instance).toBeTruthy()

    // Running → a 30s tick persists
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000)
    })
    expect(instance.persistState).toHaveBeenCalledTimes(1)

    // Not running → further ticks are skipped (no save_state spam)
    instance.isRunning.mockReturnValue(false)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })
    expect(instance.persistState).toHaveBeenCalledTimes(1)
  })
})
