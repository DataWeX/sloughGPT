import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import React from 'react'

const mockGet = vi.hoisted(() => vi.fn())
const mockSetModelReadiness = vi.hoisted(() => vi.fn())

vi.mock('@/lib/bootstrap-controller', () => ({
  bootstrapController: { get: mockGet },
}))
vi.mock('@/lib/store', () => ({
  setModelReadiness: mockSetModelReadiness,
}))
vi.mock('@/lib/dev-log', () => ({
  logger: { child: () => ({ warning: vi.fn() }) },
}))

import { BootstrapProvider, useBootstrap } from './BootstrapContext'

function payload(phase: string) {
  return {
    members: { startup_progress: { phase, step: 1, total: 9, message: phase } },
    errors: {},
  }
}

function wrapper({ children }: { children: React.ReactNode }) {
  return <BootstrapProvider>{children}</BootstrapProvider>
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe('BootstrapContext', () => {
  it('useBootstrap outside the provider degrades to the null fallback', () => {
    const { result } = renderHook(() => useBootstrap())
    expect(result.current.data).toBeNull()
    expect(result.current.loading).toBe(true)
    expect(result.current.error).toBeNull()
  })

  it('fetches once on mount and exposes the payload', async () => {
    mockGet.mockResolvedValue(payload('ready'))
    const { result } = renderHook(() => useBootstrap(), { wrapper })
    expect(result.current.loading).toBe(true)
    await act(async () => {})
    expect(mockGet).toHaveBeenCalledTimes(1)
    expect(result.current.loading).toBe(false)
    expect(result.current.data?.members.startup_progress?.phase).toBe('ready')
  })

  it('feeds the startup_progress slice into the readiness store', async () => {
    mockGet.mockResolvedValue(payload('model_load'))
    const { result } = renderHook(() => useBootstrap(), { wrapper })
    await act(async () => {})
    expect(mockSetModelReadiness).toHaveBeenCalledWith({
      ready: false,
      phase: 'model_load',
      step: 1,
      total: 9,
      message: 'model_load',
    })
    expect(result.current.error).toBeNull()
  })

  it('re-fetches every 3s while the server reports a non-ready phase, then stops at ready', async () => {
    vi.useFakeTimers()
    mockGet.mockResolvedValueOnce(payload('model_load')).mockResolvedValueOnce(payload('ready'))
    const { result } = renderHook(() => useBootstrap(), { wrapper })
    await act(async () => {})
    expect(mockGet).toHaveBeenCalledTimes(1)
    expect(result.current.data?.members.startup_progress?.phase).toBe('model_load')

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3100)
    })
    expect(mockGet).toHaveBeenCalledTimes(2)
    expect(result.current.data?.members.startup_progress?.phase).toBe('ready')

    // Ready is terminal for the poll — no further fetches, ever.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30000)
    })
    expect(mockGet).toHaveBeenCalledTimes(2)
  })

  it('surfaces fetch failures without entering a retry loop', async () => {
    vi.useFakeTimers()
    mockGet.mockRejectedValue(new Error('offline'))
    const { result } = renderHook(() => useBootstrap(), { wrapper })
    await act(async () => {})
    expect(result.current.error).toBe('offline')
    expect(result.current.data).toBeNull()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(15000)
    })
    // Dead server: one attempt — the SSE health stream owns recovery UX.
    expect(mockGet).toHaveBeenCalledTimes(1)
  })

  it('refresh() re-fetches on demand', async () => {
    mockGet.mockResolvedValue(payload('ready'))
    const { result } = renderHook(() => useBootstrap(), { wrapper })
    await act(async () => {})
    await act(async () => {
      await result.current.refresh()
    })
    expect(mockGet).toHaveBeenCalledTimes(2)
  })
})
