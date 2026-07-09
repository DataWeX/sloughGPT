import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.mock('@/lib/system-controller', () => ({
  systemController: {
    streamOutput: vi.fn(),
  },
}))

import { systemController } from '@/lib/system-controller'
import { useServerOutput } from './useServerOutput'

describe('useServerOutput', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('returns empty lines initially', () => {
    ;(systemController.streamOutput as any).mockReturnValue({
      [Symbol.asyncIterator]: () => ({ next: () => Promise.resolve({ done: true }) }),
    })
    const { result } = renderHook(() => useServerOutput())
    expect(result.current.lines).toEqual([])
    expect(result.current.streaming).toBe(true)
    expect(result.current.paused).toBe(false)
  })

  it('provides clear function', () => {
    ;(systemController.streamOutput as any).mockReturnValue({
      [Symbol.asyncIterator]: () => ({ next: () => Promise.resolve({ done: true }) }),
    })
    const { result } = renderHook(() => useServerOutput())
    act(() => result.current.clear())
    expect(result.current.lines).toEqual([])
  })

  it('provides scrollRef', () => {
    ;(systemController.streamOutput as any).mockReturnValue({
      [Symbol.asyncIterator]: () => ({ next: () => Promise.resolve({ done: true }) }),
    })
    const { result } = renderHook(() => useServerOutput())
    expect(result.current.scrollRef).toBeDefined()
  })

  it('togglePause toggles paused state', () => {
    ;(systemController.streamOutput as any).mockReturnValue({
      [Symbol.asyncIterator]: () => ({ next: () => Promise.resolve({ done: true }) }),
    })
    const { result } = renderHook(() => useServerOutput())
    expect(result.current.paused).toBe(false)
    act(() => result.current.togglePause())
    expect(result.current.paused).toBe(true)
    act(() => result.current.togglePause())
    expect(result.current.paused).toBe(false)
  })

  it('exportLines is a function', () => {
    ;(systemController.streamOutput as any).mockReturnValue({
      [Symbol.asyncIterator]: () => ({ next: () => Promise.resolve({ done: true }) }),
    })
    const { result } = renderHook(() => useServerOutput())
    expect(typeof result.current.exportLines).toBe('function')
  })
})
